"""Scene mode: a whole multi-shot clip in, every shot with a chosen character swapped, the scene reassembled over
the original audio. The per-shot pipeline runs inside ComfyUI's core Start Loop / End Loop (List mode), so the
models load once outside the loop and each shot is one iteration."""
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

import folder_paths
from comfy_api.latest import InputImpl, io

from . import d1
from .starters import MY_VIDEO, STARTERS, ANIMAL_IDEAS, parse_swaps

Shot = io.Custom("CATSWAP_SHOT")
ShotResult = io.Custom("CATSWAP_RESULT")
Spans = io.Custom("CATSWAP_SPANS")


def ffmpeg_exe():
    """System ffmpeg, else the one imageio-ffmpeg ships (Comfy cloud workers have none)."""
    import shutil
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def has_video(path):
    import av
    try:
        with av.open(path) as c:
            return bool(c.streams.video)
    except Exception:
        return False


def fetch_starter(name):
    """The starter's clip, cached in input/catswap_starters/. A starter's `mirror` (a plain https mp4) is tried first:
    YouTube refuses datacenter IPs such as Comfy cloud workers. Otherwise yt-dlp at the LOWEST resolution."""
    s = STARTERS[name]
    vid = s["url"].rsplit("=", 1)[-1]
    out_dir = os.path.join(folder_paths.get_input_directory(), "catswap_starters")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{vid}.mp4")
    if os.path.exists(path) and has_video(path):
        return path
    mirror = os.environ.get("CATSWAP_STARTER_MIRROR", "").rstrip("/")
    for url in ([f"{mirror}/{vid}.mp4"] if mirror else []) + ([s["mirror"]] if s.get("mirror") else []):
        try:
            import urllib.request
            urllib.request.urlretrieve(url, path)
            if has_video(path):
                return path
        except Exception:
            pass
    subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--force-overwrites",
                    "--ffmpeg-location", ffmpeg_exe(),
                    # LOWEST resolution, https only: HLS (m3u8) downloads crawl
                    "-f", "worstvideo[height>=144][protocol=https]+worstaudio[protocol=https]"
                          "/worst[height>=144][vcodec!=none][protocol=https]",
                    "--merge-output-format", "mp4", "-o", path, s["url"]], check=True)
    if not has_video(path):
        os.remove(path)
        raise RuntimeError(f"{name}: the download has no video stream")
    return path


def cuts(images, threshold):
    """Frame indices where a new shot starts. The luma jump is divided by the frames' brightness, so a cut in a dark
    bar (jump ~8 at luma 14) and in a bright B&W film (jump ~45 at luma 72) both score ~0.3-0.6, while camera and
    actor motion stays under ~0.06. Must also be 8x the clip's median."""
    if len(images) < 3 or not threshold:
        return []
    small = torch.cat([F.interpolate(c[..., :3].mean(-1, keepdim=True).movedim(-1, 1), size=(48, 64), mode="area")
                       for c in images.split(64)])[:, 0] * 255
    luma = small.mean((1, 2))
    d = (small[1:] - small[:-1]).abs().mean((1, 2)) / ((luma[1:] + luma[:-1]) / 2 + 10)
    hit = torch.nonzero((d > threshold) & (d > 8 * d.median()), as_tuple=True)[0]
    return [int(i) + 1 for i in hit]


def pad_8n1(frames):
    n = len(frames)
    pad = (8 - (n - 1) % 8) % 8
    return torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)]) if pad else frames


class CatSwapStarter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapStarter",
            display_name="CatSwap Starter Scene",
            category="catswap",
            description="Pick a famous dialog scene to start from (fetched at the lowest resolution), or use your "
                        "own upload. Leave swaps empty to use the scene's suggestion.",
            inputs=[
                io.Combo.Input("starter", options=list(STARTERS) + [MY_VIDEO], default=list(STARTERS)[0]),
                io.String.Input("swaps", default="", multiline=True,
                                tooltip="One per line: who = animal. Describe what's visible (\"the bald man with "
                                        "glasses = cat\"). Empty = the starter's suggestion. Animal ideas: "
                                        + ", ".join(ANIMAL_IDEAS)),
                io.Video.Input("video", optional=True, tooltip=f"Used when starter is '{MY_VIDEO}'."),
            ],
            outputs=[io.Video.Output(display_name="video"), io.String.Output(display_name="swaps")],
        )

    @classmethod
    def execute(cls, starter, swaps, video=None) -> io.NodeOutput:
        if starter == MY_VIDEO:
            if video is None:
                raise ValueError(f"Starter is '{MY_VIDEO}' but no video is connected.")
            if not parse_swaps(swaps):
                raise ValueError("Say who becomes what, one per line, e.g. 'the man in the blue suit = cat'.")
            return io.NodeOutput(video, swaps)
        return io.NodeOutput(InputImpl.VideoFromFile(fetch_starter(starter)),
                             swaps if parse_swaps(swaps) else STARTERS[starter]["swaps"])


class CatSwapScenePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapScenePlan",
            display_name="CatSwap Scene Plan (d1)",
            category="catswap",
            description="Split the scene at its cuts, then ask the d1 decision model about every shot: which of the "
                        "characters is in it, where, and in what posture. Emits one item per shot to swap, for "
                        "Start Loop's List mode.",
            inputs=[
                io.Image.Input("images"),
                io.String.Input("swaps", force_input=True),
                io.Float.Input("cut_threshold", default=0.15, min=0.01, max=2.0,
                               tooltip="Brightness-normalized frame jump that counts as a cut."),
                io.Int.Input("min_shot_frames", default=6, min=1, max=100,
                             tooltip="Shorter 'shots' (flashes, whip pans) are merged into the previous shot."),
                io.Int.Input("max_shot_frames", default=161, min=9, max=1025,
                             tooltip="Longer shots are split into pieces this long (8n+1)."),
                io.Float.Input("min_present", default=0.8, min=0.0, max=1.0,
                               tooltip="d1's confidence that the character is in the shot."),
            ],
            outputs=[Shot.Output(display_name="shots", is_output_list=True), io.String.Output(display_name="plan"),
                     Spans.Output(display_name="spans", tooltip="[start, end) frames of every swapped shot.")],
        )

    @classmethod
    def execute(cls, images, swaps, cut_threshold, min_shot_frames, max_shot_frames, min_present) -> io.NodeOutput:
        pairs = parse_swaps(swaps)
        bounds = [0]
        for c in cuts(images, cut_threshold) + [len(images)]:
            if c - bounds[-1] >= min_shot_frames:  # flashes and fast whips aren't shots: merge them into the last one
                bounds.append(c)
        if len(bounds) == 1:
            bounds.append(len(images))
        bounds[-1] = len(images)
        pieces = []
        for a, b in zip(bounds, bounds[1:]):
            for s in range(a, b, max_shot_frames):
                pieces.append((s, min(b, s + max_shot_frames)))
        model = d1.load()
        shots, lines = [], []
        try:
            for i, (a, b) in enumerate(pieces):
                frame = images[(a + b) // 2]
                best = None
                for who, animal in pairs:
                    present, bbox, posture = d1.ask(model, frame, who)
                    if present >= min_present and bbox and (best is None or present > best["present"]):
                        best = {"start": a, "end": b, "who": who, "animal": animal, "bbox": bbox,
                                "posture": posture, "present": round(present, 2)}
                if best:
                    shots.append(best)
                lines.append(f"shot {i:2d}  frames {a:4d}-{b:4d}  " +
                             (f"{best['who']} -> {best['animal']}  ({best['present']}, {best['posture']})"
                              if best else "-"))
            voice = "shot"
            if shots:  # whose voice to silence: d1 on the frame where the character was clearest
                best = max(shots, key=lambda x: x["present"])
                woman = d1.ask_choice(model, images[(best["start"] + best["end"]) // 2],
                                      f"Is {best['who']} a man or a woman?", {"man": "A man", "woman": "A woman"})
                voice = "high" if woman == "woman" else "low"
        finally:
            d1.free(model)
        plan = f"{len(pieces)} shots, {len(shots)} to swap, muting the {voice} voice\n" + "\n".join(lines)
        return io.NodeOutput(shots, plan, {"spans": [[s["start"], s["end"]] for s in shots], "voice": voice})


class CatSwapShot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapShot",
            display_name="CatSwap Shot",
            category="catswap",
            description="One loop iteration's shot: its frames (padded to 8n+1), first frame, and d1's decision.",
            inputs=[io.Image.Input("images"), Shot.Input("shot")],
            outputs=[io.Image.Output(display_name="frames"), io.Image.Output(display_name="first"),
                     io.Int.Output(display_name="frame_count"), io.String.Output(display_name="who"),
                     io.String.Output(display_name="animal"), io.String.Output(display_name="posture"),
                     io.BoundingBox.Output(display_name="bbox")],
        )

    @classmethod
    def execute(cls, images, shot) -> io.NodeOutput:
        frames = pad_8n1(images[shot["start"]:shot["end"]])
        return io.NodeOutput(frames, frames[:1], len(frames), shot["who"], shot["animal"], shot["posture"],
                             shot["bbox"])


class CatSwapShotResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapShotResult",
            display_name="CatSwap Shot Result",
            category="catswap",
            inputs=[Shot.Input("shot"), io.Image.Input("images")],
            outputs=[ShotResult.Output(display_name="result")],
        )

    @classmethod
    def execute(cls, shot, images) -> io.NodeOutput:
        n = shot["end"] - shot["start"]
        out = images[:n]
        if len(out) < n:  # never let the original person show through; hold the last swapped frame
            out = torch.cat([out, out[-1:].expand(n - len(out), -1, -1, -1)])
        return io.NodeOutput({"start": shot["start"], "end": shot["end"], "images": out.cpu()})


class CatSwapAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapAssemble",
            display_name="CatSwap Assemble Scene",
            category="catswap",
            description="Put every swapped shot back into the original scene; untouched shots stay original.",
            is_input_list=True,
            inputs=[io.Image.Input("images"), ShotResult.Input("results", optional=True)],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, results=None) -> io.NodeOutput:
        out = images[0].clone()
        for r in results or []:
            if r is not None:
                out[r["start"]:r["end"]] = r["images"].to(out)
        return io.NodeOutput(out)


class CatSwapMuteVoice(io.ComfyNode):
    """Animals don't talk. Splits the soundtrack with HDemucs (torchaudio) into vocals and everything else, and fades
    the vocals out during the swapped shots: in shot/reverse-shot dialog the person on screen is the one talking.
    Music, room tone and everyone else's lines stay."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapMuteVoice",
            display_name="CatSwap Mute Voice",
            category="catswap",
            description="Silence the animal's lines: vocals are separated (HDemucs) and faded out during the swapped "
                        "shots; music, ambience and the other actors' lines in other shots are untouched.",
            inputs=[io.Audio.Input("audio"), Spans.Input("spans"),
                    io.Float.Input("fps", default=12.0, min=1.0, max=240.0, force_input=True),
                    io.Float.Input("fade", default=0.08, min=0.0, max=1.0, tooltip="Seconds of fade at each edge."),
                    io.Boolean.Input("enabled", default=True)],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, audio, spans, fps, fade, enabled) -> io.NodeOutput:
        voice = spans.get("voice", "shot") if isinstance(spans, dict) else "shot"
        spans = spans.get("spans", []) if isinstance(spans, dict) else spans
        if not enabled or audio is None or not spans:
            return io.NodeOutput(audio)
        import torchaudio
        import comfy.model_management as mm
        wav, sr = audio["waveform"][0].float(), audio["sample_rate"]  # [C, T]
        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        x = torchaudio.functional.resample(wav, sr, bundle.sample_rate) if sr != bundle.sample_rate else wav
        x = x.expand(2, -1) if x.shape[0] == 1 else x[:2]
        device = mm.get_torch_device()
        model = bundle.get_model().to(device).eval()
        try:
            vocals = separate_vocals(model, x, bundle.sample_rate, device)
        finally:
            del model
            mm.soft_empty_cache()
        if sr != bundle.sample_rate:
            vocals = torchaudio.functional.resample(vocals, bundle.sample_rate, sr)
        vocals = vocals[:, :wav.shape[1]]
        if wav.shape[0] == 1:
            vocals = vocals.mean(0, keepdim=True)
        n = wav.shape[1]
        gain = torch.ones(n)
        f = max(1, int(fade * sr))
        for a, b in spans:
            s0, s1 = int(a / fps * sr), min(n, int(b / fps * sr))
            if s1 <= s0:
                continue
            gain[s0:s1] = 0
            lo, hi = max(0, s0 - f), min(n, s1 + f)
            gain[lo:s0] = torch.minimum(gain[lo:s0], torch.linspace(1, 0, s0 - lo))
            gain[s1:hi] = torch.minimum(gain[s1:hi], torch.linspace(0, 1, hi - s1))
        if voice in ("high", "low"):  # two people in one shot: only mute the swapped character's side of ~165 Hz
            gain = torch.maximum(gain, 1 - voice_mask(vocals, sr, voice, f))
        out = wav - vocals * (1 - gain)  # = everything else + vocals * gain
        return io.NodeOutput({**audio, "waveform": out[None].to(audio["waveform"].dtype)})


def voice_mask(vocals, sr, voice, fade, split_hz=165.0):
    """1 where the separated vocals are voiced speech on the given side of split_hz (high = typically a woman's
    voice, low = a man's), smoothed over ~0.3 s; 0 elsewhere. Same length as the audio."""
    import torchaudio
    mono = torchaudio.functional.resample(vocals.mean(0), sr, 16000)
    hop = 160  # 10 ms
    pitch = torchaudio.functional.detect_pitch_frequency(mono[None], 16000, frame_time=0.01, win_length=30,
                                                         freq_low=70, freq_high=400)[0]
    frames = mono[: len(mono) // hop * hop].reshape(-1, hop)
    energy = frames.pow(2).mean(1).sqrt()
    n = min(len(pitch), len(energy))
    pitch, energy = pitch[:n], energy[:n]
    voiced = energy > max(1e-4, 0.15 * float(energy.max()))
    side = pitch > split_hz if voice == "high" else pitch <= split_hz
    m = (voiced & side).float()
    k = 31  # ~0.3 s majority vote, then grow a little so syllable edges are covered
    m = (F.avg_pool1d(m[None, None], k, stride=1, padding=k // 2)[0, 0] > 0.5).float()
    m = F.max_pool1d(m[None, None], 11, stride=1, padding=5)[0, 0]
    k = 2 * max(1, int(fade / sr / 0.01) // 2) + 1  # soft edges, smoothed at the 10 ms frame rate
    m = F.avg_pool1d(m[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]
    return F.interpolate(m[None, None], size=vocals.shape[-1], mode="linear")[0, 0].clamp(0, 1)


def separate_vocals(model, x, sr, device, segment=10.0, overlap=0.1):
    """torchaudio's chunked HDemucs recipe: normalize, separate in overlapping segments with linear fades."""
    from torchaudio.transforms import Fade
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    x = ((x - mean) / std)[None].to(device)
    length = x.shape[-1]
    chunk = int(sr * segment * (1 + overlap))
    ov = int(overlap * sr)
    fade = Fade(fade_in_len=0, fade_out_len=ov, fade_shape="linear")
    out = torch.zeros(1, len(model.sources), 2, length, device=device)
    start, end = 0, chunk
    with torch.inference_mode():
        while start < length - ov:
            piece = x[:, :, start:end]
            y = fade(model(piece))
            out[:, :, :, start:start + y.shape[-1]] += y
            if start == 0:
                fade.fade_in_len = ov
                start += int(chunk - ov)
            else:
                start += chunk
            end += chunk
            if end >= length:
                fade.fade_out_len = 0
    vocals = out[0, model.sources.index("vocals")]
    return (vocals * std + mean).cpu()


NODES = [CatSwapStarter, CatSwapScenePlan, CatSwapShot, CatSwapShotResult, CatSwapAssemble, CatSwapMuteVoice]
