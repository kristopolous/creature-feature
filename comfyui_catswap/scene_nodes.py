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


def fetch_starter(name):
    s = STARTERS[name]
    vid = s["url"].rsplit("=", 1)[-1]
    out_dir = os.path.join(folder_paths.get_input_directory(), "catswap_starters")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{vid}.mp4")
    if not os.path.exists(path):
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--force-overwrites",
                        "-f", "worstvideo[height>=144]+worstaudio/worst[height>=144][vcodec!=none]",  # LOWEST res
                        "--merge-output-format", "mp4",
                        "-o", path, s["url"]], check=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index",
                                "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
        if not probe:
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
            outputs=[Shot.Output(display_name="shots", is_output_list=True), io.String.Output(display_name="plan")],
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
        finally:
            d1.free(model)
        plan = f"{len(pieces)} shots, {len(shots)} to swap\n" + "\n".join(lines)
        return io.NodeOutput(shots, plan)


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


NODES = [CatSwapStarter, CatSwapScenePlan, CatSwapShot, CatSwapShotResult, CatSwapAssemble]
