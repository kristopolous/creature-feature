"""ComfyUI-CatSwap: swap one character in a video for a real-behaving animal. Everything else, including the
dialog audio, stays the original footage."""
import json

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class CameoBoxNoiseMask(io.ComfyNode):
    """Regenerate only a box of a video latent; everything else stays the original footage."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CameoBoxNoiseMask",
            display_name="Cameo Box Noise Mask",
            category="catswap",
            description="Set a video latent's noise_mask to 1 inside a box (FLUX 3 grid: [y_min, x_min, y_max, x_max] "
                        "on 0-1000) and 0 elsewhere, so the sampler only regenerates that region. "
                        "The first keep_frames latent frames stay fully frozen (e.g. a pre-edited first frame).",
            inputs=[
                io.Latent.Input("latent", tooltip="5D video latent [B, C, F, H, W]."),
                io.String.Input("boxes", default="[[0, 0, 1000, 1000]]",
                                tooltip="JSON: one box for every latent frame, or a single box for all of them."),
                io.Int.Input("dilate", default=1, min=0, max=8, tooltip="Grow the box by this many latent cells."),
                io.Int.Input("keep_frames", default=1, min=0, max=64, tooltip="Leading latent frames to freeze."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, boxes, dilate, keep_frames) -> io.NodeOutput:
        samples = latent["samples"]
        if samples.ndim != 5:
            raise ValueError(f"Expected a 5D video latent, got {list(samples.shape)}")
        b, _, f, h, w = samples.shape
        boxes = json.loads(boxes)
        if len(boxes) == 1:
            boxes = boxes * f
        if len(boxes) != f:
            raise ValueError(f"Got {len(boxes)} boxes for {f} latent frames")
        mask = torch.zeros((b, 1, f, h, w), dtype=torch.float32)
        for t, (y0, x0, y1, x1) in enumerate(boxes):
            if t < keep_frames:
                continue
            r0 = max(0, int(y0 * h / 1000) - dilate)
            r1 = min(h, -(-int(y1 * h) // 1000) + dilate)
            c0 = max(0, int(x0 * w / 1000) - dilate)
            c1 = min(w, -(-int(x1 * w) // 1000) + dilate)
            mask[:, :, t, r0:r1, c0:c1] = 1.0
        out = latent.copy()
        out["noise_mask"] = mask
        return io.NodeOutput(out)


class CatSwapMaskNoiseMask(io.ComfyNode):
    """Regenerate only where a pixel mask (e.g. a SAM3 track of the swapped person) is set."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapMaskNoiseMask",
            display_name="CatSwap Mask Noise Mask",
            category="catswap",
            description="Downsample a pixel-space MASK (one frame, or one per video frame) onto a causal video "
                        "latent's grid (latent frame 0 = pixel frame 0, latent frame k = pixel frames 8k-7..8k), "
                        "max-pooled so any touched cell is regenerated, then dilated. The first keep_frames latent "
                        "frames stay frozen.",
            inputs=[
                io.Latent.Input("latent", tooltip="5D video latent [B, C, F, H, W]."),
                io.Mask.Input("mask", tooltip="[T, H, W] pixel mask; T = 1 (static) or the window's frame count."),
                io.Int.Input("temporal_scale", default=8, min=1, max=16, tooltip="Pixel frames per latent frame."),
                io.Int.Input("dilate", default=1, min=0, max=8, tooltip="Grow the mask by this many latent cells."),
                io.Int.Input("keep_frames", default=1, min=0, max=64, tooltip="Leading latent frames to freeze."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, mask, temporal_scale, dilate, keep_frames) -> io.NodeOutput:
        samples = latent["samples"]
        if samples.ndim != 5:
            raise ValueError(f"Expected a 5D video latent, got {list(samples.shape)}")
        b, _, f, h, w = samples.shape
        m = mask.float().reshape(-1, mask.shape[-2], mask.shape[-1])
        # pixel frames -> latent frames (causal VAE: first frame alone, then groups of temporal_scale)
        if m.shape[0] == 1:
            per = m.expand(f, -1, -1)
        else:
            groups = [m[:1]] + [m[1 + (k - 1) * temporal_scale: 1 + k * temporal_scale] for k in range(1, f)]
            per = torch.stack([g.amax(0) if len(g) else m[-1] for g in groups])
        cells = F.adaptive_max_pool2d(per[:, None], (h, w))
        if dilate:
            cells = F.max_pool2d(cells, 2 * dilate + 1, stride=1, padding=dilate)
        cells = (cells > 0.5).float()[:, 0]
        cells[:keep_frames] = 0
        out = latent.copy()
        out["noise_mask"] = cells[None, None].expand(b, 1, f, h, w).to(samples.device).clone()
        return io.NodeOutput(out)


from . import d1, place  # noqa: E402  (place.py is shared with the CLI via a symlink)
from .scene_nodes import NODES as SCENE_NODES  # noqa: E402

POSTURES = ["standing", "sitting", "walking", "lying"]


def _grid_box(mask, thresh=0.5):
    ys, xs = torch.nonzero(mask > thresh, as_tuple=True)
    if not len(ys):
        return None
    h, w = mask.shape
    return [round(ys.min().item() * 1000 / h), round(xs.min().item() * 1000 / w),
            round((ys.max().item() + 1) * 1000 / h), round((xs.max().item() + 1) * 1000 / w)]


class CatSwapFindCharacter(io.ComfyNode):
    """d1 decision model: find the described character, say whether they are there, and read their posture."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapFindCharacter",
            display_name="CatSwap Find Character (d1)",
            category="catswap",
            description="Runs LiquidAI d1-2B, an open decision model, in one forward pass on the first frame: is the "
                        "described character there, where (box), and what posture. No text generation.",
            inputs=[
                io.Image.Input("image", tooltip="Frame to look at; only the first image of the batch is used."),
                io.String.Input("who", default="the man in the brown suit", tooltip="Who becomes the animal."),
                io.Combo.Input("posture_override", options=["auto"] + POSTURES, default="auto"),
            ],
            outputs=[io.BoundingBox.Output(display_name="bbox"), io.String.Output(display_name="posture"),
                     io.Float.Output(display_name="present"), io.String.Output(display_name="who")],
        )

    @classmethod
    def execute(cls, image, who, posture_override) -> io.NodeOutput:
        model = d1.load()
        try:
            present, bbox, posture = d1.ask(model, image[0], who)
        finally:
            d1.free(model)
        h, w = image.shape[1], image.shape[2]
        bbox = bbox or {"x": 0, "y": 0, "width": w, "height": h}
        posture = posture if posture_override == "auto" else posture_override
        return io.NodeOutput(bbox, posture, present, who)


class CatSwapFitFrames(io.ComfyNode):
    """LOWEST resolution, enforced in the graph: any uploaded video becomes ~192 px tall, ~12 fps, 8n+1 frames."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapFitFrames",
            display_name="CatSwap Fit Frames (lowest res, 8n+1)",
            category="catswap",
            description="Scale to `height` (width rounded to a multiple of 32), keep every Nth frame to get near "
                        "target_fps (duration is preserved, so the original audio stays in sync), cap at the video "
                        "model's 8n+1 frame count, and trim the audio to the same length.",
            inputs=[io.Image.Input("images"), io.Float.Input("fps", default=24.0, min=1.0, max=240.0, force_input=True),
                    io.Audio.Input("audio", optional=True),
                    io.Int.Input("height", default=192, min=64, max=2160, step=32),
                    io.Float.Input("target_fps", default=12.0, min=1.0, max=60.0),
                    io.Int.Input("max_frames", default=161, min=9, max=1025, tooltip="Upper bound (8n+1)."),
                    io.Float.Input("cut_threshold", default=12.0, min=0.0, max=255.0,
                                   tooltip="Stop at the first camera cut: a jump in mean luma difference above this "
                                           "(0-255 scale) and 8x the clip's median. 0 = never cut.")],
            outputs=[io.Image.Output(display_name="images"), io.Int.Output(display_name="frames"),
                     io.Float.Output(display_name="fps"), io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, images, fps, height, target_fps, max_frames, cut_threshold=12.0, audio=None) -> io.NodeOutput:
        stride = max(1, round(fps / target_fps))
        imgs = images[::stride]
        fps = fps / stride
        n = min(len(imgs), max_frames)
        if cut_threshold and n > 2:
            # the video model can't carry a person-shaped region across a cut; one shot per run
            small = F.interpolate(imgs[:n, ..., :3].mean(-1, keepdim=True).movedim(-1, 1), size=(48, 64),
                                  mode="area")[:, 0] * 255
            d = (small[1:] - small[:-1]).abs().mean((1, 2))
            cuts = torch.nonzero((d > cut_threshold) & (d > 8 * d.median()), as_tuple=True)[0]
            if len(cuts):
                n = int(cuts[0]) + 1
        n = max(1, n - (n - 1) % 8)
        imgs = imgs[:n]
        h, w = imgs.shape[1:3]
        width = max(32, round(w * height / h / 32) * 32)
        if (h, w) != (height, width):
            imgs = torch.cat([F.interpolate(c.movedim(-1, 1), size=(height, width), mode="area").movedim(1, -1)
                              for c in imgs.split(16)])
        if audio is not None:
            sr = audio["sample_rate"]
            audio = {**audio, "waveform": audio["waveform"][..., :round(n / fps * sr)]}
        return io.NodeOutput(imgs, n, float(fps), audio)


class CatSwapPlan(io.ComfyNode):
    """From the tracked person: where the animal goes, the Klein swap record, the LTX prompt and the region."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapPlan",
            display_name="CatSwap Plan",
            category="catswap",
            description="Takes the SAM3 track of the person. Puts a real-size animal on their footprint (floor, seat, "
                        "or rising into frame if their feet are cut off), writes the FLUX 3 Klein i2i_bbox swap "
                        "record, the LTX motion prompt (the animal does NOT copy the actor), and the static region "
                        "LTX regenerates: everywhere the person went plus the animal's box.",
            inputs=[
                io.Mask.Input("masks", tooltip="[T, H, W] track of the person (SAM3 Track to Mask)."),
                io.String.Input("who", default="the person", tooltip="Who is being removed (for the removal pass)."),
                io.String.Input("animal", default="cat", tooltip="Any animal: cat, monkey, rabbit, camel, a fluffy "
                                "white goat... FLUX 3 Klein draws whatever is named."),
                io.Float.Input("size", default=0.0, min=0.0, max=4.0, step=0.01,
                               tooltip="Animal height as a fraction of a standing adult. 0 = look it up "
                                       "(rabbit 0.15, cat 0.22, goat 0.5, horse 1.0, camel 1.15)."),
                io.String.Input("posture", default="standing", tooltip="standing / sitting / walking / lying"),
                io.String.Input("scene", default="", multiline=True, tooltip="Optional scene line for LTX."),
                io.Int.Input("grow", default=6, min=0, max=64, tooltip="Grow the region by this many pixels."),
            ],
            outputs=[io.String.Output(display_name="remove_prompt"), io.String.Output(display_name="add_prompt"),
                     io.String.Output(display_name="ltx_prompt"), io.Mask.Output(display_name="region"),
                     io.String.Output(display_name="plan")],
        )

    @classmethod
    def execute(cls, masks, who, animal, size, posture, scene, grow) -> io.NodeOutput:
        m = masks.float().reshape(-1, masks.shape[-2], masks.shape[-1])
        h, w = m.shape[-2:]
        person = _grid_box(m[0]) or _grid_box(m.amax(0))
        if person is None:
            raise ValueError("The character track is empty: SAM3 did not find them on the first frame.")
        posture = posture if posture in POSTURES else "standing"
        touches = person[2] >= 985
        abox = place.animal_box(person, posture, w / h, touches, animal, size)
        region = m.amax(0)
        y0, x0, y1, x1 = abox
        region[int(y0 * h / 1000):-(-y1 * h // 1000), int(x0 * w / 1000):-(-x1 * w // 1000)] = 1
        region = (region > 0.5).float()
        if grow:
            region = F.max_pool2d(region[None, None], 2 * grow + 1, stride=1, padding=grow)[0, 0]
        remove = place.remove_record(person, who)
        add = place.add_record(abox, animal, posture, touches)
        ltx = place.motion_prompt(animal, posture, scene, touches)
        plan = json.dumps({"person": person, "animal_box": abox, "posture": posture})
        return io.NodeOutput(remove, add, ltx, region[None], plan)


class CatSwapSetFirstFrame(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapSetFirstFrame",
            display_name="CatSwap Set First Frame",
            category="catswap",
            description="Replace frame 0 of a batch with an image (resized to fit), e.g. the Klein-swapped frame.",
            inputs=[io.Image.Input("images"), io.Image.Input("first")],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, first) -> io.NodeOutput:
        f = first[:1].movedim(-1, 1)
        f = F.interpolate(f, size=images.shape[1:3], mode="area").movedim(1, -1)
        out = images.clone()
        out[0] = f[0, ..., :images.shape[-1]].to(out)
        return io.NodeOutput(out)


class CatSwapComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CatSwapComposite",
            display_name="CatSwap Composite",
            category="catswap",
            description="Paste the generated frames over the ORIGINAL frames inside the (feathered) region only. "
                        "The video VAE shifts pixels everywhere, so outside the region the original always wins.",
            inputs=[io.Image.Input("original"), io.Image.Input("generated"), io.Mask.Input("region"),
                    io.Int.Input("feather", default=2, min=0, max=32)],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, original, generated, region, feather) -> io.NodeOutput:
        n = min(len(original), len(generated))
        a = region.float().reshape(-1, region.shape[-2], region.shape[-1])[:1, None]
        if feather:
            k = 2 * feather + 1
            a = F.avg_pool2d(F.pad(a, [feather] * 4, mode="replicate"), k, stride=1)
        a = F.interpolate(a, size=original.shape[1:3], mode="bilinear")[0, 0][None, ..., None].to(original)
        gen = generated[:n]
        if gen.shape[1:3] != original.shape[1:3]:
            gen = F.interpolate(gen.movedim(-1, 1), size=original.shape[1:3], mode="bilinear").movedim(1, -1)
        return io.NodeOutput(original[:n] * (1 - a) + gen.to(original) * a)


class CatSwapExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CameoBoxNoiseMask, CatSwapMaskNoiseMask, CatSwapFindCharacter, CatSwapFitFrames, CatSwapPlan,
                CatSwapSetFirstFrame, CatSwapComposite, *SCENE_NODES]


async def comfy_entrypoint() -> CatSwapExtension:
    return CatSwapExtension()
