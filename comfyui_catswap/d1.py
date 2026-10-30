"""LiquidAI d1-2B, an open decision model: typed answers (yes/no, choice, box) in one forward pass, no text
generation. Loaded on demand, used, and freed, because the apps also hold LTX, Gemma, Klein and SAM3."""
import gc

import numpy as np
import torch
from PIL import Image

MODEL = "LiquidAI/d1-2B"


def load():
    import shutil
    import transformers.utils
    import transformers.utils.import_utils
    if not (shutil.which("cc") or shutil.which("gcc")):  # Comfy cloud workers: no compiler, so Triton JIT fails
        transformers.utils.is_triton_available = transformers.utils.import_utils.is_triton_available = lambda: False
    from transformers import AutoModel
    import comfy.model_management as mm
    device = mm.get_torch_device()
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    try:  # cached copy first: the hub check can hang for minutes on a bad network
        m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype, local_files_only=True)
    except Exception:
        m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype)
    return m.to(device).eval()


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def questions(who):
    return {
        "present": {"type": "noul", "instructions": f"Is {who} visible in this image?"},
        "where": {"type": "box", "instructions": f"Where is {who}? Box their whole visible body."},
        "posture": {"type": "choice", "instructions": f"What is {who} doing with their body?",
                    "criteria": {"standing": "Standing still or talking on their feet",
                                 "sitting": "Sitting on a chair, sofa, bench or in a car",
                                 "walking": "Walking or moving through the scene",
                                 "lying": "Lying down"}},
    }


def to_pil(image):
    """One IMAGE frame [H, W, C] in 0..1 -> PIL."""
    return Image.fromarray((image[..., :3].float().cpu().numpy() * 255).round().astype(np.uint8))


def ask(model, frame, who):
    """-> (present 0..1, bbox {x, y, width, height} in pixels or None, posture)."""
    h, w = frame.shape[0], frame.shape[1]
    with torch.inference_mode():
        ans = model.system_one(None, questions(who), media=[to_pil(frame)])["answers"]
    b = ans.get("where", {}).get("box")
    bbox = None if not b else {"x": round(b["left"] * w), "y": round(b["top"] * h),
                               "width": max(1, round((b["right"] - b["left"]) * w)),
                               "height": max(1, round((b["bottom"] - b["top"]) * h))}
    return float(ans["present"]["noul"]), bbox, ans["posture"]["choice"]


def ask_choice(model, frame, question, criteria):
    with torch.inference_mode():
        ans = model.system_one(None, {"q": {"type": "choice", "instructions": question, "criteria": criteria}},
                               media=[to_pil(frame)])["answers"]
    return ans["q"]["choice"]
