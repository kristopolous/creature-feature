"""Per-shot decisions with an open decision model (LiquidAI d1-2B): is the chosen character here,
where are they, and what are they doing (so the animal gets a matching non-human action).

    python -m catswap.decide work/test/shots.json --who "the man in the brown suit"

One forward pass per shot answers every question about its keyframe; no tokens are generated.
Adds a "decision" block to each shot in shots.json.
"""
import argparse
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are cached; the hub check otherwise hangs for minutes
import json
import time

import torch
from PIL import Image
from transformers import AutoModel

MODEL = "LiquidAI/d1-2B"


def questions(who):
    return {
        "present": {"type": "noul", "instructions": f"Is {who} visible in this image?"},
        "where": {"type": "box", "instructions": f"Where is {who}? Box their whole visible body."},
        "posture": {"type": "choice", "instructions": f"What is {who} doing with their body?",
                    "criteria": {"standing": "Standing still or talking on their feet",
                                 "sitting": "Sitting on a chair, sofa, bench or in a car",
                                 "walking": "Walking or moving through the scene",
                                 "lying": "Lying down"}},
        "framing": {"type": "choice", "instructions": f"How is {who} framed?",
                    "criteria": {"wide": "Their full body is visible",
                                 "medium": "From about the waist up",
                                 "closeup": "Their face fills most of the frame"}},
    }


def load(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype).to(device)


def decide(model, image_path, who):
    image = Image.open(image_path).convert("RGB")
    return model.system_one(None, questions(who), media=[image])["answers"]


def swap(answers, min_present=0.8):
    """Swap this shot? The character must be there and boxed. Close-ups are kept: a cat face where
    the actor's face was, mid-monologue, is the best shot of the lot."""
    return answers["present"]["noul"] >= min_present and answers.get("where", {}).get("box") is not None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shots")
    ap.add_argument("--who", required=True, help='the character to swap, e.g. "the man in the brown suit"')
    a = ap.parse_args()
    data = json.load(open(a.shots))
    t = time.time()
    model = load()
    print(f"loaded in {time.time() - t:.1f}s")
    data["who"] = a.who
    for shot in data["shots"]:
        t = time.time()
        ans = decide(model, shot["keyframe"], a.who)
        shot["decision"] = {"answers": ans, "swap": swap(ans), "seconds": round(time.time() - t, 2)}
        print(shot["id"], json.dumps(shot["decision"], default=str)[:400])
    json.dump(data, open(a.shots, "w"), indent=2, default=str)
