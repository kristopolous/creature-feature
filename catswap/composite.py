"""Frame plumbing: extract frames, build the regenerated region, paste results back over the original,
blend window overlaps, and mux the ORIGINAL audio (the dialog) back on. CPU only, LOWEST resolution."""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def extract(clip, start, end, out_dir):
    """Frames [start, end) of the clip as PNGs; returns their paths in order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip),
                    "-vf", f"select=between(n\\,{start}\\,{end - 1})", "-fps_mode", "passthrough",
                    str(out_dir / "f%05d.png")], check=True)
    return sorted(out_dir.glob("f*.png"))


def encode(frames, path, fps):
    """PNG frames -> lossless-ish mp4 for LoadVideo."""
    lst = Path(path).with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(f).resolve()}'\nduration {1 / fps}\n" for f in frames))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-r", str(fps), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)], check=True)
    return Path(path)


def load(path, mode="RGB"):
    return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0


def save(arr, path):
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(path)
    return Path(path)


def bbox(mask, thresh=0.5):
    """Pixel mask -> FLUX 3 grid box [y0, x0, y1, x1] on 0..1000, or None if empty."""
    ys, xs = np.nonzero(mask > thresh)
    if not len(ys):
        return None
    h, w = mask.shape
    return [round(ys.min() * 1000 / h), round(xs.min() * 1000 / w),
            round((ys.max() + 1) * 1000 / h), round((xs.max() + 1) * 1000 / w)]


def box_mask(box, h, w):
    m = np.zeros((h, w), np.float32)
    y0, x0, y1, x1 = box
    m[int(y0 * h / 1000):int(np.ceil(y1 * h / 1000)), int(x0 * w / 1000):int(np.ceil(x1 * w / 1000))] = 1
    return m


def region(person_masks, animal_box, grow=6):
    """The area LTX regenerates for a window: everywhere the person went (so they're erased) plus where
    the animal sits, grown a little. Static over the window, so the animal is free to move inside it."""
    h, w = person_masks[0].shape
    m = np.maximum(np.max(np.stack(person_masks), 0), box_mask(animal_box, h, w))
    img = Image.fromarray((m > 0.5).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * grow + 1))
    return np.asarray(img, np.float32) / 255.0


def feather(mask, radius=2):
    img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    return (np.asarray(img, np.float32) / 255.0)[..., None]


def paste(original, generated, alpha):
    return original * (1 - alpha) + generated * alpha


def mux(frames, clip, start, end, fps, out):
    """Write the swapped frames with the clip's original audio for the same span."""
    video = encode(frames, Path(out).with_suffix(".noaudio.mp4"), fps)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
                    "-ss", f"{start / fps:.4f}", "-to", f"{end / fps:.4f}", "-i", str(clip),
                    "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    return Path(out)
