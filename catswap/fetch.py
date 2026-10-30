"""Fetch a video section at the LOWEST resolution and normalize it for the pipeline.

    python -m catswap.fetch URL --start 0:30 --end 0:40 --out work/clip

Writes work/clip/clip.mp4: a short height, a width rounded to a multiple of 32, and a low fps.
Videos are never committed to the repo; they are fetched at runtime from a URL.
"""
import argparse
import json
import subprocess
from pathlib import Path

# LOWEST RESOLUTION. Raise these only for an explicit final render.
HEIGHT = 192
FPS = 12
MULT = 32


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]


def fetch(url, out, start=None, end=None, height=HEIGHT, fps=FPS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.mp4"
    cmd = ["yt-dlp", "-f", "worstvideo[height>=144]+worstaudio/worst[height>=144]/worst",
           "--merge-output-format", "mp4", "--quiet", "--no-warnings", "-o", str(raw), "--force-overwrites"]
    if start or end:
        cmd += ["--download-sections", f"*{start or '0'}-{end or 'inf'}"]
    run(cmd + [url])

    w, h = probe(raw)
    width = max(MULT, round(w * height / h / MULT) * MULT)
    clip = out / "clip.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", f"fps={fps},scale={width}:{height}:flags=area",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)])
    meta = {"url": url, "start": start, "end": end, "width": width, "height": height, "fps": fps}
    (out / "clip.json").write_text(json.dumps(meta, indent=2))
    return clip, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="work/clip")
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args()
    print(fetch(a.url, a.out, a.start, a.end, a.height, a.fps))
