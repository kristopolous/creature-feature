"""Split a clip into shots, then split long shots into overlapping windows the video model can handle.

    python -m catswap.shots work/clip/clip.mp4

Writes shots.json next to the clip, plus one small keyframe per shot (the first frame of the shot).
Shot cuts are natural chunk boundaries: no continuity is needed across a cut.
"""
import argparse
import json
import subprocess
from pathlib import Path

from scenedetect import ContentDetector, detect

# Video-model window. LTX-style frame counts are 8n+1. Keep windows short (LOWEST cost).
WINDOW = 25
OVERLAP = 8


def windows(start, end, window=WINDOW, overlap=OVERLAP):
    """[start, end) frame ranges, each at most `window` long, consecutive ones sharing `overlap` frames."""
    out, s = [], start
    while True:
        e = min(s + window, end)
        out.append([s, e])
        if e >= end:
            return out
        s = e - overlap


def split(clip, threshold=15.0, min_len=6):
    clip = Path(clip)
    meta = json.loads((clip.parent / "clip.json").read_text())
    # luma only: B&W films have no hue/saturation, so the default HSV score misses their cuts
    scenes = detect(str(clip), ContentDetector(threshold=threshold, min_scene_len=min_len, luma_only=True))
    if not scenes:
        n = int(subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(clip)],
            check=True, capture_output=True, text=True).stdout.strip())
        bounds = [(0, n)]
    else:
        bounds = [(a.frame_num, b.frame_num) for a, b in scenes]

    keydir = clip.parent / "keyframes"
    keydir.mkdir(exist_ok=True)
    shots = []
    for i, (a, b) in enumerate(bounds):
        key = keydir / f"shot{i:04d}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip),
                        "-vf", f"select=eq(n\\,{a})", "-frames:v", "1", str(key)], check=True)
        shots.append({"id": i, "start": a, "end": b, "frames": b - a,
                      "keyframe": str(key), "windows": windows(a, b)})
    out = {"clip": str(clip), **meta, "shots": shots}
    (clip.parent / "shots.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--threshold", type=float, default=15.0)
    a = ap.parse_args()
    r = split(a.clip, a.threshold)
    print(f"{len(r['shots'])} shots, {sum(len(s['windows']) for s in r['shots'])} windows")
