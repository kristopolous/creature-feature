"""Swap one character in a clip for an animal. Everything else, including the dialog audio, stays.

    python -m catswap.run work/test/clip.mp4 --who "the man in the brown suit" --animal cat
    python -m catswap.run URL --start 1:02 --end 1:10 --who "the woman on the left" --shots 0
    python -m catswap.run work/test/clip.mp4 --who "the man" --point 120,90   # click on shot 0's first frame

Needs a ComfyUI with comfyui_catswap installed at COMFY_URL (default: tunnel on 127.0.0.1:8188).
Every size is the clip's size: LOWEST resolution (fetch.py makes it 192 px tall).
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from . import comfy, composite as cx, graphs, place, shots as shotlib

LTX_FRAMES = lambda n: n if (n - 1) % 8 == 0 else n + (8 - (n - 1) % 8)  # pad to 8k+1


def grid_to_xywh(box, w, h):
    y0, x0, y1, x1 = box
    return [x0 * w / 1000, y0 * h / 1000, (x1 - x0) * w / 1000, (y1 - y0) * h / 1000]


def track(shot_video, w, h, points=None, seed_box=None, out=None):
    name = comfy.upload(shot_video)
    g = graphs.sam3_track(name, points=points, box=grid_to_xywh(seed_box, w, h) if seed_box and not points else None)
    paths, secs = comfy.run(g, out)
    print(f"  sam3 track: {len(paths)} masks in {secs}s")
    return [cx.load(p, "L") for p in sorted(paths) if p.suffix == ".png"]


# The ONE exception to the lowest-resolution rule: the single first-frame Klein edit runs at 2x, because at
# 192 px Klein ignores small boxes (a 32 px cat box came out 3x too big and off to the side). ~2 s per shot;
# the result is scaled straight back down and all video stays at clip size.
KLEIN_SCALE = 2


def swap_first_frame(frame, person, animal, posture, w, h, mode, out, seed=1, size=0.0):
    touches = person[2] >= 985
    abox = place.animal_box(person, posture, w / h, touches, animal, size)
    rec = place.swap_record(person, abox, animal, posture, mode, touches)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    big = out / "f0_big.png"
    Image.open(frame).convert("RGB").resize((w * KLEIN_SCALE, h * KLEIN_SCALE), Image.LANCZOS).save(big)
    name = comfy.upload(big, f"cs_{out.parent.name}_{out.name}_f0.png")
    paths, secs = comfy.run(graphs.klein_edit(rec, [name], w * KLEIN_SCALE, h * KLEIN_SCALE, seed=seed), out)
    first = out / "first.png"
    Image.open(paths[0]).convert("RGB").resize((w, h), Image.LANCZOS).save(first)
    print(f"  klein swap ({mode}, {KLEIN_SCALE}x): {secs}s  animal box {abox}")
    return first, abox, rec


def run_window(win_frames, first, alpha_mask, prompt, fps, seed, out):
    n = len(win_frames)
    frames = [first] + list(win_frames[1:])
    frames += [frames[-1]] * (LTX_FRAMES(n) - n)
    video = cx.encode(frames, Path(out) / "window.mp4", fps)
    mask_png = cx.save(alpha_mask, Path(out) / "region.png")
    vname = comfy.upload(video, f"cs_{Path(out).parent.name}_{Path(out).name}.mp4")
    mname = comfy.upload(mask_png, f"cs_{Path(out).parent.name}_{Path(out).name}_region.png")
    g = graphs.ltx_mask_inpaint(vname, mname, prompt, LTX_FRAMES(n), fps, seed=seed)
    paths, secs = comfy.run(g, Path(out) / "gen")
    print(f"  ltx window: {len(paths)} frames in {secs}s")
    return [cx.load(p) for p in sorted(x for x in paths if x.suffix == ".png")[:n]]


def swap_shot(data, shot, a, work):
    w, h, fps = data["width"], data["height"], data["fps"]
    d = shot.get("decision", {}).get("answers", {})
    posture = a.posture or d.get("posture", {}).get("choice", "standing")
    sdir = work / f"shot{shot['id']:04d}"
    orig_paths = cx.extract(data["clip"], shot["start"], shot["end"], sdir / "orig")
    orig = [cx.load(p) for p in orig_paths]
    seed_box = graphs.to_grid(d["where"]["box"]) if d.get("where", {}).get("box") else None
    masks = track(cx.encode(orig_paths, sdir / "shot.mp4", fps), w, h,
                  points=a.point if shot["id"] == a.point_shot else None, seed_box=seed_box, out=sdir / "track")
    person = cx.bbox(masks[0]) or seed_box
    if person is None:
        print(f"  shot {shot['id']}: lost the character, skipping")
        return None
    first, abox, rec = swap_first_frame(orig_paths[0], person, a.animal, posture, w, h, a.mode, sdir / "klein", a.seed, a.size)
    prompt = place.motion_prompt(a.animal, posture, a.scene, person[2] >= 985)
    (sdir / "plan.json").write_text(json.dumps({"posture": posture, "person": person, "animal_box": abox,
                                                "klein_record": rec, "ltx_prompt": prompt}, indent=2))
    done = {}
    for k, (s, e) in enumerate(shot["windows"][:a.max_windows]):
        i0, i1 = s - shot["start"], e - shot["start"]
        alpha_mask = cx.region(masks[i0:i1], abox)
        alpha = cx.feather(alpha_mask)
        wdir = sdir / f"win{k:02d}"
        wdir.mkdir(parents=True, exist_ok=True)
        first_frame = first if k == 0 else cx.save(done[i0], wdir / "first.png")
        gen = run_window(orig_paths[i0:i1], first_frame, alpha_mask, prompt, fps, a.seed + k, wdir)
        overlap = sum(1 for i in range(i0, i1) if i in done)
        for j, g in enumerate(gen):
            i = i0 + j
            c = cx.paste(orig[i], g, alpha)
            if i in done:  # overlap with the previous window: frame 0 is its frame, then crossfade
                if i == i0:
                    continue
                t = j / overlap
                c = done[i] * (1 - t) + c * t
            done[i] = c
    n = max(done) + 1  # all of the shot, or as far as --max-windows got
    frames = [cx.save(done[i], sdir / f"out{i:05d}.png") for i in range(n)]
    out = cx.mux(frames, data["clip"], shot["start"], shot["start"] + n, fps, sdir / "swapped.mp4")
    print(f"  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="a clip.mp4 made by catswap.fetch, or a URL to fetch")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--who", required=True, help='who becomes the animal, e.g. "the man in the brown suit"')
    ap.add_argument("--animal", default="cat", help="any animal: cat, monkey, rabbit, camel, ...")
    ap.add_argument("--size", type=float, default=0.0, help="animal height / adult height; 0 = look it up")
    ap.add_argument("--posture", choices=["standing", "sitting", "walking", "lying"], help="override d1")
    ap.add_argument("--point", type=lambda s: [tuple(map(float, s.split(",")))],
                    help="x,y click on the character in the first frame of --point-shot (pixels)")
    ap.add_argument("--point-shot", type=int, default=0)
    ap.add_argument("--shots", help="comma-separated shot ids to swap (default: every shot d1 says to)")
    ap.add_argument("--mode", choices=["two_rows", "replace"], default="replace", help="Klein edit-row style")
    ap.add_argument("--scene", default="", help="optional one-line scene description for the LTX prompt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-windows", type=int, help="only do the first N windows of each shot (for tests)")
    ap.add_argument("--work", default="work/run")
    a = ap.parse_args()

    work = Path(a.work)
    clip = Path(a.clip)
    if not clip.exists():
        from . import fetch
        clip, _ = fetch.fetch(a.clip, work / "clip", a.start, a.end)
    shots_json = clip.parent / "shots.json"
    data = json.loads(shots_json.read_text()) if shots_json.exists() else shotlib.split(clip)
    if data.get("who") != a.who:
        from . import decide
        model = decide.load()
        data["who"] = a.who
        for s in data["shots"]:
            ans = decide.decide(model, s["keyframe"], a.who)
            s["decision"] = {"answers": ans, "swap": decide.swap(ans)}
            print(f"shot {s['id']}: swap={s['decision']['swap']} posture={ans['posture']['choice']}")
        shots_json.write_text(json.dumps(data, indent=2, default=str))

    pick = set(map(int, a.shots.split(","))) if a.shots else None
    outs = []
    for s in data["shots"]:
        if (pick is not None and s["id"] not in pick) or (pick is None and not s["decision"]["swap"]):
            continue
        print(f"shot {s['id']} frames {s['start']}-{s['end']}:")
        out = swap_shot(data, s, a, work)
        if out:
            outs.append(out)
    print(json.dumps([str(o) for o in outs], indent=2))


if __name__ == "__main__":
    main()
