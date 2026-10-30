"""Check every catswap graph against a live ComfyUI's /object_info (node types, input names, link types).
    COMFY_URL=http://127.0.0.1:8199 python scripts/check_graphs.py"""
import json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catswap import graphs, place

URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8199")
info = json.load(urllib.request.urlopen(URL + "/object_info", timeout=120))

def check(name, g):
    errs = []
    for nid, n in g.items():
        spec = info.get(n["class_type"])
        if not spec:
            errs.append(f"{nid} unknown node {n['class_type']}"); continue
        req, opt = spec["input"].get("required", {}), spec["input"].get("optional", {})
        allowed = {**req, **opt}
        for k in req:
            if k not in n["inputs"] and not any(i.startswith(k + ".") for i in n["inputs"]):
                errs.append(f"{nid} {n['class_type']} missing required '{k}'")
        for k, v in n["inputs"].items():
            base = k.split(".")[0]
            if base not in allowed:
                errs.append(f"{nid} {n['class_type']} unknown input '{k}'"); continue
            if isinstance(v, list):
                src = g[v[0]]; outs = info[src["class_type"]]["output"]
                want = allowed[base][0]
                got = outs[v[1]] if v[1] < len(outs) else None
                if base == k and isinstance(want, str) and want != "*" and got != "*" and got != want \
                        and not (want == "COMFY_DYNAMICCOMBO_V3"):
                    errs.append(f"{nid} {n['class_type']}.{k} wants {want}, got {src['class_type']}[{v[1]}]={got}")
            else:
                want = allowed[base][0]
                if isinstance(want, list) and v not in want and not str(v).endswith((".safetensors", ".gguf", ".png", ".mp4")):
                    errs.append(f"{nid} {n['class_type']}.{k}={v!r} not in options (e.g. {want[:4]})")
    print(("OK   " if not errs else "FAIL ") + name)
    for e in errs: print("     ", e)
    return not errs

rec = place.swap_record([200, 300, 1000, 600], [550, 330, 1000, 570])
ok = all([
    check("sam3_track(points)", graphs.sam3_track("x.mp4", points=[(100, 80)])),
    check("sam3_track(box)", graphs.sam3_track("x.mp4", box=[50, 20, 100, 160])),
    check("sam3_track(text)", graphs.sam3_track("x.mp4", text="the man in the suit")),
    check("klein_edit(swap)", graphs.klein_edit(rec, ["f0.png"], 352, 192)),
    check("ltx_mask_inpaint", graphs.ltx_mask_inpaint("w.mp4", "r.png", "a cat", 25, 12)),
    check("ltx_box_inpaint", graphs.ltx_box_inpaint("w.mp4", [[0, 0, 1000, 1000]], "a cat", 25, 12)),
])
sys.exit(0 if ok else 1)
