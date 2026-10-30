"""Turn an API-format graph into a frontend workflow (editable in ComfyUI, listed under Workflows),
using comfy-cli's own edit ops so node layouts and widgets match the live catalog.

    python -m catswap.ui workflows/cameo_insert_api.json workflows/cameo_insert.json
    python -m catswap.ui --app workflows/catswap_app_api.json workflows/catswap_app.json   # App mode

COMFY_UI_PORT picks the ComfyUI whose live catalog is used (default 8189: the vastie tunnel).
"""
import os
import json
import subprocess
import sys

EMPTY = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "groups": [],
         "config": {}, "extra": {}, "version": 0.4}
CATALOG = ["--where", "local", "--host", "127.0.0.1", "--port", os.environ.get("COMFY_UI_PORT", "8189")]


def op(*args):
    out = subprocess.run(["comfy", "workflow", *args, *CATALOG], capture_output=True, text=True)
    env = json.loads(out.stdout.splitlines()[0])
    if not env["ok"]:
        raise RuntimeError(f"{args}: {env['error']['message']}")
    return env["data"]


def convert(api, path):
    json.dump(EMPTY, open(path, "w"))
    ids, pos = {}, {}
    # lay nodes out left to right by dependency depth
    def depth(nid, seen=()):
        links = [v for v in api[nid]["inputs"].values() if isinstance(v, list)]
        return 0 if not links else 1 + max(depth(v[0], seen + (nid,)) for v in links)
    rows = {}
    for nid in api:
        d = depth(nid)
        pos[nid] = f"{40 + d * 360},{60 + rows.get(d, 0) * 220}"
        rows[d] = rows.get(d, 0) + 1
    for nid, node in api.items():
        data = op("add-node", path, node["class_type"], "--at", pos[nid])
        ids[nid] = data["op"]["node_id"]
    for nid, node in api.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                op("connect", path, f"{ids[value[0]]}.{value[1]}", f"{ids[nid]}.{name}")
            else:
                op("set-widget", path, f"{ids[nid]}.{name}", json.dumps(value))
    return ids


# What the App shows: (class_type, widget) inputs, in order, and the output node types.
APP_INPUTS = [("LoadVideo", "file"), ("CatSwapFindCharacter", "who"), ("CatSwapPlan", "animal"),
              ("CatSwapPlan", "scene")]
APP_OUTPUTS = ["SaveVideo"]


def make_app(api, path):
    """Convert, then switch the workflow into App (linear) mode with the App's inputs and outputs."""
    ids = convert(api, path)
    by_type = {}
    for nid, node in api.items():
        by_type.setdefault(node["class_type"], ids[nid])
    wf = json.load(open(path))
    wf.setdefault("extra", {})["linearMode"] = True
    wf["extra"]["linearData"] = {"inputs": [[by_type[t], w] for t, w in APP_INPUTS],
                                 "outputs": [by_type[t] for t in APP_OUTPUTS]}
    json.dump(wf, open(path, "w"), indent=1)
    return path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--app"]
    (make_app if "--app" in sys.argv else convert)(json.load(open(args[0])), args[1])
    print("wrote", args[1])
