"""Dev harness: queue the scene App graph once. Usage: run_scene.py STARTER [SWAPS] [OUT_DIR]"""
import json, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from catswap import comfy, graphs
starter = sys.argv[1]
swaps = sys.argv[2] if len(sys.argv) > 2 else ""
out = sys.argv[3] if len(sys.argv) > 3 else "work/app/scene_out"
g = graphs.scene_app(starter, swaps, None)  # starter only: no upload node
paths, secs = comfy.run(g, out, timeout=7200)
print("scene run", secs, "s", [str(p) for p in paths])
