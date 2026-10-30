#!/bin/sh
# Start the cat ComfyUI on _vastie (7900 XTX). Reach it from the laptop with:
#   ssh -f -N -L 8189:127.0.0.1:8188 _vastie   then COMFY_URL=http://127.0.0.1:8189
export MIOPEN_FIND_MODE=FAST  # skip most of the per-shape conv kernel search (first LTX run took 17 min without it)
cd $HOME/cat/ComfyUI && echo $$ > $HOME/cat/comfy.pid && exec $HOME/cat/venv/bin/python main.py \
  --listen 127.0.0.1 --port 8188 --output-directory $HOME/cat/output --input-directory $HOME/cat/input \
  --extra-model-paths-config $HOME/cat/extra_models.yaml --use-pytorch-cross-attention --cache-none  # cache-none: the App holds ~50 GB of models; caching them OOM-killed a 62 GB box
