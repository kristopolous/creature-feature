#!/bin/bash
# One-time setup of the cat ComfyUI on _vastie (7900 XTX, ROCm). Logs to ~/cat/setup.log.
set -euo pipefail
R=/raid/ai/cameo
C=$HOME/cat
mkdir -p $C/models/{diffusion_models,text_encoders,vae,checkpoints,loras} $C/output $C/input

# models: dereference the NFS symlinks into real local files (fast loads, no load on wolfers' disk)
copy() { rsync -aL --info=progress2 "$R/models/$1" "$C/models/$(dirname $1)/"; }
(
copy diffusion_models/flux3-image-klein-int8-convrot.safetensors
copy text_encoders/qwen3vl_4b_int8_convrot.safetensors
copy vae/flux2-vae.safetensors
copy checkpoints/sam3.1_multiplex_fp16.safetensors
copy vae/ltx-2.5-video-vae-bf16.safetensors
copy vae/ltx-2.5-audio-vae-bf16.safetensors
copy loras/ltx-2.5-22b-ic-lora-bbox-control-3000-rank-32.safetensors
copy text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
copy diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
copy diffusion_models/ltx-2.5-22b-distilled-Q4_K_M.gguf
echo MODELS_DONE
) > $C/models.log 2>&1 &

# ComfyUI at the exact core commit wolfers runs, plus the same custom nodes
[ -d $C/ComfyUI ] || git clone -q $R/ComfyUI $C/ComfyUI
for n in $(ls $R/ComfyUI/custom_nodes); do
  src=$R/ComfyUI/custom_nodes/$n
  [ -d $src/.git ] && [ ! -e $C/ComfyUI/custom_nodes/$n ] && git clone -q $src $C/ComfyUI/custom_nodes/$n || true
done
python3 -m venv $C/venv
. $C/venv/bin/activate
pip install -q -U pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1 \
  || pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
pip install -q -r $C/ComfyUI/requirements.txt
for r in $C/ComfyUI/custom_nodes/*/requirements.txt; do pip install -q -r $r || echo "WARN reqs failed: $r"; done
python -c "import torch;print('torch',torch.__version__,'hip',torch.version.hip,'gpu',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
wait
echo SETUP_DONE
