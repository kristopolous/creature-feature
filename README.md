# Creature Feature

**Pick anyone in a movie scene and turn them into an animal.** The rest of the scene stays exactly as it was: the set, the other actors, and the original dialog and music. The animal doesn't act out the part. It behaves like a normal animal while everyone else carries on talking to it.

Ilsa in *Casablanca* becomes a tabby cat who looks around the room while "Play it, Sam" plays over it. Sam becomes a parrot on the piano bench. You type who to replace and which animal ("cat", "parrot", "camel", "a fluffy white goat"), and the app does the rest.

It is a **ComfyUI App**: one workflow in App mode with four inputs (**video, who, animal, scene**) and the swapped video as its output. Everything runs on **open-weight models** on one consumer GPU (tested on an RX 7900 XTX, 24 GB, ROCm).

## How it works

```
video ─► Fit Frames ─► d1 decision model ──► SAM3 ──► Plan ──► FLUX 3 Klein ×2 ──► LTX-2.5 ──► Composite ─► video
         192 px,       "who" → box +         track     animal    remove person,     regenerate    paste back over
         12 fps,       posture               the       box,      then add animal    only the      the original;
         8n+1 frames,                        person    prompts,  to frame 0         person's      original audio
         stop at cut                                   region                       region
```

1. **CatSwap Fit Frames** keeps every run at the lowest resolution: 192 px tall, about 12 fps, an 8n+1 frame count, and it stops at the first camera cut. The audio is trimmed to match.
2. **CatSwap Find Character** runs [LiquidAI d1-2B](https://huggingface.co/LiquidAI/d1-2B), an open *decision model*, inside the graph. In one forward pass it decides whether "who" is in the frame, where they are (a box), and what they're doing (standing, sitting, walking or lying). It writes no text.
3. **SAM3** (in ComfyUI core) segments the person using the same "who" text, with d1's box to pick out the right one. It then tracks them through the shot.
4. **CatSwap Plan** puts an animal of real-world size where the person was. A rabbit is ankle-high, a goat waist-high, and a camel towers over the spot. If their feet are out of frame, the animal takes their place on the seat. Plan also writes:
   - the FLUX 3 box-edit records,
   - the motion prompt, which tells the animal to behave like a normal animal, not copy the actor,
   - the region to regenerate: everywhere the person went, plus the animal's box.
5. **FLUX 3 Klein** (open weights) edits frame 0 twice. Pass 1 **removes** the person and rebuilds the background behind them. Pass 2 **adds** the animal into that clean frame. A single pass can't erase someone who fills a third of the frame. This one frame is edited at 2× resolution, because at 192 px Klein ignores small boxes.
6. **LTX-2.5** (22B distilled, int8) regenerates **only the region** across the shot, starting from the edited frame 0. That's done by **CatSwap Mask Noise Mask**, which maps a pixel mask onto LTX's causal latent grid.
7. **CatSwap Composite** pastes the result back over the **original** frames, inside a feathered region only. Every pixel outside the region is the real film, so seams can only appear on the animal.

## Nodes (`comfyui_catswap/`)

| Node | What it does |
|---|---|
| CatSwap Fit Frames | Enforces the lowest resolution: 192 px, about 12 fps, 8n+1 frames, stop at the first cut, trim the audio to match. |
| CatSwap Find Character (d1) | Runs a decision model in the graph: is "who" present, where (bbox), and what posture. |
| CatSwap Plan | Animal box sized by species, Klein remove/add records, LTX prompt, regenerated region. |
| CatSwap Set First Frame | Replaces frame 0 of a batch with the edited frame. |
| CatSwap Mask Noise Mask | Pixel mask to a video-latent `noise_mask` (temporal ×8, max-pooled, dilated, frame 0 frozen). |
| Cameo Box Noise Mask | The same idea with boxes instead of a mask. |
| CatSwap Composite | Pastes the generated frames over the original inside a feathered region. |

## Setup

Requires **ComfyUI 0.43** or later, which includes the SAM3, LTX-2.5, FLUX 3 and App mode support used here.

```sh
# 1. node pack
cd ComfyUI/custom_nodes && git clone https://github.com/kristopolous/creature-feature
ln -s creature-feature/comfyui_catswap comfyui_catswap
pip install transformers  # for the d1 decision model

# 2. models (into ComfyUI/models/...)
#   diffusion_models/  flux3-image-klein-int8-convrot.safetensors              (black-forest-labs, gated)
#                      ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors  (Lightricks/LTX-2.5, gated)
#   text_encoders/     qwen3vl_4b_int8_convrot.safetensors
#                      gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
#   vae/               flux2-vae.safetensors, ltx-2.5-video-vae-bf16.safetensors, ltx-2.5-audio-vae-bf16.safetensors
#   checkpoints/       sam3.1_multiplex_fp16.safetensors                        (Comfy-Org/sam3.1)
# d1-2B is pulled from Hugging Face (LiquidAI/d1-2B) on first use.

# 3. run (24 GB GPU; --cache-none keeps RAM under ~60 GB, since the app touches ~50 GB of weights)
python main.py --cache-none
```

Then load `workflows/catswap_app.json`. It opens in **App mode**. Upload a clip of one shot (static camera works best), type **who** ("the woman in the white dress") and **animal**, and run.

**Tips:**
- **Describe what's visible, not the character's role.** "the man in the white jacket and bow tie" finds Sam at 0.99 confidence; "the piano player" scores 0.2.
- **One shot per run.** The app stops at the first cut. To do a whole scene, split it with the CLI below and run each shot.
- **First run on AMD:** MIOpen tunes kernels for each new shape, which can take a while. Set `MIOPEN_FIND_MODE=FAST`. After that, LTX takes seconds per shot.

## CLI (development harness)

`catswap/` drives the same graphs from Python, for batch work and tests:

```sh
python -m catswap.fetch URL --out work/clip        # lowest-res fetch (yt-dlp + ffmpeg), 192 px, 12 fps
python -m catswap.shots work/clip/clip.mp4         # shot split (luma only, works on B&W) + windows
python -m catswap.decide work/clip/shots.json --who "the woman in the white dress"   # d1 on CPU, ~2.5 s/shot
COMFY_URL=http://127.0.0.1:8188 python -m catswap.run work/clip/clip.mp4 --who "..." --animal parrot --shots 5
python -m catswap.ui --app workflows/catswap_app_api.json workflows/catswap_app.json  # API graph -> App workflow
```

Video files are never committed; clips are fetched from a URL at runtime.

## License

MIT
