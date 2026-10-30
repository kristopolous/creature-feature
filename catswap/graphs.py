"""API-format ComfyUI graphs, built as plain dicts. Every default is the LOWEST resolution."""
import json
import math

KLEIN = "flux3-image-klein-int8-convrot.safetensors"
KLEIN_TE = "qwen3vl_4b_int8_convrot.safetensors"
FLUX_VAE = "flux2-vae.safetensors"


class Graph(dict):
    """Node ids are assigned in insertion order; add() returns an output reference factory."""

    def add(self, class_type, **inputs):
        nid = str(len(self) + 1)
        self[nid] = {"class_type": class_type, "inputs": inputs}
        return lambda slot=0: [nid, slot]


def _klein_base(g, width, height, prompt):
    unet = g.add("UNETLoader", unet_name=KLEIN, weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=KLEIN_TE, type="flux3", device="default")
    vae = g.add("VAELoader", vae_name=FLUX_VAE)
    mu = math.log(16.0)  # alpha=16, the Klein reference workflow's timestep shift
    model = g.add("ModelSamplingFlux", model=unet(), max_shift=mu, base_shift=mu, width=width, height=height)
    pos = g.add("CLIPTextEncode", clip=clip(), text=prompt)
    neg = g.add("CLIPTextEncode", clip=clip(), text="")
    latent = g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    return model, vae, pos, neg, latent


def _sample(g, model, vae, pos, neg, latent, steps, seed, prefix):
    s = g.add("KSampler", model=model(), seed=seed, steps=steps, cfg=1.0, sampler_name="euler",
              scheduler="simple", positive=pos, negative=neg, latent_image=latent, denoise=1.0)
    img = g.add("VAEDecode", samples=s(), vae=vae())
    g.add("SaveImage", images=img(), filename_prefix=prefix)
    return g


def klein_t2i(prompt, width=256, height=256, steps=16, seed=1, prefix="catswap/t2i"):
    g = Graph()
    model, vae, pos, neg, latent = _klein_base(g, width, height, prompt)
    return _sample(g, model, vae, pos(), neg(), latent(), steps, seed, prefix)


def klein_edit(prompt, images, width, height, steps=16, seed=1, ref_megapixels=0.07, prefix="catswap/edit"):
    """images: uploaded names in order; images[0] is ref_image_0 (the scene), images[1] is ref_image_1, ..."""
    g = Graph()
    model, vae, pos, neg, latent = _klein_base(g, width, height, prompt)
    refs = {f"images.image{i}": g.add("LoadImage", image=name)() for i, name in enumerate(images)}
    ref = g.add("ReferenceMultiLatent", positive=pos(), negative=neg(), vae=vae(),
                resolution="resize", **{"resolution.megapixels": ref_megapixels}, **refs)
    return _sample(g, model, vae, ref(0), ref(1), latent(), steps, seed, prefix)


def to_grid(box):
    """d1 box {top,left,bottom,right} in 0..1 -> FLUX 3 [y_min, x_min, y_max, x_max] on 0..1000."""
    return [round(box["top"] * 1000), round(box["left"] * 1000),
            round(box["bottom"] * 1000), round(box["right"] * 1000)]


def insert_record(tgt_bbox, desc, where, person_src_bbox=(0, 0, 1000, 1000)):
    """The i2i_bbox annotation record Klein was trained on: instruction paragraph + one JSON table."""
    para = (f"Add the person <person_1> from <ref_image_1> to the scene, {where}, matching the scene's "
            f"lighting, color grade, film grain and perspective. Everything else stays identical to the source.")
    rows = [{"id": "person_1", "from": "ref_image_1", "src_bbox": list(person_src_bbox),
             "tgt_bbox": list(tgt_bbox), "desc": desc}]
    return f"{para} {json.dumps(rows)}"


LTX = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX_TE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LTX_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, person, human, man, woman"
LTX_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # distilled, stage 1 only


def _ltx_inpaint(g, vid, masker, prompt, frames, fps, seed, negative, prefix):
    comp = g.add("GetVideoComponents", video=vid())
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    lat = g.add("VAEEncode", pixels=comp(0), vae=vae())
    masked = masker(lat)
    audio = g.add("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=avae())
    av = g.add("LTXVConcatAVLatent", video_latent=masked(), audio_latent=audio())
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    pos = g.add("CLIPTextEncode", clip=clip(), text=prompt)
    neg = g.add("CLIPTextEncode", clip=clip(), text=negative)
    cond = g.add("LTXVConditioning", positive=pos(), negative=neg(), frame_rate=float(fps))
    model = g.add("UNETLoader", unet_name=LTX, weight_dtype="default")
    guider = g.add("CFGGuider", model=model(), positive=cond(0), negative=cond(1), cfg=1.0)
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(), guider=guider(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(), latent_image=av())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    img = g.add("VAEDecode", samples=sep(0), vae=vae())
    g.add("SaveImage", images=img(), filename_prefix=prefix)
    return g


def ltx_box_inpaint(video, boxes, prompt, frames, fps, seed=1, dilate=1, keep_frames=1,
                    negative=NEGATIVE, prefix="catswap/ltx"):
    """Regenerate only `boxes` of an uploaded window video whose frame 0 is already edited.
    No upscaler, no second stage: LOWEST resolution."""
    g = Graph()
    vid = g.add("LoadVideo", file=video)
    masker = lambda lat: g.add("CameoBoxNoiseMask", latent=lat(), boxes=json.dumps(boxes),
                               dilate=dilate, keep_frames=keep_frames)
    return _ltx_inpaint(g, vid, masker, prompt, frames, fps, seed, negative, prefix)


def ltx_mask_inpaint(video, mask_image, prompt, frames, fps, seed=1, dilate=1, keep_frames=1,
                     negative=NEGATIVE, prefix="catswap/ltx"):
    """Regenerate only the white area of `mask_image` (the swapped person's region, static over the window)
    of an uploaded window video whose frame 0 already has the animal. LOWEST resolution, stage 1 only."""
    g = Graph()
    vid = g.add("LoadVideo", file=video)
    mask = g.add("LoadImageMask", image=mask_image, channel="red")
    masker = lambda lat: g.add("CatSwapMaskNoiseMask", latent=lat(), mask=mask(), temporal_scale=8,
                               dilate=dilate, keep_frames=keep_frames)
    return _ltx_inpaint(g, vid, masker, prompt, frames, fps, seed, negative, prefix)


SAM3 = "sam3.1_multiplex_fp16.safetensors"


def sam3_track(video, points=None, box=None, text=None, threshold=0.5, prefix="catswap/track"):
    """Segment the chosen character on frame 0 (click points, a box, or text) and track them through
    the uploaded window/shot video. Saves one mask PNG per frame. Inputs are already LOWEST resolution."""
    g = Graph()
    ckpt = g.add("CheckpointLoaderSimple", ckpt_name=SAM3)
    vid = g.add("LoadVideo", file=video)
    frames = g.add("GetVideoComponents", video=vid())
    first = g.add("ImageFromBatch", image=frames(0), batch_index=0, length=1)
    det = {"model": ckpt(0), "image": first(), "threshold": threshold, "refine_iterations": 2,
           "individual_masks": False}
    if text:
        det["conditioning"] = g.add("CLIPTextEncode", clip=ckpt(1), text=text)()
    if points:
        det["positive_coords"] = g.add("PrimitiveString", value=json.dumps(
            [{"x": int(x), "y": int(y)} for x, y in points]))()
    if box:
        x, y, w, h = box
        det["bboxes"] = g.add("PrimitiveBoundingBox", x=int(x), y=int(y), width=int(w), height=int(h))()
    seed_mask = g.add("SAM3_Detect", **det)
    track = g.add("SAM3_VideoTrack", images=frames(0), model=ckpt(0), initial_mask=seed_mask(0),
                  detection_threshold=threshold, max_objects=1, detect_interval=1)
    masks = g.add("SAM3_TrackToMask", track_data=track(), object_indices="0")
    g.add("SaveImage", images=g.add("MaskToImage", mask=masks())(), filename_prefix=prefix)
    return g


def app(video="input.mp4", who="the man in the brown suit", animal="cat", scene="", seed=1,
        height=192, target_fps=12.0, max_frames=161, klein_scale=2.0, prefix="catswap/app"):
    """The whole CatSwap App as ONE graph: video + who + animal in, swapped video (original audio) out.
    LOWEST resolution is enforced inside the graph by CatSwapFitFrames. The only thing run bigger is the single
    first-frame Klein edit (klein_scale), because at 192 px Klein ignores small boxes."""
    g = Graph()
    vid = g.add("LoadVideo", file=video)
    comp = g.add("GetVideoComponents", video=vid())
    fit = g.add("CatSwapFitFrames", images=comp(0), fps=comp(2), audio=comp(1), height=height,
                target_fps=target_fps, max_frames=max_frames, cut_threshold=12.0)
    first = g.add("ImageFromBatch", image=fit(0), batch_index=0, length=1)

    # who -> box + posture (d1 decision model) -> person track (SAM3)
    find = g.add("CatSwapFindCharacter", image=first(), who=who, posture_override="auto")
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM3)
    sam_text = g.add("CLIPTextEncode", clip=sam(1), text=find(3))  # SAM3 takes the same words; d1's box picks who
    det = g.add("SAM3_Detect", model=sam(0), image=first(), conditioning=sam_text(), bboxes=find(0), threshold=0.5,
                refine_iterations=2, individual_masks=False)
    track = g.add("SAM3_VideoTrack", images=fit(0), model=sam(0), initial_mask=det(0),
                  detection_threshold=0.5, max_objects=1, detect_interval=1)
    masks = g.add("SAM3_TrackToMask", track_data=track(), object_indices="0")
    plan = g.add("CatSwapPlan", masks=masks(), who=find(3), animal=animal, size=0.0, posture=find(1), scene=scene,
                 grow=6)

    # frame 0, two FLUX 3 Klein box edits: remove the person (clean plate), then add the animal into the plate
    big = g.add("ImageScaleBy", image=first(), upscale_method="lanczos", scale_by=klein_scale)
    size = g.add("GetImageSize", image=big())
    unet = g.add("UNETLoader", unet_name=KLEIN, weight_dtype="default")
    kclip = g.add("CLIPLoader", clip_name=KLEIN_TE, type="flux3", device="default")
    kvae = g.add("VAELoader", vae_name=FLUX_VAE)
    kmodel = g.add("ModelSamplingFlux", model=unet(), max_shift=math.log(16.0), base_shift=math.log(16.0),
                   width=size(0), height=size(1))
    kneg = g.add("CLIPTextEncode", clip=kclip(), text="")
    klat = g.add("EmptyFlux2LatentImage", width=size(0), height=size(1), batch_size=1)

    def klein_pass(prompt, image):
        pos = g.add("CLIPTextEncode", clip=kclip(), text=prompt)
        ref = g.add("ReferenceMultiLatent", positive=pos(), negative=kneg(), vae=kvae(), resolution="resize",
                    **{"resolution.megapixels": 0.07, "images.image0": image})
        ks = g.add("KSampler", model=kmodel(), seed=seed, steps=16, cfg=1.0, sampler_name="euler",
                   scheduler="simple", positive=ref(0), negative=ref(1), latent_image=klat(), denoise=1.0)
        return g.add("VAEDecode", samples=ks(), vae=kvae())

    plate = klein_pass(plan(0), big())
    swapped = klein_pass(plan(1), plate())
    window = g.add("CatSwapSetFirstFrame", images=fit(0), first=swapped())

    # the shot: LTX-2.5 regenerates only the person's region, starting from the swapped frame
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    lat = g.add("VAEEncode", pixels=window(), vae=vae())
    masked = g.add("CatSwapMaskNoiseMask", latent=lat(), mask=plan(3), temporal_scale=8, dilate=1, keep_frames=1)
    alat = g.add("LTXVEmptyLatentAudio", frames_number=fit(1), frame_rate=fit(2), batch_size=1, audio_vae=avae())
    av = g.add("LTXVConcatAVLatent", video_latent=masked(), audio_latent=alat())
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    pos = g.add("CLIPTextEncode", clip=clip(), text=plan(2))
    neg = g.add("CLIPTextEncode", clip=clip(), text=NEGATIVE)
    cond = g.add("LTXVConditioning", positive=pos(), negative=neg(), frame_rate=fit(2))
    model = g.add("UNETLoader", unet_name=LTX, weight_dtype="default")
    guider = g.add("CFGGuider", model=model(), positive=cond(0), negative=cond(1), cfg=1.0)
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(), guider=guider(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(), latent_image=av())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    gen = g.add("VAEDecode", samples=sep(0), vae=vae())

    # paste back over the original, keep the original dialog
    final = g.add("CatSwapComposite", original=fit(0), generated=gen(), region=plan(3), feather=2)
    video_out = g.add("CreateVideo", images=final(), fps=fit(2), audio=fit(3))
    g.add("SaveVideo", video=video_out(), filename_prefix=prefix, format="auto", **{"format.codec": "auto"})
    return g
