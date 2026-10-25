# Workflow node and settings reference (tooltips)

Part of [workflow-implementation-plan.md](../workflow-implementation-plan.md). This document is the specification for
every node of the workflow system and for every parameter a user can change. The **Tooltip** column is the exact
English text the editor shows when the pointer rests on that widget or socket (W1.5); translations start from it.

## How tooltips are written

- **Widgets** (values a user can change): one or two sentences on what the parameter does and what raising or lowering
  it changes, then **Example:** with concrete values and their effect. The coverage test (W1.6) checks for `Example:`
  (or `e.g.` in very short tips).
- **Sockets** (link-only inputs such as MODEL or IMAGE): a short description of what to connect, with an example where
  the right source is not obvious.
- Shared parameters use a shared tooltip key (§C). In the node tables the tooltip cell then shows the key, e.g.
  `→ C.seed`; the UI shows the full shared text. A node may append one node-specific sentence to a shared tip (shown
  after `+`).
- Combo options that need explaining have per-option tooltips (§O), shown when hovering the option in the drop-down.
- Ranges are `min … max`, `step`, default in **bold**. "Model list" means the combo lists files from that model folder.
- Node ids are in `code`; display names are the node titles.

Contents: [C Common](#c-common-tooltips) · [O Options](#o-option-tooltips) · [N1 Loaders](#n1-loaders) ·
[N2 Text](#n2-text-and-prompts) · [N3 Conditioning](#n3-conditioning) · [N4 Control](#n4-control-vision-and-adapters) ·
[N5 Sampling](#n5-sampling) · [N6 Custom sampling](#n6-custom-sampling) · [N7 Model sampling and patches](#n7-model-sampling-and-model-patches) ·
[N8 Latent](#n8-latent) · [N9 Image](#n9-image) · [N10 Mask](#n10-mask) · [N11 Input and output](#n11-input-and-output) ·
[N12 Utility](#n12-utility) · [N13 Hooks](#n13-hooks-scheduled-loras) · [N14 Merge and save](#n14-merging-and-saving-models) ·
[N15 Family-specific](#n15-family-specific-image-nodes) · [N16 Video](#n16-video) · [N17 Audio](#n17-audio) · [N18 3D](#n18-3d) ·
[N19 Training](#n19-training-stretch) · [S Settings](#s-settings)

---

## C. Common tooltips

| Key | Tooltip |
|---|---|
| `C.seed` | The random number that creates the starting noise. The same seed with the same settings gives the same image; a different seed gives a different composition. Example: seed 42 always gives the same lighthouse; change it to 43 for a new variation with everything else unchanged. |
| `C.seedControl` | What happens to the seed after each run is queued: *fixed* keeps it, *increment* adds 1, *decrement* subtracts 1, *randomize* picks a new one. Example: set *increment* and queue 4 times to get seeds 100, 101, 102, 103. |
| `C.steps` | How many denoising steps the sampler takes. More steps give finer detail up to a point, then only cost time. Example: 20–30 for most models with *euler* or *dpm++ 2M*; 1–4 for Turbo/Lightning/LCM models; 50 rarely helps. |
| `C.cfg` | Classifier-free guidance scale: how strongly the image follows the prompt versus the negative prompt. Higher is more literal but can look burnt or oversaturated. Example: 7.5 for SD 1.5, 7 for SDXL (the built-in family defaults), 1.0 for Turbo and distilled FLUX models (1.0 also skips the negative pass and runs twice as fast). |
| `C.sampler` | The numerical method used to remove noise step by step. Hover an option to see what it is good for. Example: *euler* is fast and safe; *dpm++ 2M* is sharp at 20–30 steps; *euler ancestral* adds variety but never fully converges. |
| `C.scheduler` | How the noise level drops across the steps. Hover an option for details. Example: *karras* spends more steps at low noise for fine detail; *simple* or *normal* suit flow models such as SD3 and FLUX; *sgm uniform* suits Turbo/LCM. |
| `C.denoise` | How much of the input latent is replaced: 1.0 starts from pure noise, lower values keep more of the input image. Example: 1.0 for text-to-image; 0.5 for image-to-image that keeps the composition; 0.3 for a light retouch of a hi-res-fix pass. |
| `C.model` | The diffusion model that predicts the noise. Connect a loader's MODEL output, or a patched model (LoRA, FreeU, ...). Example: the MODEL output of Load Checkpoint. |
| `C.textEncoder` | The text encoder(s) that turn the prompt into numbers the model understands. Example: the TEXT_ENCODER output of Load Checkpoint, or of Load LoRA if a LoRA should also change how text is read. |
| `C.vae` | The autoencoder that converts between pixels and latents. It must match the model family. Example: the VAE output of Load Checkpoint, or an fp16-safe SDXL VAE from Load VAE to avoid black images. |
| `C.positive` | Conditioning that describes what you want. Example: the output of a Text Encode node with "a red fox in snow, golden hour". |
| `C.negative` | Conditioning that describes what to avoid. Ignored when cfg is 1.0. Example: "blurry, low quality, extra fingers". |
| `C.latentIn` | The latent to start from. Example: Empty Latent Image for text-to-image, or VAE Encode of a photo for image-to-image. |
| `C.width` | Width in pixels (latent nodes divide it by 8, or by the model's compression factor). Use the resolution the model was trained near. Example: 512 for SD 1.5, 1024 for SDXL/SD3/FLUX; 1216 × 832 for a landscape SDXL image. |
| `C.height` | Height in pixels; same rules as width. Example: 768 with width 512 gives a portrait SD 1.5 image. |
| `C.batchSize` | How many images (latents) are made in one run, side by side in memory. Faster per image than separate runs but needs more memory. Example: 4 gives four variations from consecutive noise in one run. |
| `C.filenamePrefix` | Start of the output file name. May contain folders and tokens: `%date:yyyy-MM-dd%`, `%time:HHmmss%`, `%node:<title or id>.<input>%`, `%width%`, `%height%`, `%seed%`. A counter is appended. Example: `portraits/%date:yyyy-MM-dd%/fox_%seed%` saves as `portraits/2026-10-25/fox_42_00001.png`. |
| `C.upscaleMethod` | Interpolation used when resizing. Hover an option for details. Example: *lanczos* for photos, *nearest exact* for pixel art, *bislerp* for latents. |
| `C.crop` | What to do when the new aspect ratio differs: *disabled* stretches, *center* crops the middle to keep proportions. Example: resizing 1024 × 768 to 512 × 512 with *center* cuts the sides instead of squashing the subject. |
| `C.strength` | How strongly this effect is applied, 0 = off, 1 = as trained. Example: 0.6 keeps the effect but lets the prompt win more often. |
| `C.startPercent` | When the effect starts, as a fraction of the sampling run (0 = first step). Example: 0.2 lets the composition form freely before the effect applies. |
| `C.endPercent` | When the effect stops, as a fraction of the run (1 = last step). Example: 0.6 lets the last 40 % of steps refine details without the effect. |
| `C.mask` | A mask: white (1) areas are affected, black (0) areas are not; grey is partial. Example: a mask painted over the sky in the mask editor. |
| `C.image` | An image or a batch of images (video frames are a batch). Example: the IMAGE output of Load Image or VAE Decode. |
| `C.x` | Horizontal offset in pixels from the left edge. Example: 256 places the item a quarter of the way across a 1024-wide image. |
| `C.y` | Vertical offset in pixels from the top edge. Example: 0 aligns with the top. |
| `C.device` | Which Vulkan GPU this component runs on, and whether its weights live in VRAM. *auto* follows the memory mode; *GPU, weights in RAM* keeps the weights in system RAM and streams them to the GPU, saving VRAM at some speed cost. Example: put a 9 GB T5 text encoder on *GPU, weights in RAM* so the diffusion model stays fully in VRAM on a 12 GB card. |
| `C.weightDtype` | Number format of the weights in memory. Smaller formats use less memory and may lose a little quality. Example: *fp8 e4m3* halves the VRAM of a FLUX model compared with *fp16*; *default* keeps the file's format. |
| `C.tileSize` | Size in pixels of the tiles processed one at a time. Smaller tiles use less memory but take longer and may show seams. Example: 512 for SDXL VAE decode on an 8 GB GPU; 256 if it still runs out of memory. |
| `C.overlap` | Pixels shared between neighbouring tiles, blended to hide seams. Example: 64 with tile size 512; raise it if you see grid lines. |
| `C.temporalSize` | For video: how many frames are processed per chunk. Example: 64; lower to 32 if decoding a long clip runs out of memory. |
| `C.temporalOverlap` | For video: frames shared between chunks, blended to hide jumps. Example: 8. |
| `C.batchIndex` | Position in the batch, starting at 0. Example: 2 picks the third image of a batch of four. |
| `C.length` | Number of items (or frames) to take or create. Example: 1 takes a single image; 81 frames is 5 s of Wan video at 16 fps. |
| `C.fps` | Frames per second of the video. Example: 24 for film-like motion, 16 for Wan, 8 for a quick preview. |

---

## O. Option tooltips

### O.sampler — samplers
| Option | Tooltip |
|---|---|
| euler | First-order, one model call per step. Fast, stable, a good default. Example: 20 steps, any model. |
| euler cfg++ | Euler with the CFG++ correction, which allows low guidance without washed-out images. Example: cfg 0.6–1.5 instead of 7. |
| euler ancestral | Euler that adds fresh noise each step: more variety and softer detail, and the image keeps changing as steps rise. Example: 30 steps for painterly results. |
| euler ancestral cfg++ | Ancestral Euler with the CFG++ correction. Example: cfg ~1. |
| heun | Second-order, two model calls per step: more accurate, twice as slow. Example: 10 steps ≈ the cost of 20 euler steps. |
| heun++ | Heun variant with an improved correction; same cost as heun. Example: 12 steps. |
| dpm2 | Second-order DPM solver, two calls per step. Example: 15 steps. |
| dpm2 ancestral | dpm2 with added noise each step. Example: 20 steps for varied results. |
| lms | Linear multistep: reuses previous steps, one call per step; can be unstable at very few steps. Example: 30 steps with karras. |
| dpm fast | DPM solver with fixed order mix; quick drafts. Example: 10–15 steps. |
| dpm adaptive | Chooses its own step sizes until an error target is met; ignores the step count. Slow but accurate. Example: final renders where time does not matter. |
| dpm++ 2S ancestral | Second-order single-step DPM++ with added noise. Example: 20–25 steps. |
| dpm++ 2S ancestral cfg++ | The above with the CFG++ correction. Example: cfg ~1. |
| dpm++ SDE | Stochastic DPM++ (two calls per step): rich detail, varies with step count. Example: 15–20 steps with karras. |
| dpm++ 2M | Second-order multistep, one call per step: sharp and converges well. Example: 20–30 steps with karras — a common best default. |
| dpm++ 2M cfg++ | dpm++ 2M with the CFG++ correction. Example: cfg ~1. |
| dpm++ 2M SDE | Stochastic multistep DPM++: more texture than 2M. Example: 25 steps, karras. |
| dpm++ 3M SDE | Third-order stochastic multistep; needs more steps to shine. Example: 30–40 steps with exponential. |
| ddpm | The original ancestral sampler; needs many steps. Example: 100+ steps. |
| ddim | Deterministic sampler from the DDIM paper; behaves like euler on eps models. Example: 25 steps with ddim uniform. |
| lcm | For Latent Consistency (LCM) models and LoRAs only. Example: 4–8 steps, cfg 1–2. |
| ipndm | Improved pseudo-numerical multistep (fourth order): good quality at low step counts. Example: 10–15 steps. |
| ipndm v | ipndm with variable step-size coefficients. Example: 10–15 steps. |
| deis | Diffusion exponential integrator: strong at low step counts. Example: 10 steps. |
| uni pc | Unified predictor-corrector: very good at 8–15 steps. Example: 10 steps. |
| uni pc bh2 | UniPC with the B(h)=2 variant; sometimes better on SDXL. Example: 10 steps. |
| res multistep | Second-order exponential multistep (RES); good on flow models. Example: 20 steps on FLUX. |
| res multistep ancestral | RES with added noise. Example: 25 steps. |
| res multistep cfg++ | RES with the CFG++ correction. Example: cfg ~1. |
| gradient estimation | Uses an estimate of the denoiser gradient for a second-order update at first-order cost. Example: 15 steps. |
| er sde | Extended reverse-time SDE solver; stochastic with good detail. Example: 20 steps. |
| seeds 2 / seeds 3 | Exponential stochastic solvers of order 2 / 3. Example: 20 steps. |
| sa solver | Stochastic Adams solver: multistep with tunable noise. Example: 20–30 steps. |

### O.scheduler — sigma schedules
| Option | Tooltip |
|---|---|
| normal | Evenly spaced in the model's timesteps. Example: a neutral choice for any model. |
| karras | Spaces noise levels so more steps fall at low noise (fine detail). Example: with dpm++ 2M at 20–30 steps on SD/SDXL. |
| exponential | Evenly spaced in log noise; similar to karras with less tail. Example: with dpm++ 3M SDE. |
| sgm uniform | Uniform spacing that includes the final step; made for distilled models. Example: SDXL Turbo, LCM. |
| simple | Takes every n-th timestep of the model's own schedule. Example: SD3 and FLUX. |
| ddim uniform | The timestep spacing used by DDIM. Example: with the ddim sampler. |
| beta | Beta-distribution spacing: more steps at both high and low noise. Example: SDXL at 20 steps for crisper structure. |
| linear quadratic | Linear at first, quadratic at the end; designed for video flow models. Example: Mochi. |
| kl optimal | Spacing that minimises a divergence bound; good at low step counts. Example: 10–15 steps. |

### O.upscaleMethod
| Option | Tooltip |
|---|---|
| nearest exact | Copies the nearest pixel; blocky, no blur. Example: pixel art, masks. |
| bilinear | Smooth, slightly soft. Example: quick previews. |
| area | Averages pixels; best for shrinking. Example: making thumbnails. |
| bicubic | Smooth and a bit sharper than bilinear. Example: general enlarging. |
| lanczos | Sharpest classic filter; may ring at hard edges. Example: photos (images only). |
| bislerp | Spherical interpolation that keeps latent statistics; latents only. Example: latent hi-res fix. |

### O.blendMode
| Option | Tooltip |
|---|---|
| normal | Mixes the two by the blend factor. Example: 0.5 = half each. |
| multiply | Darkens: white leaves the base unchanged. Example: adding shadows. |
| screen | Lightens: black leaves the base unchanged. Example: adding glow. |
| overlay | Increases contrast using both images. Example: texture overlays. |
| soft light | Gentle contrast change. Example: subtle grading. |
| difference | Absolute difference; shows what changed. Example: comparing two renders. |

### O.predictionType (Model Sampling)
| Option | Tooltip |
|---|---|
| eps | The model predicts the added noise (SD 1.x, SDXL). Example: most SD 1.5 checkpoints. |
| v prediction | The model predicts velocity (SD 2.x 768, some SDXL fine-tunes). Example: an SD 2.1 768 model gives grey images unless this is set. |
| lcm | Latent-consistency parameterisation. Example: LCM LoRA on SD 1.5. |
| x0 | The model predicts the clean image directly. Example: some distilled models. |
| img to img | Special mode for models trained to start from an input image. Example: edit models that expect it. |

---

## N1. Loaders

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Load Checkpoint (`loader.checkpoint`) | checkpoint | model list *checkpoints* | A single file holding the diffusion model, text encoder(s) and VAE. Example: `sd_xl_base_1.0.safetensors` gives MODEL, TEXT_ENCODER and VAE outputs for SDXL. |
| | device | combo auto, vulkan0, vulkan1 … · **auto** | Which Vulkan GPU runs the model. *auto* uses the GPU chosen in Settings (`gpuDevice`). Example: *vulkan1* to render on a second card while the first drives the desktop. |
| | flashAttention | bool · **auto** | A faster, memory-saving attention kernel. *auto* turns it on for SD 1.x above 512² on Vulkan, where it was measured 5× faster. Example: on for SD 1.5 at 768². |
| | offloadToRam | bool · **auto** | Keep the weights in system RAM and stream them to the GPU, for cards with little VRAM. Measured almost free on SDXL (3.04 vs 2.89 s/step). Example: on for SDXL on an 8 GB card. |
| | idleUnloadMinutes | int 0 … 1440 · **10** | Unload the model after this many idle minutes; 0 keeps it loaded. Example: 0 on a dedicated server. |
| Load Diffusion Model (`loader.diffusionModel`) | model | model list *diffusion_models* | A diffusion model without text encoder or VAE (common for FLUX, SD3.5, Wan). Load the encoders and VAE separately. Example: `flux1-dev.safetensors`. |
| | weightDtype | combo default, fp8 e4m3, fp8 e4m3 fast, fp8 e5m2 · **default** | → C.weightDtype + *fp8 e4m3 fast* also computes in fp8 where the GPU supports it. |
| Load Text Encoders (`loader.textEncoders`) | encoder1 … encoder4 | model list *text_encoders* (2–4 optional) | One file per encoder; the family decides how many are needed. Example: FLUX needs `clip_l.safetensors` and `t5xxl_fp16.safetensors`; SD3.5 adds `clip_g.safetensors`; HiDream uses four. |
| | family | combo auto, sd1, sdxl, sd3, flux, hunyuan video, hidream, stable cascade, stable audio, mochi, ltxv, pixart, cosmos, lumina2, wan, chroma, ace, omnigen2, qwen image · **auto** | Which model family these encoders are for; *auto* guesses from the files. Example: set *flux* when a T5 file alone is ambiguous. |
| | device | combo auto, GPU weights in VRAM, GPU weights in RAM · **auto** | → C.device |
| Load VAE (`loader.vae`) | vae | model list *vae* + *preview decoders* + *pixel space* | The autoencoder to use instead of the checkpoint's. Example: `sdxl_vae_fp16fix.safetensors`; *taesdxl* decodes very fast at lower quality; *pixel space* for models that work directly on pixels. |
| Load LoRA (`loader.lora`) | lora | model list *loras* | A small add-on that teaches a style, character or concept. Example: `watercolor_v2.safetensors`; add its trigger word to the prompt if it has one. |
| | strengthModel | float −100 … 100 · 0.01 · **1.0** | How strongly the LoRA changes the diffusion model; negative values push away from it. Example: 0.7 for a softer style; −0.5 to suppress what the LoRA adds. |
| | strengthText | float −100 … 100 · 0.01 · **1.0** | How strongly the LoRA changes the text encoder (how words are understood). Example: 0 to apply only the visual part; 1.0 when the trigger word matters. |
| Load LoRA (model only) (`loader.loraModel`) | lora, strengthModel | as above | As Load LoRA, for models whose LoRAs do not touch the text encoder. Example: a FLUX style LoRA at 0.9. |
| Load ControlNet (`loader.controlNet`) | controlNet | model list *controlnet* | A guide model that makes the image follow a structure image (edges, depth, pose ...). Also loads T2I-Adapters and control-LoRAs. Example: `control_v11p_sd15_canny.safetensors`. |
| Load Diff ControlNet (`loader.diffControlNet`) | model, controlNet | MODEL; model list | A ControlNet stored as a difference to a base model; needs that model. Example: a diff-canny file with the SD 1.5 model it was made from. |
| Load Style Model (`loader.styleModel`) | styleModel | model list *style_models* | A model that turns an image embedding into style conditioning. Example: `flux1-redux-dev.safetensors` for FLUX image variations. |
| Load CLIP Vision (`loader.clipVision`) | clipVision | model list *clip_vision* | An image encoder that describes images the way the text encoder describes words. Needed for unCLIP, style models and image-to-video. Example: `clip_vision_h.safetensors` for Wan image-to-video. |
| Load Upscale Model (`loader.upscaleModel`) | upscaleModel | model list *upscale_models* | A super-resolution network (ESRGAN family, ×2/×4). Example: `4x-UltraSharp.pth` enlarges 1024² to 4096² with added detail. |
| Load Hypernetwork (`loader.hypernetwork`) | hypernetwork | model list *hypernetworks* | An older style add-on that modifies attention layers. Example: `anime_style.pt`. |
| | strength | float −10 … 10 · 0.01 · **1.0** | → C.strength |
| Load GLIGEN (`loader.gligen`) | gligen | model list *gligen* | A layout model that places described objects in boxes. Example: `gligen_sd14_textbox.safetensors` with GLIGEN Text Box nodes. |
| Load unCLIP Checkpoint (`loader.unclipCheckpoint`) | checkpoint | model list *checkpoints* | A checkpoint trained to take image embeddings as prompts; also outputs its CLIP-vision encoder. Example: `sd21-unclip-h.ckpt`. |
| Load PhotoMaker (`loader.photomaker`) | photomaker | model list *photomaker* | An identity model that keeps a person's face from reference photos. Example: `photomaker-v2.bin` with PhotoMaker Encode. |
| Load Preview Decoder (`loader.previewDecoder`) | decoder | model list *vae_approx* | A tiny VAE used for fast previews or quick drafts. Example: `taesdxl_decoder.safetensors`. |

## N2. Text and prompts

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Text Encode (`text.encode`) | text | string, multiline, dynamic prompts | The prompt. `(word:1.3)` weights a word, `((word))` is ×1.21, `\(` types a bracket, `embedding:name` uses a textual inversion, `{a\|b}` picks one option per seed, `//` starts a comment, `BREAK` starts a new 75-token chunk. Example: `a (red:1.2) fox, {winter\|autumn} forest, embedding:goodhands`. |
| | textEncoder | TEXT_ENCODER | → C.textEncoder |
| Text Encode SDXL (`text.encodeSdxl`) | textG | string multiline | Prompt for SDXL's large encoder (overall content and composition). Example: `a lighthouse on a cliff at dusk`. |
| | textL | string multiline | Prompt for SDXL's small encoder (style and details); often the same as textG. Example: `oil painting, thick brush strokes`. |
| | width, height | int 0 … 16384 · 8 · **1024** | The image size the model is told it was trained at. Larger values suggest higher-resolution training images. Example: 4096 × 4096 can sharpen detail even when rendering 1024². |
| | cropW, cropH | int 0 … 16384 · 8 · **0** | Tells SDXL how much of the training image was cropped. 0 means uncropped, which keeps subjects centred and whole. Example: 0, 0 avoids cut-off heads. |
| | targetWidth, targetHeight | int 0 … 16384 · 8 · **1024** | The size of the final image SDXL should aim for. Example: set to the actual render size, e.g. 1216 × 832. |
| Text Encode SDXL Refiner (`text.encodeSdxlRefiner`) | aestheticScore | float 0 … 1000 · 0.01 · **6.0** | Aesthetic score the refiner is conditioned on; higher means "prettier" in its training data. Example: 6.0 for the positive prompt, 2.5 for the negative. |
| | width, height | int 0 … 16384 · 8 · **1024** | → C.width / C.height (size conditioning, as in SDXL). Example: 1024 × 1024. |
| | text | string multiline | The refiner prompt. Example: the same prompt as the base pass. |
| Text Encode SD3 (`text.encodeSd3`) | clipL, clipG, t5 | string multiline | Separate prompts for SD3's three encoders; usually all the same. T5 handles long, detailed sentences best. Example: the same sentence in all three, or a short style tag in clipL and a long description in t5. |
| | emptyPadding | combo none, empty prompt · **none** | How empty prompt parts are filled. *none* gives zeros; *empty prompt* encodes an empty string. Example: *empty prompt* for some fine-tunes that were trained that way. |
| Text Encode FLUX (`text.encodeFlux`) | clipL, t5 | string multiline | Prompts for FLUX's two encoders; T5 carries most of the meaning. Example: t5 = `a red bicycle leaning on a blue door, morning light`; clipL can repeat it. |
| | guidance | float 0 … 100 · 0.1 · **3.5** | Distilled guidance strength baked into FLUX dev (not CFG). Higher follows the prompt more literally. Example: 3.5 for photos, 2.0 for softer, more natural images. |
| Set Text Encoder Last Layer (`text.clipSkip`) | stopAtLayer | int −24 … −1 · 1 · **−1** | Use an earlier layer of the text encoder ("clip skip"). −1 is the last layer. Example: −2 for many anime SD 1.5 models trained that way. |
| | textEncoder | TEXT_ENCODER | → C.textEncoder |
| Text Encode (weighting options) (`text.encodeAdvanced`) | text | string multiline | → as Text Encode. |
| | weighting | combo relative to empty, multiply, mean-normalised · **relative to empty** | How word weights are applied: scale against an empty prompt (default), multiply the vectors, or multiply then restore the average. Example: *mean-normalised* lets very high weights like 1.6 work without artefacts. |
| | tokenNormalisation | combo none, mean, length · **none** | Rescale weights across the prompt. Example: *length* divides the effect of a long weighted phrase among its tokens. |

## N3. Conditioning

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Combine Conditioning (`cond.combine`) | a, b | CONDITIONING | Use both conditionings; the model sees each and averages its predictions. Used with areas or masks for region prompts. Example: a "sunset sky" area prompt plus a "beach" prompt for the lower half. |
| Average Conditioning (`cond.average`) | to, from | CONDITIONING | The two prompts to blend into one before sampling. |
| | toStrength | float 0 … 1 · 0.01 · **1.0** | 1.0 keeps only *to*, 0.0 only *from*. Example: 0.5 for a cat that is half fox. |
| Concat Conditioning (`cond.concat`) | to, from | CONDITIONING | Append one prompt's tokens to another so the model reads them as one longer prompt. Example: add "film grain" without changing how the main prompt is weighted. |
| Set Area (`cond.setArea`) | width, height | int 64 … 16384 · 8 · **64** | Size of the region this prompt applies to, in pixels. Example: 512 × 1024 for the left half of a 1024² image. |
| | x, y | int 0 … 16384 · 8 · **0** | → C.x / C.y (top-left corner of the area). |
| | strength | float 0 … 10 · 0.01 · **1.0** | How much this area's prompt counts compared with others. Example: 1.5 to make the area's subject dominate. |
| Set Area (percent) (`cond.setAreaPercent`) | width, height, x, y | float 0 … 1 · 0.01 · **1, 1, 0, 0** | The same as Set Area, as fractions of the image, so it works at any resolution. Example: width 0.5, x 0.5 is the right half. |
| | strength | float 0 … 10 · **1.0** | → as Set Area. |
| Set Area Strength (`cond.setAreaStrength`) | strength | float 0 … 10 · **1.0** | Change only the strength of an existing area prompt. Example: 0.8. |
| Set Mask (`cond.setMask`) | mask | MASK | → C.mask (where this prompt applies). |
| | strength | float 0 … 10 · **1.0** | → as Set Area. |
| | area | combo default, mask bounds · **default** | *default* conditions the whole image with the mask as weight; *mask bounds* renders the masked box at full resolution, giving that region more detail. Example: *mask bounds* for a small face region. |
| Set Timestep Range (`cond.setTimestepRange`) | start, end | float 0 … 1 · 0.001 · **0, 1** | Use this prompt only between these fractions of the run. Example: start 0, end 0.4 for a composition prompt, combined with a detail prompt from 0.4 to 1. |
| Zero Out Conditioning (`cond.zeroOut`) | conditioning | CONDITIONING | Replace the prompt with zeros (a true "nothing" prompt). Example: as negative for SD3/FLUX instead of an empty text. |
| GLIGEN Text Box (`cond.gligenBox`) | gligen, textEncoder | GLIGEN, TEXT_ENCODER | Inputs for the layout model. |
| | text | string | What to place in the box. Example: `a yellow umbrella`. |
| | width, height, x, y | int 8 … 16384 · 8 · **64, 64, 0, 0** | Box size and position in pixels. Example: 256 × 256 at x 640, y 128 puts the umbrella top right. |
| unCLIP Conditioning (`cond.unclip`) | clipVisionOutput | CLIP_VISION_OUTPUT | The encoded reference image. Example: CLIP Vision Encode of a painting. |
| | strength | float −10 … 10 · 0.01 · **1.0** | How strongly the reference image steers the result. Example: 0.7 keeps the prompt in charge. |
| | noiseAugmentation | float 0 … 1 · 0.01 · **0** | Adds noise to the image embedding so the result copies it less exactly. Example: 0.2 for looser variations. |
| Inpaint Conditioning (`cond.inpaint`) | vae, pixels, mask | VAE, IMAGE, MASK | For inpainting models: adds the masked image to the conditioning. Example: the photo, its mask and the model's VAE. |
| | useNoiseMask | bool · **true** | Also restrict denoising to the mask so the rest stays pixel-identical. Example: off to let the model blend edges more freely. |
| Instruct Edit Conditioning (`cond.instructEdit`) | vae, pixels | VAE, IMAGE | For instruction-editing models (pix2pix style): the image to edit. Example: a photo with prompt "make it winter". |
| Reference Latent (`cond.referenceLatent`) | latent | LATENT (optional) | Attaches a reference image (as latent) that edit models such as FLUX Kontext and Qwen-Image-Edit look at. Example: VAE Encode of the image to edit. |
| Conditioning Properties (`cond.properties`) | strength, area, mask, hooks, timesteps | float **1.0**; combo; MASK; HOOKS; TIMESTEP_KEYFRAMES | Set several properties at once, including scheduled LoRA hooks. Example: strength 1.0, mask of the face, hooks from Create LoRA Hook. |

## N4. Control, vision and adapters

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Apply ControlNet (`control.apply`) | positive, negative | CONDITIONING | → C.positive / C.negative; both get the control so CFG stays balanced. |
| | controlNet | CONTROL_NET | The guide model. Example: a canny ControlNet. |
| | image | IMAGE | The structure image, already pre-processed (edge map, depth map, pose skeleton). Example: the output of Canny. |
| | vae | VAE (optional) | Needed by ControlNets that work in latent space (SD3/FLUX ones). Example: the checkpoint's VAE. |
| | strength | float 0 … 10 · 0.01 · **1.0** | How strictly the image follows the structure. Example: 1.0 for exact edges; 0.5 to keep only the rough shape. |
| | startPercent | float 0 … 1 · 0.001 · **0** | → C.startPercent |
| | endPercent | float 0 … 1 · 0.001 · **1** | → C.endPercent + Example: 0.8 frees the final steps for natural textures. |
| Set Union ControlNet Type (`control.unionType`) | type | combo auto, openpose, depth, soft edge, canny/lineart, normal, segment, tile, repaint · **auto** | Tells a union (all-in-one) ControlNet what kind of control image it gets. Example: *depth* when feeding a depth map. |
| Canny (`control.canny`) | lowThreshold, highThreshold | float 0.01 … 0.99 · 0.01 · **0.4, 0.8** | Edge detector thresholds: lower values find more (weaker) edges. Example: 0.1 / 0.3 for a detailed line map, 0.5 / 0.9 for main outlines only. |
| CLIP Vision Encode (`vision.encode`) | clipVision, image | CLIP_VISION, IMAGE | Encodes an image into an embedding for unCLIP, style models or video models. |
| | crop | combo center, none · **center** | *center* crops to a square first; *none* squashes the whole image. Example: *none* when the important part is near the edge. |
| Apply Style Model (`vision.applyStyle`) | conditioning, styleModel, clipVisionOutput | CONDITIONING, STYLE_MODEL, CLIP_VISION_OUTPUT | Adds image-derived style tokens to the prompt. |
| | strength | float 0 … 10 · 0.001 · **1.0** | → C.strength + Example: 0.4 lets the text prompt still change the subject. |
| | strengthType | combo multiply, attention bias · **multiply** | *multiply* scales the style tokens; *attention bias* changes how much attention they get, which is gentler at low strength. Example: *attention bias* at 0.3. |
| PhotoMaker Encode (`vision.photomakerEncode`) | photomaker, image, textEncoder | PHOTOMAKER, IMAGE, TEXT_ENCODER | Reference photos of one person (a batch of 1–4 works best). |
| | text | string multiline | Prompt containing the trigger word `img` after the class word. Example: `photo of a man img wearing a space suit`. |

## N5. Sampling

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Sampler (`sampling.sample`) | model, positive, negative, latent | MODEL, CONDITIONING ×2, LATENT | → C.model, C.positive, C.negative, C.latentIn |
| | seed | int 0 … 2⁶⁴−1 · **0** | → C.seed |
| | seedControl | combo fixed, increment, decrement, randomize · **randomize** | → C.seedControl |
| | steps | int 1 … 10000 · **20** | → C.steps |
| | cfg | float 0 … 100 · 0.1 · **family default** (7.5 SD 1.x, 7.0 SDXL) | → C.cfg |
| | sampler | combo (O.sampler) · **euler** | → C.sampler |
| | scheduler | combo (O.scheduler) · **normal** | → C.scheduler |
| | denoise | float 0 … 1 · 0.01 · **1.0** | → C.denoise |
| Sampler (advanced) (`sampling.sampleAdvanced`) | addNoise | bool · **true** | Add the seed's noise to the input first. Turn off when the input already carries leftover noise from a previous sampler. Example: off for the second stage of a base → refiner chain. |
| | seed, seedControl, steps, cfg, sampler, scheduler | as Sampler | → C.seed, C.seedControl, C.steps, C.cfg, C.sampler, C.scheduler |
| | startStep | int 0 … 10000 · **0** | First step of the schedule to run. Example: steps 30, start 20 runs only the last 10 steps (like denoise ≈ 0.33 but on the same schedule). |
| | endStep | int 0 … 10000 · **10000** | Step to stop at (exclusive). Example: steps 30, end 24 hands the last 6 steps to an SDXL refiner. |
| | returnWithNoise | bool · **false** | Return the latent before the final noise is removed, for chaining into another sampler. Example: on for the base pass when a refiner continues from step 24. |

## N6. Custom sampling

The split pipeline: **Noise** + **Guider** + **Sampler** + **Sigmas** → Sample (custom).

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Sample (custom) (`sampling.custom`) | noise, guider, sampler, sigmas, latent | NOISE, GUIDER, SAMPLER, SIGMAS, LATENT | Runs any sampler with any schedule and any guidance method. Outputs the result and the denoised estimate. Example: Random Noise + CFG Guider + dpm++ 2M + Karras Schedule. |
| Random Noise (`sampling.noiseRandom`) | seed, seedControl | as Sampler | → C.seed, C.seedControl |
| No Noise (`sampling.noiseNone`) | — | — | Supplies zero noise, for continuing from a latent that already has noise. |
| Basic Guider (`sampling.guiderBasic`) | model, conditioning | MODEL, CONDITIONING | Uses only the positive prompt (no CFG). Example: FLUX dev with Flux Guidance. |
| CFG Guider (`sampling.guiderCfg`) | model, positive, negative | MODEL, CONDITIONING ×2 | Standard classifier-free guidance. |
| | cfg | float 0 … 100 · 0.1 · **8.0** | → C.cfg |
| Dual CFG Guider (`sampling.guiderDualCfg`) | condA, condB, negative | CONDITIONING ×3 | Guides with two positive prompts at separate strengths (e.g. an edit instruction and an image). |
| | cfgA | float 0 … 100 · 0.1 · **8.0** | Guidance from the first prompt against the second. Example: 6 for the text instruction. |
| | cfgB | float 0 … 100 · 0.1 · **8.0** | Guidance from the second prompt against the negative. Example: 1.5 for the reference image. |
| | style | combo regular, nested · **regular** | *nested* applies cfgA on top of the result of cfgB, which some edit models expect. Example: *nested* for instruct-edit SDXL models. |
| Perpendicular Negative Guider (`sampling.guiderPerpNeg`) | model, positive, negative, emptyConditioning | MODEL, CONDITIONING ×3 | CFG where the negative only removes what does not overlap with the positive, so "no glasses" does not erase the face. |
| | cfg | float 0 … 100 · **8.0** | → C.cfg |
| | negativeScale | float 0 … 100 · 0.01 · **1.0** | Strength of the perpendicular negative. Example: 1.5 to push harder away from the negative. |
| Choose Sampler (`sampling.samplerSelect`) | sampler | combo (O.sampler) · **euler** | → C.sampler |
| Euler Ancestral Sampler (`sampling.samplerEulerA`) | eta | float 0 … 100 · 0.01 · **1.0** | Amount of fresh noise per step; 0 turns it into plain euler. Example: 0.5 for a middle ground. |
| | noiseScale | float 0 … 100 · 0.01 · **1.0** | Multiplier on that fresh noise. Example: 1.02 slightly increases texture. |
| DPM++ 2M SDE Sampler (`sampling.samplerDpmpp2mSde`) | solver | combo midpoint, heun · **midpoint** | Second-order correction method. Example: *heun* is a little sharper. |
| | eta, noiseScale | float · **1.0, 1.0** | As Euler Ancestral. Example: eta 0.7 for calmer textures. |
| | noiseGenerator | combo setting default, cpu-compatible, philox · **setting default** | Random generator for the SDE noise (made on the host, so it is identical on every GPU). *cpu-compatible* matches the generator most tools use for seeds. Example: *cpu-compatible* to reproduce a seed shared from another tool. |
| DPM++ 3M SDE / DPM++ SDE Sampler (`sampling.samplerDpmpp3mSde`, `sampling.samplerDpmppSde`) | eta, noiseScale, noiseDevice | as above | As DPM++ 2M SDE. |
| | r (SDE only) | float 0 … 100 · 0.01 · **0.5** | Position of the intermediate step. Example: 0.5 (midpoint). |
| DPM Adaptive Sampler (`sampling.samplerDpmAdaptive`) | order | int 2 … 3 · **3** | Solver order. Example: 2 for speed. |
| | rtol, atol | float · **0.05, 0.0078** | Relative and absolute error targets; smaller is more accurate and slower. Example: rtol 0.02 for a final render. |
| | hInit | float · **0.05** | First step size. Example: 0.05. |
| | pcoeff, icoeff, dcoeff | float · **0, 1, 0** | Step-size controller gains. Example: leave at defaults unless steps oscillate. |
| | acceptSafety | float · **0.81** | Safety factor when accepting a step. Example: 0.9 accepts more steps. |
| | eta, noiseScale | float · **0, 1** | Stochasticity, as above. Example: eta 0 for deterministic output. |
| LMS Sampler (`sampling.samplerLms`) | order | int 1 … 100 · **4** | Number of previous steps combined. Example: 4. |
| SA-Solver Sampler (`sampling.samplerSaSolver`) | predictorOrder, correctorOrder | int 1 … 6 · **3, 4** | Orders of the multistep predictor and corrector. Example: 3 / 4. |
| | etaStart, etaEnd | float 0 … 1 · 0.01 · **0.2, 0.8** | Fraction of the run where noise is injected. Example: 0.2–0.8. |
| | eta, noiseScale | float · **1.0, 1.0** | As above. |
| ER-SDE Sampler (`sampling.samplerErSde`) | solverType | combo ER-SDE, reverse ODE · **ER-SDE** | *reverse ODE* makes it deterministic. Example: ER-SDE for texture. |
| | maxStage | int 1 … 3 · **3** | Solver order. Example: 3. |
| | eta, noiseScale | float · **1.0, 1.0** | As above. |
| Basic Schedule (`sampling.scheduleBasic`) | model | MODEL | The model whose noise range is used. |
| | scheduler | combo (O.scheduler) · **simple** | → C.scheduler |
| | steps | int 1 … 10000 · **20** | → C.steps |
| | denoise | float 0 … 1 · **1.0** | → C.denoise (drops the noisiest part of the schedule). |
| Karras Schedule (`sampling.scheduleKarras`) | steps | int · **20** | → C.steps |
| | sigmaMax, sigmaMin | float 0 … 5000 · **14.61, 0.03** | Highest and lowest noise level. Example: SD 1.5 uses 14.6 → 0.03; lowering sigmaMax gives a gentler start. |
| | rho | float 0 … 100 · **7.0** | Curve shape: higher spends more steps at low noise. Example: 7 (paper value). |
| Exponential Schedule (`sampling.scheduleExponential`) | steps, sigmaMax, sigmaMin | as Karras | Noise falls evenly in log scale. Example: 20 steps, 14.6 → 0.03. |
| Poly-exponential Schedule (`sampling.schedulePolyExp`) | steps, sigmaMax, sigmaMin, rho | as Karras, rho **1.0** | Between exponential (rho 1) and steeper curves. Example: rho 2. |
| VP Schedule (`sampling.scheduleVp`) | steps | int · **20** | → C.steps |
| | betaD, betaMin, epsS | float · **19.9, 0.1, 0.001** | Variance-preserving schedule parameters. Example: defaults reproduce the classic VP curve. |
| Beta Schedule (`sampling.scheduleBeta`) | steps | int · **20** | → C.steps |
| | alpha, beta | float 0 … 50 · **0.6, 0.6** | Shape of the beta distribution; values under 1 put more steps at both ends. Example: 0.6 / 0.6. |
| Laplace Schedule (`sampling.scheduleLaplace`) | steps, sigmaMax, sigmaMin | as Karras | → C.steps; noise range. |
| | mu, beta | float · **0, 0.5** | Centre and width of the Laplace distribution in log noise. Example: beta 0.5. |
| Align-Your-Steps Schedule (`sampling.scheduleAys`) | modelType | combo SD1, SDXL, SVD · **SD1** | Uses a pre-optimised schedule for that family. Example: *SDXL* with 10 steps. |
| | steps, denoise | int **10**; float **1.0** | → C.steps, C.denoise |
| GITS Schedule (`sampling.scheduleGits`) | coeff | float 0.8 … 1.5 · 0.05 · **1.2** | Coefficient of the geometric search. Example: 1.2. |
| | steps, denoise | int **10**; float **1.0** | → C.steps, C.denoise |
| Optimal-Steps Schedule (`sampling.scheduleOptimal`) | modelType | combo FLUX, Wan, Chroma · **FLUX** | Pre-computed best steps for that family. Example: *FLUX* at 10 steps. |
| | steps, denoise | int **20**; float **1.0** | → C.steps, C.denoise |
| Turbo Schedule (`sampling.scheduleTurbo`) | model, steps, denoise | MODEL; int 1 … 10 **1**; float **1.0** | Schedule for one-to-few-step Turbo models. Example: 1 step for SDXL Turbo at 512². |
| Split Sigmas (`sampling.splitSigmas`) | step | int 0 … 10000 · **0** | Cuts a schedule in two at this step, for two-stage sampling. Example: 20 steps split at 15: base model runs 15, refiner runs 5. |
| Split Sigmas by Denoise (`sampling.splitSigmasDenoise`) | denoise | float 0 … 1 · **1.0** | Cuts so the second part covers this fraction. Example: 0.25 gives the last quarter to the second sampler. |
| Flip Sigmas (`sampling.flipSigmas`) | sigmas | SIGMAS | Reverses the schedule to go from image to noise (inversion). Example: find the noise that recreates a photo. |
| Set First Sigma (`sampling.setFirstSigma`) | sigma | float 0 … 20000 · **136.0** | Replaces the starting noise level. Example: 136 for a fully noised start on a high-noise model. |
| Extend Intermediate Sigmas (`sampling.extendSigmas`) | steps | int 1 … 100 · **2** | New steps inserted between each pair. Example: 2 turns 10 steps into 28. |
| | startAt, endAt | float · **−1, 12** | Only extend between these noise levels (−1 = any). Example: 0–2 adds steps only in the fine-detail phase. |
| | spacing | combo linear, cosine, sine · **linear** | How the new points are spaced. Example: *cosine* clusters them near the ends. |

## N7. Model sampling and model patches

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Model Sampling (discrete) (`model.samplingDiscrete`) | prediction | combo (O.predictionType) · **eps** | Overrides what the model predicts when the file does not say. Example: *v prediction* for an SD 2.1 768 checkpoint that renders grey. |
| | zeroSnr | bool · **false** | Rescale the schedule to reach true zero signal at the end; for models trained with zero terminal SNR. Example: on for zero-SNR v-prediction SDXL fine-tunes; gives real blacks. |
| Model Sampling (continuous EDM) (`model.samplingEdm`) | prediction | combo v prediction, edm, eps, playground v2.5, cosmos rflow · **v prediction** | Prediction type for continuous-noise models. Example: *playground v2.5* for that model. |
| | sigmaMax, sigmaMin | float · **120, 0.002** | Noise range of the model. Example: use the values from the model card. |
| Model Sampling (continuous V) (`model.samplingV`) | sigmaMax, sigmaMin | float · **500, 0.03** | Noise range for continuous v-prediction models. Example: model card values. |
| Model Sampling (flow shift) (`model.samplingShift`) | shift | float 0 … 100 · 0.01 · **3.0** | For flow models (SD3, AuraFlow, Wan, HiDream ...): higher shift spends more steps at high noise, which helps composition at high resolution. Example: 3.0 for SD3.5 at 1024²; 5–8 for Wan video. |
| Model Sampling (FLUX) (`model.samplingFlux`) | maxShift, baseShift | float 0 … 100 · **1.15, 0.5** | Shift range that FLUX scales with image size. Example: raise maxShift to 1.5 for very large images. |
| | width, height | int 16 … 16384 · 8 · **1024** | The image size the shift is computed for. Example: set to the render size. |
| Rescale CFG (`model.rescaleCfg`) | multiplier | float 0 … 1 · 0.01 · **0.7** | Pulls the guided prediction back toward the un-guided brightness, fixing over-saturation at high CFG and with zero-SNR models. Example: 0.7. |
| CFG-Zero* (`model.cfgZeroStar`) | zeroInitSteps | int 0 … 10 · **0** | Steps at the start where the prediction is zeroed (skips unreliable early guidance). Example: 1 for FLUX/Wan. |
| Adaptive Projected Guidance (`model.apg`) | eta | float −10 … 10 · **1.0** | How much of the guidance parallel to the prediction is kept; lower reduces over-saturation. Example: 0 removes the parallel part completely. |
| | normThreshold | float 0 … 50 · **5.0** | Caps the size of the guidance update. Example: 5; lower for gentler colours. |
| | momentum | float −5 … 1 · **0** | Averages guidance over steps; negative values sharpen. Example: −0.5. |
| FreeU (`model.freeu`) | b1, b2 | float 0 … 10 · 0.01 · **1.1, 1.2** (v2: **1.3, 1.4**) | Boost of the UNet backbone features in two depths: stronger structure and contrast. Example: 1.3 / 1.4 on SD 1.5. |
| | s1, s2 | float 0 … 10 · 0.01 · **0.9, 0.2** | Damping of skip-connection features: less high-frequency noise. Example: 0.9 / 0.2. |
| | version | combo v1, v2 · **v2** | v2 scales the boost by feature strength, which is safer. Example: *v2*. |
| HyperTile (`model.hypertile`) | tileSize | int 1 … 2048 · **256** | Splits self-attention into tiles to speed up large images on UNet models. Example: 256 at 1536²; too small hurts coherence. |
| | swapSize | int 1 … 128 · **2** | Randomises tile layout between steps to hide seams. Example: 2. |
| | maxDepth | int 0 … 10 · **0** | Deepest UNet level that is tiled. Example: 0 (top level only). |
| | scaleDepth | bool · **false** | Shrink tiles with depth. Example: on for 2048² images. |
| Perturbed-Attention Guidance (`model.pag`) | scale | float 0 … 100 · 0.01 · **3.0** | Extra guidance from a version of the model with broken self-attention; improves structure without a negative prompt. Costs one more model call per step. Example: 3.0 with cfg 4. |
| Self-Attention Guidance (`model.sag`) | scale | float −2 … 5 · 0.01 · **0.5** | Guidance from blurring the regions the model attends to; sharpens detail. Example: 0.5. |
| | blurSigma | float 0 … 10 · 0.1 · **2.0** | Blur strength used for that guidance. Example: 2.0. |
| Token Merging (`model.tome`) | ratio | float 0 … 1 · 0.01 · **0.3** | Merges similar image tokens in attention to save time at a small quality cost. Example: 0.3 speeds up 1024² SD 1.5 by roughly a quarter. |
| Deep Shrink (`model.deepShrink`) | blockNumber | int 1 … 32 · **3** | UNet block whose features are shrunk during early steps, letting a model render above its native size without duplicate subjects. Example: 3. |
| | downscaleFactor | float 0.1 … 9 · 0.001 · **2.0** | How much the features are shrunk. Example: 2 for rendering SD 1.5 at 1024². |
| | startPercent, endPercent | float · **0, 0.35** | → C.startPercent / C.endPercent. Example: 0 to 0.35 shrinks only during composition. |
| | downscaleAfterSkip | bool · **true** | Shrink after the skip connection is taken. Example: on. |
| | downscaleMethod, upscaleMethod | combo (O.upscaleMethod) · **bicubic** | → C.upscaleMethod |
| Differential Diffusion (`model.differentialDiffusion`) | strength | float 0 … 1 · **1.0** | Makes grey mask values mean "partially change": white changes fully, black stays, grey changes a bit. Example: a soft-edged mask for seamless inpainting. |
| Skip-Layer Guidance (`model.slg`) | layers | string · **"7, 8, 9"** | Transformer layers skipped in an extra guidance pass; improves anatomy on DiT models. Example: `7, 8, 9` for SD3.5 Medium. |
| | scale | float 0 … 10 · **3.0** | Strength of that guidance. Example: 3.0. |
| | startPercent, endPercent | float · **0.01, 0.15** | → C.startPercent / C.endPercent. |
| | rescale | float 0 … 10 · **0** | Rescales the result like Rescale CFG. Example: 0 (off). |
| Attention Multiply (`model.attentionMultiply`) | q, k, v, out | float 0 … 10 · 0.01 · **1.0** | Scale the query/key/value/output weights of self-attention (also exists for text encoders). Experimental. Example: k 1.1 strengthens global coherence. |
| Epsilon Scaling (`model.epsilonScaling`) | factor | float 0.5 … 1.5 · 0.001 · **1.005** | Slightly scales the noise prediction to counter exposure bias. Example: 1.005 adds detail in low-step runs. |
| Video Linear CFG (`model.videoLinearCfg`) | minCfg | float 0 … 100 · **1.0** | CFG rises linearly from this value on the first frame to the sampler's cfg on the last. Example: 1.0 with cfg 2.5 for SVD. |
| Video Triangle CFG (`model.videoTriangleCfg`) | minCfg | float 0 … 100 · **1.0** | As linear, rising then falling across frames. Example: 1.0. |

## N8. Latent

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Empty Latent Image (`latent.empty`) | width, height | int 16 … 16384 · 8 · **512** | → C.width / C.height |
| | batchSize | int 1 … 4096 · **1** | → C.batchSize |
| | format | combo auto (from model), 4-channel, 16-channel, 128-channel · **auto** | Latent layout. *auto* needs the optional model input. Example: 16-channel for SD3/FLUX when not connected. |
| VAE Decode (`latent.decode`) | latent, vae | LATENT, VAE | Turns a latent into an image. → C.vae |
| VAE Decode (tiled) (`latent.decodeTiled`) | tileSize | int 64 … 4096 · 32 · **512** | → C.tileSize |
| | overlap | int 0 … 4096 · 32 · **64** | → C.overlap |
| | temporalSize, temporalOverlap | int · **64, 8** | → C.temporalSize / C.temporalOverlap |
| VAE Encode (`latent.encode`) | pixels, vae | IMAGE, VAE | Turns an image into a latent, for image-to-image. Example: a photo before a Sampler with denoise 0.6. |
| VAE Encode (tiled) (`latent.encodeTiled`) | tileSize, overlap, temporalSize, temporalOverlap | as decode | → C.tileSize, C.overlap, C.temporalSize, C.temporalOverlap |
| VAE Encode for Inpaint (`latent.encodeInpaint`) | pixels, vae, mask | IMAGE, VAE, MASK | Encodes with the masked area blanked, for inpainting checkpoints. |
| | growMask | int 0 … 64 · **6** | Pixels to expand the mask so edges blend. Example: 16 for hard-edged masks. |
| Set Noise Mask (`latent.setNoiseMask`) | mask | MASK | → C.mask (only these areas are denoised; the rest is kept). Example: inpaint a face with a normal checkpoint. |
| Upscale Latent (`latent.upscale`) | method | combo (O.upscaleMethod) · **nearest exact** | → C.upscaleMethod |
| | width, height | int 0 … 16384 · 8 · **512** | Target size in pixels; 0 keeps the aspect ratio from the other side. Example: 1024 × 0 doubles a 512² latent. |
| | crop | combo disabled, center · **disabled** | → C.crop |
| Upscale Latent By (`latent.upscaleBy`) | method | combo · **bilinear** | → C.upscaleMethod |
| | scale | float 0.01 … 8 · 0.01 · **1.5** | Size multiplier. Example: 1.5 then a Sampler at denoise 0.5 is the classic hi-res fix. |
| Composite Latents (`latent.composite`) | destination, source | LATENT | Paste one latent into another. |
| | x, y | int 0 … 16384 · 8 · **0** | → C.x / C.y (where the source goes). |
| | feather | int 0 … 16384 · 8 · **0** | Soft edge width in pixels. Example: 32 hides the seam. |
| Composite Latents (masked) (`latent.compositeMasked`) | x, y, mask | int; MASK | As above, through a mask. |
| | resizeSource | bool · **false** | Resize the source to the destination first. Example: on when sizes differ. |
| Blend Latents (`latent.blend`) | factor | float 0 … 1 · 0.01 · **0.5** | Mix of the two latents. Example: 0.3 keeps 70 % of the first. |
| | mode | combo normal · **normal** | Blend formula. Example: *normal*. |
| Latent Add / Subtract (`latent.add`, `latent.subtract`) | a, b | LATENT | Element-wise arithmetic; experimental mixing. Example: subtract a plain latent to isolate a change. |
| Multiply Latent (`latent.multiply`) | multiplier | float −10 … 10 · 0.01 · **1.0** | Scale the latent. Example: 1.05 adds contrast before decoding. |
| Interpolate Latents (`latent.interpolate`) | ratio | float 0 … 1 · 0.01 · **1.0** | Spherical blend that keeps magnitude. Example: 0.5 for a mid-point between two seeds. |
| Batch Latents (`latent.batch`) | a, b | LATENT | Join two latents into one batch. Example: compare two seeds in one decode. |
| Latent from Batch (`latent.fromBatch`) | batchIndex | int 0 … 63 · **0** | → C.batchIndex |
| | length | int 1 … 64 · **1** | → C.length |
| Repeat Latent Batch (`latent.repeat`) | amount | int 1 … 64 · **1** | Copies of the latent in the batch. Example: 4 for four img2img variations. |
| Rebatch Latents (`latent.rebatch`) | batchSize | int 1 … 4096 · **1** | Split a list of latents into batches of this size (saves memory). Example: 2. |
| Crop Latent (`latent.crop`) | width, height | int 64 … 16384 · 8 · **512** | → C.width / C.height (crop size). |
| | x, y | int · **0** | → C.x / C.y |
| Rotate Latent (`latent.rotate`) | rotation | combo none, 90° right, 180°, 270° right · **none** | Rotate the latent. Example: 90° to make a portrait from a landscape. |
| Flip Latent (`latent.flip`) | direction | combo horizontal, vertical · **horizontal** | Mirror the latent. Example: horizontal. |
| Batch Seed Behaviour (`latent.batchSeedBehaviour`) | behaviour | combo random, fixed · **fixed** | *fixed* gives every batch item the same noise. Example: *fixed* to compare LoRA strengths with identical noise. |
| Save Latent (`latent.save`) | filenamePrefix | string · **latents/latent** | → C.filenamePrefix |
| Load Latent (`latent.load`) | file | list of `input/*.latent` | A saved latent. Example: continue a run later. |

## N9. Image

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Scale Image (`image.scale`) | method | combo (O.upscaleMethod) · **nearest exact** | → C.upscaleMethod |
| | width, height | int 0 … 16384 · 1 · **512** | Target size; 0 keeps aspect ratio from the other side. Example: 768 × 0. |
| | crop | combo · **disabled** | → C.crop |
| Scale Image By (`image.scaleBy`) | method, scale | combo; float 0.01 … 8 · **1.0** | → C.upscaleMethod; size multiplier. Example: 0.5 halves the image. |
| Scale to Megapixels (`image.scaleToPixels`) | method | combo · **nearest exact** | → C.upscaleMethod |
| | megapixels | float 0.01 … 16 · 0.01 · **1.0** | Target pixel count, keeping aspect ratio. Example: 1.0 gives about 1024² for any input. |
| Upscale with Model (`image.upscaleModel`) | upscaleModel, image | UPSCALE_MODEL, IMAGE | Enlarge with a super-resolution network (fixed factor, e.g. ×4). Example: follow with Scale Image By 0.5 for a clean ×2. |
| | tileSize | int 128 … 2048 · 32 · **512** | → C.tileSize |
| Invert Image (`image.invert`) | image | IMAGE | Photographic negative. Example: turn a white-on-black sketch into black-on-white. |
| Batch Images (`image.batch`) | a, b | IMAGE | Join into one batch (b is resized to a). |
| Image from Batch (`image.fromBatch`) | batchIndex, length | int · **0, 1** | → C.batchIndex, C.length |
| Repeat Image Batch (`image.repeat`) | amount | int 1 … 4096 · **1** | Copies in the batch. Example: 4. |
| Crop Image (`image.crop`) | width, height | int 1 … 16384 · **512** | Crop size. Example: 512 × 512 face crop. |
| | x, y | int · **0** | → C.x / C.y |
| Pad for Outpainting (`image.padOutpaint`) | left, top, right, bottom | int 0 … 16384 · 8 · **0** | Pixels to add on each side; also outputs a mask of the new area. Example: right 256 extends a scene to the right. |
| | feathering | int 0 … 16384 · **40** | Soft mask edge reaching into the original. Example: 64 for smoother joins. |
| Blend Images (`image.blend`) | factor | float 0 … 1 · 0.01 · **0.5** | Mix amount. Example: 0.2 for a subtle overlay. |
| | mode | combo (O.blendMode) · **normal** | Blend formula; hover options. |
| Blur (`image.blur`) | radius | int 1 … 31 · **1** | Kernel radius in pixels. Example: 5 for a soft background. |
| | sigma | float 0.1 … 10 · 0.1 · **1.0** | Gaussian spread. Example: 3. |
| Sharpen (`image.sharpen`) | radius | int 1 … 31 · **1** | Unsharp-mask radius. Example: 2. |
| | sigma | float 0.1 … 10 · **1.0** | Spread. Example: 1. |
| | alpha | float 0 … 5 · 0.01 · **1.0** | Strength. Example: 0.5 for gentle sharpening after upscaling. |
| Quantize Colours (`image.quantize`) | colours | int 1 … 256 · **256** | Number of colours in the palette. Example: 16 for a retro look. |
| | dither | combo none, floyd-steinberg, bayer 2/4/8/16 · **floyd-steinberg** | Pattern used to hide banding. Example: *bayer 4* for a pixel-art grid. |
| Composite Images (masked) (`image.compositeMasked`) | x, y | int · **0** | → C.x / C.y |
| | resizeSource | bool · **false** | Resize source to destination first. Example: on for a full-frame overlay. |
| | mask | MASK (optional) | → C.mask |
| Stitch Images (`image.stitch`) | direction | combo right, left, up, down · **right** | Where the second image goes. Example: *right* for a before/after pair. |
| | matchSize | bool · **true** | Resize the second image to match. Example: on. |
| | spacing | int 0 … 1024 · 2 · **0** | Gap between them. Example: 16. |
| | spacingColour | combo white, black, red, green, blue · **white** | Gap colour. Example: *black*. |
| Rotate Image (`image.rotate`) | rotation | combo none, 90°, 180°, 270° · **none** | Example: 90°. |
| Flip Image (`image.flip`) | direction | combo horizontal, vertical · **horizontal** | Example: horizontal mirror. |
| Morphology (`image.morphology`) | operation | combo erode, dilate, open, close, gradient, top hat, bottom hat · **dilate** | Shape operation on bright areas: dilate grows, erode shrinks, open removes specks, close fills holes. Example: *close* to fill gaps in a line drawing. |
| | kernelSize | int 3 … 999 · **3** | Size of the operation. Example: 7. |
| Empty Image (`image.empty`) | width, height | int 1 … 16384 · **512** | → C.width / C.height |
| | batchSize | int 1 … 4096 · **1** | → C.batchSize |
| | colour | colour · **#000000** | Fill colour. Example: #808080 grey for an outpaint base. |
| Join Image and Alpha (`image.joinAlpha`) | image, alpha | IMAGE, MASK | Add a transparency channel (output PNG keeps it). Example: cut-out product shot. |
| Split Image and Alpha (`image.splitAlpha`) | image | IMAGE | Separate colour and transparency. |
| Porter-Duff Composite (`image.porterDuff`) | mode | combo add, clear, darken, dst, dst atop, dst in, dst out, dst over, lighten, multiply, overlay, screen, src, src atop, src in, src out, src over, xor · **dst** | Compositing operator for images with alpha. Example: *src over* puts the source on top. |

## N10. Mask

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Image to Mask (`mask.fromImage`) | channel | combo red, green, blue, alpha · **red** | Which channel becomes the mask. Example: *alpha* for a transparent PNG. |
| Mask to Image (`mask.toImage`) | mask | MASK | Show a mask as a grey image (for previews or saving). |
| Colour to Mask (`mask.fromColour`) | colour | colour · **#FFFFFF** | Pixels of exactly this colour become white. Example: #FF0000 to mask a red segmentation region. |
| Solid Mask (`mask.solid`) | value | float 0 … 1 · 0.01 · **1.0** | Uniform mask value. Example: 0.5 for half-strength everywhere. |
| | width, height | int 1 … 16384 · **512** | → C.width / C.height |
| Invert Mask (`mask.invert`) | mask | MASK | Swap masked and unmasked. Example: protect the subject, change the background. |
| Crop Mask (`mask.crop`) | x, y, width, height | int · **0, 0, 512, 512** | → C.x, C.y; crop size. |
| Combine Masks (`mask.composite`) | x, y | int · **0** | → C.x / C.y (offset of the source). |
| | operation | combo multiply, add, subtract, and, or, xor · **multiply** | How the masks combine. Example: *subtract* removes the face from a body mask. |
| Feather Mask (`mask.feather`) | left, top, right, bottom | int 0 … 16384 · **0** | Fade-out width at each edge. Example: 32 on all sides. |
| Grow Mask (`mask.grow`) | expand | int −16384 … 16384 · **0** | Pixels to grow (positive) or shrink (negative). Example: 8 to cover a hair outline. |
| | taperedCorners | bool · **true** | Rounded instead of square corners when growing. Example: on. |
| Threshold Mask (`mask.threshold`) | value | float 0 … 1 · 0.01 · **0.5** | Values above become 1, below become 0. Example: 0.3 keeps faint strokes. |
| Mask editor | brush size, hardness, opacity | editor 1 … 500 px · **40**; 0 … 1 · **0.8**; 0 … 1 · **1.0** | Brush for painting masks: size in pixels, edge hardness, paint strength. Example: size 20, hardness 0.5 for a soft edge around hair. |

## N11. Input and output

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Load Image (`io.loadImage`) | image | file list of `input/` + upload | An image from the input folder; drag a file, paste, or pick. Outputs IMAGE and a MASK from its alpha (transparent = masked). Animated files load all frames. Example: `photo.png`. |
| Load Image as Mask (`io.loadMask`) | image | file list + upload | → as Load Image. |
| | channel | combo alpha, red, green, blue · **alpha** | → as Image to Mask. |
| Save Image (`io.saveImage`) | filenamePrefix | string · **GraphicsHost** | → C.filenamePrefix |
| | format | combo png, jpeg, webp · **png** | File type. PNG is lossless and carries the workflow. Example: *webp* for small files. |
| | quality | int 1 … 100 · **90** | JPEG/WebP quality. Example: 95 for prints; 75 for sharing. |
| | embedWorkflow | combo setting default, on, off · **setting default** | Embed the workflow so the image reopens the graph when dropped on the canvas. Example: *off* before publishing images. |
| Preview Image (`io.previewImage`) | image | IMAGE | Show the image in the node without saving it (temp folder, cleared at start). |
| Save Animated WebP (`io.saveWebpAnim`) | filenamePrefix | string | → C.filenamePrefix |
| | fps | float 0.01 … 1000 · **6** | → C.fps |
| | lossless | bool · **false** | Lossless frames, larger file. Example: off. |
| | quality | int 0 … 100 · **80** | Lossy quality. Example: 80. |
| | method | combo fastest, default, slowest · **default** | Encoder effort: slower = smaller file. Example: *slowest* for final output. |
| Save Animated PNG (`io.saveApng`) | filenamePrefix, fps | string; float · **6** | → C.filenamePrefix, C.fps |
| | compressLevel | int 0 … 9 · **4** | PNG compression effort. Example: 9 for smallest files. |
| Webcam Capture (`io.webcam`) | width, height | int 0 … 4096 · **640, 480** | Capture size (0 = camera default). Example: 1280 × 720. |
| | captureOnQueue | bool · **true** | Take a new frame each time the graph is queued. Example: off to freeze one frame. |

## N12. Utility

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Integer / Float / Text / Boolean (`util.int`, `util.float`, `util.string`, `util.bool`) | value | as type | A constant that can feed several inputs at once. Example: one Integer feeding the width of three nodes. |
| Constant (any widget) (`util.primitive`) | value | takes the type of the input it connects to | A value node that copies the widget of the first input it is linked to, including its range and options, and supports *control after run*. Example: one seed shared by two samplers. |
| Reroute (`util.reroute`) | — | any | A pass-through point to tidy links. |
| Note (`util.note`), Markdown Note (`util.markdown`) | text | string multiline | Text for people reading the graph; not executed. Example: "Set denoise to 0.4 for subtle changes." |
| Join Text (`util.textJoin`) | a, b, separator | string; string · **", "** | Joins two texts. Example: style + subject with ", ". |
| Replace Text (`util.textReplace`) | text, find, replace, regex | string ×3; bool · **false** | Replace occurrences; regex enables patterns. Example: find `cat`, replace `dog`. |
| Substring (`util.textSlice`) | start, end | int · **0, −1** | Characters from start to end (−1 = end). Example: 0, 75. |
| Compare (`util.compare`) | a, b, operation | any; combo =, ≠, <, ≤, >, ≥ · **=** | Outputs a boolean. Example: width > 1024. |
| Switch (`util.switch`) | condition, onTrue, onFalse | BOOLEAN; any (lazy) | Passes one input; the other branch is not run. Example: run a hi-res pass only when a toggle is on. |
| Math (`util.math`) | expression, a, b, c | string · **"a + b"**; numbers | Arithmetic expression on inputs (`+ − × / % ^ min max round floor ceil`). Example: `a * 1.5` to compute a hi-res width. |
| Preview Any (`util.previewAny`) | value | any | Shows the value as text in the node. Example: inspect the seed actually used. |
| Image Size (`util.imageSize`) | image | IMAGE | Outputs width, height and batch count. Example: feed an Empty Latent with the size of a reference photo. |
| Subgraph node | (promoted widgets) | as promoted | Its inputs carry the tooltips of the widgets they were promoted from, editable in the subgraph's properties. |

## N13. Hooks (scheduled LoRAs)

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Create LoRA Hook (`hooks.lora`) | lora | model list *loras* | A LoRA applied only where the hook is attached (a prompt, a region, a time range). Example: a style LoRA only on the background prompt. |
| | strengthModel, strengthText | float −20 … 20 · **1.0** | As Load LoRA. |
| | previousHooks | HOOKS (optional) | Chain several hooks. |
| Hook Keyframe (`hooks.keyframe`) | strengthMultiplier | float −20 … 20 · **1.0** | Hook strength from this point on. Example: 0 at 0.7 to turn the LoRA off for the last 30 %. |
| | startPercent | float 0 … 1 · **0** | → C.startPercent |
| Hook Keyframes (interpolated) (`hooks.keyframesInterp`) | strengthStart, strengthEnd | float · **1.0, 1.0** | Strength at the start and end of the range. Example: 1.0 → 0.0 fades the LoRA out. |
| | interpolation | combo linear, ease in, ease out, ease in-out · **linear** | Curve between them. Example: *ease out*. |
| | startPercent, endPercent | float · **0, 1** | → C.startPercent / C.endPercent |
| | keyframeCount | int 1 … 100 · **5** | Number of steps in the fade. Example: 10 for a smooth change. |
| Set Text Encoder Hooks (`hooks.setText`) | applyToConditioning | bool · **true** | Also apply the hook when prompts are encoded. Example: on so the LoRA's trigger word works. |
| | scheduleText | bool · **false** | Re-encode prompts at each keyframe (slower, exact). Example: on when strength changes a lot over time. |
| Attach Hooks (`hooks.attach`) | conditioning, hooks | CONDITIONING, HOOKS | Bind hooks to one prompt. |

## N14. Merging and saving models

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Merge Models (`merge.simple`) | ratio | float 0 … 1 · 0.01 · **1.0** | Weight of the first model in the mix (1 = only first). Example: 0.5 for an even blend of two SDXL fine-tunes. |
| Merge Models by Block (`merge.blocks`) | input, middle, output | float 0 … 1 · **1.0** | Separate ratios for UNet input, middle and output blocks; inputs shape composition, outputs shape style. Example: input 1.0, output 0.3 keeps model A's composition with B's style. |
| Merge Models (add difference) (`merge.add`, `merge.subtract`) | multiplier | float −10 … 10 · **1.0** | Add or subtract the difference between models. Example: (inpainting − base) added to a fine-tune makes an inpainting fine-tune. |
| Merge Text Encoders (`merge.text`) | ratio | float 0 … 1 · **1.0** | As Merge Models, for text encoders. |
| Save Checkpoint (`merge.saveCheckpoint`) | filenamePrefix | string · **checkpoints/merged** | → C.filenamePrefix (saved to the checkpoints folder as safetensors with metadata). |
| Save Model / Text Encoder / VAE (`merge.saveModel`, `merge.saveText`, `merge.saveVae`) | filenamePrefix | string | → C.filenamePrefix |
| Extract LoRA (`merge.extractLora`) | rank | int 1 … 4096 · **16** | Size of the LoRA made from the difference between two models; higher keeps more but is larger. Example: 32 gives about 150 MB for SDXL. |
| | type | combo standard, full diff · **standard** | *full diff* stores the exact difference (large). |
| | biasDiff | bool · **true** | Also store bias changes. Example: on. |
| | filenamePrefix | string · **loras/extracted** | → C.filenamePrefix |

## N15. Family-specific image nodes

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Flux Guidance (`flux.guidance`) | guidance | float 0 … 100 · 0.1 · **3.5** | → as Text Encode FLUX guidance. Example: 2.5 for more natural skin. |
| Kontext Image Scale (`flux.kontextScale`) | image | IMAGE | Resizes an input image to the nearest size FLUX Kontext was trained on. Example: before VAE Encode for editing. |
| Empty Latent (SD3/FLUX) (`sd3.emptyLatent`) | width, height, batchSize | int · **1024, 1024, 1** | → C.width, C.height, C.batchSize (16-channel latent). |
| Stable Cascade Empty Latents (`cascade.emptyLatent`) | width, height | int · **1024** | → C.width / C.height |
| | compression | int 4 … 128 · **42** | Stage-C compression: higher is faster and coarser. Example: 32 for more detail. |
| | batchSize | int · **1** | → C.batchSize |
| Stable Cascade Stage-B Conditioning (`cascade.stageB`) | stageC | LATENT | The stage-C result that guides stage B. |
| Stable Cascade Stage-C Encode (`cascade.stageCEncode`) | compression | int · **42** | As above, for image-to-image. |
| Text Encode Hunyuan-DiT (`hunyuandit.encode`) | bert, mt5 | string multiline | Prompts for the two encoders (Chinese or English). Example: same prompt in both. |
| Qwen-Image Edit Encode (`qwen.editEncode`) | prompt, image, vae | string; IMAGE; VAE | Edit instruction plus the image to edit. Example: "replace the sign text with OPEN". |
| Model Sampling (AuraFlow/SD3 shift) | shift | see N7 | → `model.samplingShift`. |
| PixArt resolution hint (`pixart.resolution`) | width, height | int · **1024** | Size conditioning for PixArt models. Example: render size. |

## N16. Video

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Load Video (`video.load`) | file | file list + upload | A video from the input folder; outputs frames (IMAGE), audio and frame rate. Example: `dance.mp4`. |
| | frameLimit, skipFirst, everyNth | int · **0, 0, 1** | Frames to load (0 = all), frames to skip, keep every n-th. Example: everyNth 2 halves the frame rate. |
| Save Video (`video.save`) | filenamePrefix | string | → C.filenamePrefix |
| | format | combo mp4 (H.264), webm (VP9) · **mp4** | Container and codec. Example: *webm* for transparent-safe web clips. |
| | quality | int 0 … 51 · **23** | Constant-quality value; lower is better and larger. Example: 18 for near-lossless. |
| Create Video (`video.create`) | images, fps, audio | IMAGE; float · **24**; AUDIO (optional) | Combine frames (and sound) into a video value. → C.fps |
| Video Image-to-Video (SVD) (`svd.conditioning`) | width, height | int · **1024, 576** | → C.width / C.height |
| | frames | int 1 … 4096 · **14** | Number of frames. Example: 25 for SVD-XT. |
| | motionBucket | int 1 … 1023 · **127** | Amount of motion: higher moves more. Example: 180 for a dramatic camera move. |
| | fps | int 1 … 1024 · **6** | Frame-rate conditioning. Example: 6. |
| | augmentation | float 0 … 10 · 0.01 · **0** | Noise added to the start image: more lets the video deviate from it. Example: 0.05. |
| Wan Image-to-Video (`wan.imageToVideo`) | width, height | int · **832, 480** | → C.width / C.height |
| | length | int 1 … 1000 · 4 · **81** | → C.length (Wan wants 4n+1 frames). |
| | batchSize | int · **1** | → C.batchSize |
| | startImage, endImage, clipVisionOutput | IMAGE ×2, CLIP_VISION_OUTPUT (optional) | First and/or last frame to animate between. Example: first and last frame of a turnaround. |
| Empty Video Latent (`video.emptyLatent`) | width, height, length, batchSize | int · family defaults | → C.width, C.height, C.length, C.batchSize |
| LTX-Video Conditioning (`ltxv.conditioning`) | frameRate | float 0 … 1000 · **25** | → C.fps |
| LTX-Video Image-to-Video (`ltxv.imageToVideo`) | strength | float 0 … 1 · **1.0** | How closely the first frame matches the image. Example: 0.9. |
| Hunyuan Video Text Encode (`hunyuanvideo.encode`) | prompt | string | → as Text Encode. |
| | imageInterleave | int 1 … 512 · **2** | For image-to-video: how much the image counts compared with the text; lower is stronger image influence. Example: 2. |

## N17. Audio

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Load Audio (`audio.load`) | file | file list + upload | An audio file from the input folder. Example: `beat.wav`. |
| Save Audio (`audio.save`) | filenamePrefix | string | → C.filenamePrefix |
| | format | combo flac, mp3, opus · **flac** | File type. Example: *opus* for small files. |
| | quality | combo by format (e.g. mp3 V0/128k/320k, opus 64k–320k) · **format default** | Encoder quality. Example: mp3 320k. |
| Preview Audio (`audio.preview`) | audio | AUDIO | Play it in the node without saving. |
| Empty Audio Latent (`audio.emptyLatent`) | seconds | float 1 … 1000 · 0.1 · **47.6** | Length of the clip. Example: 10 for a short sound effect. |
| | batchSize | int · **1** | → C.batchSize |
| Text Encode ACE-Step (`ace.encode`) | tags | string | Genre and style tags. Example: `lo-fi, piano, 80 bpm`. |
| | lyrics | string multiline | Lyrics with section markers. Example: `[verse] rain on the window ...`. |
| | lyricsStrength | float 0 … 10 · **1.0** | How strongly the lyrics are followed. Example: 0.8 for looser phrasing. |
| VAE Encode / Decode Audio (`audio.encode`, `audio.decode`) | audio/latent, vae | AUDIO/LATENT, VAE | Convert between audio and latents. |

## N18. 3D

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Empty 3D Latent (`3d.emptyLatent`) | resolution | int 1 … 8192 · **3072** | Latent size of the shape model: higher is finer. Example: 4096 for detailed figurines. |
| | batchSize | int · **1** | → C.batchSize |
| Decode 3D Latent (`3d.decode`) | chunks | int 1000 … 500000 · **8000** | Points evaluated per pass; lower saves memory. Example: 4000 on 8 GB GPUs. |
| | octreeResolution | int 16 … 512 · **256** | Voxel grid size. Example: 384 for smoother surfaces. |
| Voxels to Mesh (`3d.voxelToMesh`) | algorithm | combo surface net, basic · **surface net** | Meshing method. Example: *surface net* for smoother results. |
| | threshold | float −1 … 1 · 0.01 · **0.6** | Surface level. Example: lower to thicken thin parts. |
| Save GLB (`3d.saveGlb`) | filenamePrefix | string | → C.filenamePrefix |
| Load 3D / Preview 3D (`3d.load`, `3d.preview`) | file, width, height | file list; int · **1024, 1024** | Show a model in a viewer and render a view as image + mask. Example: render front view at 1024². |
| | camera | combo perspective, orthographic · **perspective** | Camera type. Example: *orthographic* for technical views. |
| | lightIntensity, background | float · **3**; colour | Viewer lighting and background. Example: white background for product shots. |

## N19. Training (stretch)

| Node (`id`) | Parameter | Type · range · default | Tooltip |
|---|---|---|---|
| Train LoRA (`train.lora`) | images, captions | IMAGE, STRING list | Training images and their captions. Example: 20 photos of one object with captions "a photo of sks mug". |
| | steps | int 1 … 100000 · **1000** | Training steps. Example: 1500 for a 20-image style. |
| | learningRate | float 1e−6 … 1 · **0.0005** | Step size of training; too high overfits or breaks. Example: 0.0001 for SDXL. |
| | rank | int 1 … 128 · **8** | LoRA size. Example: 16 for complex styles. |
| | batchSize | int · **1** | → C.batchSize |
| | optimizer | combo AdamW, Adam, SGD, RMSprop · **AdamW** | Training algorithm. Example: AdamW. |
| | loss | combo MSE, L1, Huber, SmoothL1 · **MSE** | Error measure. Example: Huber for noisy datasets. |
| | seed | int · **0** | → C.seed |
| | filenamePrefix | string · **loras/trained** | → C.filenamePrefix |

---

## S. Settings

### S.0 Settings that already exist

The Settings tab shows these as tooltips (W1.6 checks that every key of `settings.json` has one with an example).

| Key | Values · default | Tooltip |
|---|---|---|
| `backend` | auto, vulkan, cpu · **auto** | Which compute backend runs models. *auto* uses a usable Vulkan GPU and falls back to the CPU; workflows always need Vulkan. Example: *vulkan* to get an error instead of a slow CPU fallback. |
| `gpuDevice` | index · **0** | Which Vulkan GPU runs the models when there are several. Example: 1 selects the second card in the list. |
| `interfaceMode` | user, power, developer · **user** | How much of the Generate tab is shown. Example: *power* adds sampler, steps, CFG and device; *developer* also shows the request JSON. |
| `theme` | system, light, dark · **system** | Window colours. Example: *dark* stays dark even when Windows is set to light. |
| `accent` | colour · **default blue** | Colour of selections and links. Example: `#2b6cb0`; leave empty for the default. |
| `notifications` | bool · **true** | Tray messages when a download finishes or an image is ready while the window is in the background. Example: off while screen sharing. |
| `server.enabled` | bool · **true** | Run the API server that serves the browser UI and the OpenAI-compatible and native APIs. Example: off if you only use the desktop window. |
| `server.port` | 1 … 65535 · **1234** | The port the server listens on. Example: 8080 when 1234 is already taken. |
| `server.lan` | bool · **false** | Let other computers connect, not just this one; needs an access token. Example: on to open `http://192.168.1.20:1234` on your phone. |
| `server.requireToken` | bool · **false** | Also require an access token from clients on this computer (remote clients always need one). Example: on when other people share this PC. |
| `server.https` | bool · **false** | Serve over https; a self-signed certificate is made when no files are given. Example: on when connecting over a network you do not trust. |
| `server.tlsCertFile`, `server.tlsKeyFile` | file · **none** | The PEM certificate chain and private key to use for https instead of the self-signed pair. Example: `C:\certs\host.pem` and `C:\certs\host.key`. |
| `server.allowedAddresses` | list · **empty (anyone who may connect)** | Only these addresses may connect; this computer always may. Example: `192.168.1.0/24, 10.0.0.5`. |
| `server.rateLimitPerMinute` | 0 … 100000 · **0** | Requests per client and minute before the server answers "too many requests"; 0 means no limit. Example: 120 for a shared server. |
| `logging.level` | trace, debug, info, warn, error · **info** | How much the log records. Example: *debug* while reporting a problem, *warn* for quiet long-running servers. |
| `safety.enabled` | bool · **false** | Check generated images with the optional safety filter. Example: on for a family computer. |

### S.1 Engine and server (`settings.json`)

Settings that already exist (`backend`, `gpuDevice`, `interfaceMode`, `theme`, `accent`, `notifications`, `server.*`,
`logging.level`, `safety.enabled`) are in S.0 above; W1.6 checks that each of them also
has an example. Rows marked *(existing)* below get new wording because the workflow system uses them too.

| Key | Values · default | Tooltip |
|---|---|---|
| `memory.mode` | gpu-only, high, normal, low, stream · **normal** | How models use VRAM; every mode computes on the Vulkan GPU. *high* keeps everything loaded; *normal* moves models in and out as needed; *low* keeps part of a model in RAM; *stream* keeps all weights in RAM and streams them each step. Example: *low* for SDXL on a 6 GB card. |
| `memory.reserveVramMb` | 0 … 65536 · **512** | VRAM left free for the desktop and other apps. Example: 1024 while gaming or streaming. |
| `memory.asyncOffload` | bool · **true** | Move weights between RAM and VRAM in the background while the GPU works. Example: off if the driver is unstable. |
| `graph.cachePolicy` | last, lru, ram, none · **last** | Which node results are kept between runs, so unchanged parts are skipped. Example: *lru* with size 20 when switching between several workflows. |
| `graph.cacheSize` | 1 … 1000 / GB · **10** | Entries for *lru*, or free-RAM threshold in GB for *ram*. Example: 16 GB with *ram*. |
| `graph.previewMethod` | none, fast, decoder · **fast** | Live preview during sampling: *fast* is a rough colour map, *decoder* uses a tiny VAE (sharper, slightly slower). Example: *decoder* to judge faces early. |
| `graph.previewEveryNSteps` | 1 … 100 · **1** | How often a preview is made. Example: 5 to save a little time on long video runs. |
| `graph.previewMaxSize` | 64 … 2048 · **512** | Longest side of previews in pixels. Example: 256 over a slow LAN. |
| `graph.rng` | cpu-compatible, philox · **cpu-compatible** | Random generator for noise, run on the host so the noise is identical on any GPU. *cpu-compatible* matches the generator most tools use for seeds. Example: keep *cpu-compatible* to share reproducible seeds. |
| `graph.historyMaxItems` | 0 … 1 000 000 · **10000** | History entries kept (0 = unlimited). Files are not deleted. Example: 500 for a small database. |
| `graph.seedControlTiming` | after, before · **after** | Whether seeds change after or before queueing. Example: *before* to see the seed that will be used. |
| `precision.diffusion` | auto, fp32, fp16, bf16, fp8 e4m3, fp8 e5m2 · **auto** | Default number format of diffusion models in memory. Example: *fp8 e4m3* to fit FLUX in 12 GB. |
| `precision.textEncoder` | auto, fp32, fp16, bf16, fp8 e4m3 · **auto** | Same for text encoders. Example: *fp8 e4m3* halves T5's 9 GB. |
| `precision.vae` | auto, fp32, fp16, bf16 · **auto** | Same for VAEs. Example: *fp32* if decoded images are black. |
| `precision.vaeFallbackFp32` | bool · **true** | Retry a decode in fp32 when the result contains invalid values. Example: keep on. |
| `threads` (existing) | 0 … 512 · **0** | CPU threads for host-side work in workflows (image nodes, sampler maths, encoding files; 0 = physical cores). Model computation always runs on the GPU. Example: 8 to keep the PC responsive. |
| `modelsDir` (existing) | folder · **DownloadedModels in the working directory** | Where models are found and downloaded to; kind subfolders (`loras/`, `vae/`, ...) are recognised. Example: `E:\AI\models`. |
| `paths.inputDir`, `paths.tempDir` | folder · **under the data folder** | Where uploaded inputs and temporary previews live; outputs stay in the gallery's outputs folder. Example: `D:\Inputs` for a shared input folder. |
| `model-paths.json` (file in the data folder) | per kind: list of folders · **none** | Extra model folders per kind. Example: `{"loras": ["\\\\nas\\loras"]}` adds a network share for LoRAs. |
| `output.embedWorkflow` | bool · **true** | Store the workflow inside saved files. Example: off for privacy when sharing. |
| `output.filenameCounterDigits` | 1 … 9 · **5** | Digits of the counter in file names. Example: 3 gives `fox_001.png`. |
| `server.maxUploadMb` | 1 … 4096 · **100** | Largest accepted upload. Example: 1024 for video inputs. |
| `plugins.enabled` | bool · **true** | Load installed plugins. Example: off to rule out a plugin when debugging. |
| `plugins.safeMode` | bool · **false** | Start once without plugins and turn this off again. Example: on after a plugin crash. |
| `plugins.registryUrl` | URL · **project default** | Where the plugin manager looks for plugins. Example: a company-internal index. |
| `users.multiUser` | bool · **false** | Separate settings, workflows and folders per user on a shared server. Example: on for a team machine. |
| `network.downloadsAllowed` | bool · **true** | Allow model and plugin downloads. Example: off on an air-gapped machine. |

### S.2 Editor (per user)

| Key | Values · default | Tooltip |
|---|---|---|
| `editor.tooltips` | bool · **true** | Show tooltips on hover. Example: off once you know every node. |
| `editor.tooltipDelayMs` | 0 … 5000 · **500** | Hover time before a tooltip appears. Example: 1000 to make them less intrusive. |
| `editor.linkStyle` | spline, linear, straight, hidden · **spline** | How links are drawn. Example: *straight* for circuit-like layouts. |
| `editor.snapToGrid` | bool · **false** | Snap nodes to the grid when moved. Example: on for tidy graphs. |
| `editor.gridSize` | 5 … 100 · **20** | Grid spacing in pixels. Example: 10 for finer placement. |
| `editor.showNodeIds` | off, all, selected · **off** | Show node id badges (useful with the API and filename tokens). Example: *all* while writing API scripts. |
| `editor.showMinimap` | bool · **true** | Overview of the whole graph in a corner. Example: off on small screens. |
| `editor.zoomSpeed` | 1.01 … 2.5 · **1.1** | Zoom per wheel step. Example: 1.25 for faster zoom. |
| `editor.invertScroll` | bool · **false** | Wheel pans instead of zooming (Ctrl+wheel zooms). Example: on for touchpads. |
| `editor.linkDropAction` | search, context menu, nothing · **search** | What happens when a link is dropped on empty canvas. Example: *context menu* to choose from compatible nodes. |
| `editor.autoQueue` | off, after run, on change · **off** | Queue again automatically. Example: *on change* to re-render while tweaking a prompt. |
| `editor.batchCount` | 1 … 1000 · **1** | How many times the Queue button queues the graph. Example: 8 overnight runs with *increment* seeds. |
| `editor.confirmClear` | bool · **true** | Ask before clearing the canvas. Example: keep on. |
| `editor.autoSaveSeconds` | 0 … 600 · **30** | Auto-save open workflows (0 = off). Example: 10. |
| `editor.promptHighlight` | bool · **true** | Colour weights, embeddings, comments and choices in prompts. Example: off for plain text. |
| `editor.numberDragSensitivity` | 0.1 … 10 · **1.0** | How fast number fields change when dragged. Example: 0.3 for precise CFG changes. |
| `editor.theme` | system, light, dark · **system** | Editor colours. Example: *dark*. |
| `editor.language` | system, available translations · **system** | Language of menus and tooltips. Example: *Deutsch*. |
| `editor.lowQualityZoom` | 0.1 … 1 · **0.5** | Below this zoom nodes are drawn simplified for speed. Example: 0.3 on fast GPUs. |
