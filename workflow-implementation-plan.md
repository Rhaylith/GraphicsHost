# GraphicsHost Workflows: implementation plan (C++)

This plan adds a **node-graph workflow system** to GraphicsHost. A user builds a generation pipeline from small nodes (load
a checkpoint, encode a prompt, sample, decode, upscale, save) in a visual editor, wires them together, and queues the
graph. A local engine runs only the parts that changed, keeps loaded models in memory, and streams progress and previews
back. The same graph can be saved as a file, embedded in the images it produced, and sent to an HTTP API.

It builds on [implementation-plan.md](implementation-plan.md) (phases 0 to 6: runtime DLLs, settings, model library,
HTTP server, Qt UI). Where the two disagree, this plan wins for the areas it covers (§14 lists the overrides).

Effort: **S** ≤ 2 d · **M** 3–5 d · **L** 6–10 d · **XL** > 10 d (one engineer, includes tests).

The per-parameter tooltip text is in [docs/workflow-node-reference.md](docs/workflow-node-reference.md). That document is
part of this plan: every node and setting it lists is a deliverable, and its tooltip text is the text the UI shows.

---

## 1. Ground rules

1. **All code is C++20.** Engine, executor, server, desktop editor, browser editor and plugins. The browser editor is the
   same C++ editor compiled to WebAssembly (§9). The only non-C++ files are build files (`.vcxproj`, `.props`), data
   (JSON, translations, shaders compiled by `glslc`), documentation, and the HTML/JS loader stub that Qt's WebAssembly
   tooling generates (never hand-edited).
2. **Clean room.** The design comes from public documentation, user guides, published papers and general knowledge of
   how node-graph diffusion tools behave. Contributors **must not read the source code of any existing node-graph
   diffusion front end** while working on this project. Sampler, scheduler and guidance algorithms are implemented from
   their papers (§6.4 lists them). The engine reuses stable-diffusion.cpp (MIT), which is already a dependency.
3. **Own names.** Node identifiers, display names, file formats, metadata keys, API routes and event names are our own
   (§4). Nothing in code, docs, file formats or UI names another node-graph product. CI checks this (step W0.5).
4. **Every changeable parameter has a tooltip with an example.** Node inputs, combo options where the options need
   explaining, editor settings and engine settings. A unit test fails the build if a registered parameter has no tooltip
   or its tooltip has no example (W1.6).
5. **Local first.** Everything runs offline once models are present. Network access happens only for model downloads, the
   plugin registry, and the update check, and each one is a setting that can be turned off.
6. **Hardware scope is unchanged from the main plan:** CPU and Vulkan on AMD in v1.

---

## 2. Feature inventory

What "all the features" means for this plan. Each row is delivered by the phase in the last column (§13).

| Area | Features | Phase |
|---|---|---|
| **Graph model** | Typed nodes with inputs, outputs and widgets; links; optional, required and hidden inputs; widgets convertible to inputs and back; constant (primitive) nodes; reroute points; groups; notes; node colours, titles, collapse, pin; node modes *active / muted / bypassed*; subgraphs (a selection becomes one reusable node with its own inputs and outputs) | W1, W8 |
| **Workflow files** | Editor format (layout, groups, subgraphs, viewport) and job format (just the nodes and values the engine needs); versioned schema with migrations; import/export; drag a workflow file **or an image/video/audio file that carries one** onto the canvas to open it | W1, W8 |
| **Execution** | Validation with per-node, per-input errors; run only the ancestors of output nodes; run up to one selected output ("partial run"); caching of node outputs by input fingerprint so only changed parts rerun; cache policies (keep last, LRU, RAM-pressure, none); lazy inputs (an input evaluated only when the node asks for it); list inputs and outputs (a node maps over a list or receives the whole list); node expansion (a node returns a sub-graph at run time, which makes loops and control flow possible); interrupt; per-node progress; previews during sampling | W2 |
| **Queue and history** | FIFO job queue with "add to front"; batch count (queue the same graph N times); auto-queue (re-queue when the previous run finishes, or when the graph changes); cancel a running or pending job; history with outputs, timings, status and errors; reopen any past job's graph | W2, W7 |
| **Models** | Model folders per type (checkpoints, diffusion models, text encoders, VAEs, LoRAs, ControlNets and adapters, CLIP-vision encoders, style models, upscale models, embeddings, hypernetworks, GLIGEN, PhotoMaker, preview decoders); extra model folders from a config file; hot rescan; companion detection; download of missing models referenced by a workflow | W3 |
| **Memory** | Load on demand, keep in VRAM while useful, offload to RAM, evict by LRU under memory pressure; partial loading for low VRAM; memory modes (*GPU only, high VRAM, normal, low VRAM, no VRAM, CPU*); reserved VRAM; precision per component (fp32, fp16, bf16, fp8 storage, ggml quantisations); "free memory" and "unload models" actions | W3 |
| **Weights and patches** | LoRA and LyCORIS variants (LoCon, LoHa, LoKr, DoRA, full diff) on model and text encoder with separate strengths; patches applied lazily and undone without reloading; scheduled LoRA strength over the sampling run (hooks and keyframes); model merging (simple, per block, add, subtract); save merged checkpoints and extract LoRAs | W3, W10 |
| **Text** | Tokenisers (CLIP BPE, T5/mT5 SentencePiece, LLM BPE for newer text encoders); prompt weighting `(word:1.2)`, nesting `((word))`, escaping `\(`; textual-inversion embeddings `embedding:name`; per-encoder prompts (SDXL `g`/`l`, SD3 `l`/`g`/`t5`); clip-skip; dynamic prompt choices `{red|green|blue}`; `//` and `/* */` comments in prompts | W4 |
| **Sampling** | One-node sampler and the split pipeline (noise source + guider + sampler + sigma schedule); ~35 samplers and 9 schedulers (§6.4); custom sigma tools (split, flip, set first, extend); model sampling types (eps, v-prediction, x0, LCM, flow / rectified flow with shift, EDM, zero-terminal-SNR); CFG variants (rescale, dual CFG, perpendicular negative, CFG++, APG, CFG-Zero*); reproducible seeds with a CPU RNG and a Philox RNG | W4, W10 |
| **Conditioning** | Combine, concat, average; area (pixels or percent) and strength; mask conditioning; timestep ranges; zero-out; SDXL size/crop conditioning; ControlNet, T2I-Adapter, control-LoRA, union ControlNet types, diff ControlNet, with strength and start/end; style models; CLIP-vision encode; unCLIP; GLIGEN boxes; inpaint conditioning; instruct-pix2pix; reference latents for edit models; PhotoMaker | W5 |
| **Latents, images, masks** | Empty latents per family; VAE encode/decode, tiled and temporal-tiled; inpaint encode with mask growth; latent upscale (several methods), composite, blend, arithmetic, batch tools, crop, flip, rotate, noise masks; image scale/crop/pad/blend/blur/sharpen/quantise/invert/stitch/morphology; masks from images and colours, mask maths, grow, feather, threshold; model upscalers (ESRGAN family) with tiling | W6 |
| **Inputs and outputs** | Load image (with alpha as mask), load mask, load latent, load video, load audio, webcam capture; save/preview image (PNG with embedded workflow, JPEG, WebP), animated WebP/PNG, video, audio, latents; filename prefixes with date and node-value tokens | W6, W11 |
| **Model families** | SD 1.x, SD 2.x (incl. 768 v-prediction), SDXL base/refiner/Turbo/Lightning, SD 3/3.5, FLUX.1 (dev, schnell, Fill, Kontext, Redux, depth/canny), Chroma, Stable Cascade, AuraFlow, PixArt α/Σ, Hunyuan-DiT, Kolors, Lumina 2, HiDream, Qwen-Image (incl. edit), OmniGen 2, Cosmos-Predict2, SDXL edit/instruct models | W4 (SD1/SDXL), W10 |
| **Model patches** | FreeU v1/v2, HyperTile, perturbed-attention guidance, self-attention guidance, token merging, deep shrink, differential diffusion, skip-layer guidance, attention multiply, epsilon scaling, latent/ CFG post-processing | W10 |
| **Video, audio, 3D** | Image-to-video (SVD), text/image-to-video (Wan 2.x, LTX-Video, Hunyuan Video, Mochi, Cosmos); video CFG schedules; audio generation (Stable Audio Open, ACE-Step) with audio VAE and save as FLAC/MP3/Opus; image-to-3D (Hunyuan3D 2) to voxels and mesh, save GLB, 3D preview | W11 |
| **Editor** | Infinite canvas, pan/zoom, minimap, grid snapping, alignment tools; node search (double-click canvas, or drop a link on empty canvas for a type-filtered search); node library sidebar; model library sidebar with previews; workflow browser; templates browser; queue and history (assets) sidebar; image previews in nodes; per-node progress and execution highlight; error badges on nodes and inputs; link styles; selection toolbox; copy/paste with or without links; undo/redo; workflow tabs; mask editor; image crop and compare views; settings dialog; keyboard shortcut editor; light/dark themes; translations | W8 |
| **Browser editor** | The same editor in a browser (WebAssembly), talking to the local server; works over LAN with tokens | W9 |
| **API** | Queue a job, read queue, cancel, interrupt, history, download outputs, upload inputs, node catalogue (all node definitions with types, ranges, defaults, tooltips), model lists, embeddings list, system stats, free memory; WebSocket event stream with status, start, cached nodes, executing node, step progress, node outputs, errors, interruption, completion, and binary preview frames | W7 |
| **Extensibility** | Native plugin DLLs with a C ABI for new nodes, new widget kinds (desktop), new model formats and new server routes; plugin registry and in-app plugin manager (install, update, disable, remove, safe mode); per-plugin settings | W12 |
| **Multi-user** | Optional user accounts on a shared server: own settings, workflows, inputs/outputs subfolders | W12 |
| **Training** | Train a LoRA from a folder of images inside a workflow (stretch) | W12 |

Out of scope: cloud "partner API" nodes that call paid remote services (this is a local host), and Python of any kind.

---

## 3. Architecture

```
GraphicsHost/
├─ runtime/                       existing sd-*.dll; ABI v2 adds components (§5)
│  ├─ gh_runtime.h                ABI v1 (single-shot generate, kept)
│  └─ gh_components.h             ABI v2: components, tensors, denoise step, VAE, encoders
├─ GraphicsHost.Core/
│  ├─ tensor/     Tensor (CPU float32/f16, up to 5-D), elementwise ops, resize, blur, morphology, image/mask codecs
│  ├─ text/       tokenisers (CLIP BPE, SentencePiece, LLM BPE), prompt parser (weights, embeddings, choices, comments)
│  ├─ models/     ModelFolders, ModelIndex, ComponentCache, MemoryManager, WeightPatcher (LoRA/hooks)
│  ├─ sampling/   noise/RNG, sigma schedules, model-sampling types, samplers, guiders, CFG patches
│  ├─ graph/      types, NodeDef + registry, Workflow (editor format), JobGraph (job format), Validator,
│  │              Executor, OutputCache, JobQueue, History, Events
│  ├─ nodes/      built-in nodes, one .cpp per category (loaders, conditioning, sampling, latent, image, mask, ...)
│  ├─ plugins/    plugin loader, registry client, plugin manager
│  └─ server/     /api/v1/graph routes, WebSocket events, uploads, file views
├─ GraphicsHost/ui/graph/         Qt editor: Canvas, NodeItem, LinkItem, widgets, panels, mask editor (§8)
├─ GraphicsHost.Web/              the same editor sources built with Qt for WebAssembly (§9)
├─ plugins/sdk/                   gh_node_plugin.h, C++ helper header, example plugin project
└─ GraphicsHost.Tests/            [unit] [api] [integration] [golden] tests
```

**Data flow of one job.** Editor (desktop or browser) → `Workflow` → converted to a `JobGraph` (muted nodes removed,
bypassed nodes short-circuited, subgraphs flattened, reroutes and constants folded) → `POST /api/v1/graph/jobs` or a direct
call to `JobQueue` → `Validator` → `Executor` on the **graph thread** → nodes call `ComponentCache` / `MemoryManager` /
`sampling` / runtime ABI v2 → events (`Events` bus) to the WebSocket and to the Qt editor → outputs written to the output
folder and the gallery index → `History`.

**Threads.** One graph thread runs jobs one at a time (the GPU is one resource). Node code may use a CPU thread pool for
image maths. The runtime DLL is called only from the graph thread (it is not re-entrant; ABI v1 rule). The UI thread
never blocks on the engine; it receives events through `QMetaObject::invokeMethod`.

**Relationship to the existing single-shot generator.** The Generate tab and `/v1/images/generations` keep working. In
phase W4 they are re-implemented as a fixed built-in workflow template, so there is one engine path and one place where
sampling is implemented. ABI v1 `gh_generate` stays as a reference for golden tests.

---

## 4. Graph model and file formats

### 4.1 Types

Value types carried on links. Each has a colour in the editor and a `typeId` in files.

| Type id | Meaning |
|---|---|
| `MODEL` | a diffusion model (UNet or DiT) plus its pending patches and sampling type |
| `TEXT_ENCODER` | one or more text encoders plus tokenisers, clip-skip setting and pending patches |
| `VAE` | an autoencoder (or a pixel-space pass-through) |
| `CONDITIONING` | a list of conditioning entries: embedding tensors, pooled vectors, and options (area, mask, strength, timestep range, control chain, reference latents, hooks) |
| `LATENT` | latent tensor `[B,C,(T),H,W]` + optional noise mask + batch index list + family tag |
| `IMAGE` | float RGB `[B,H,W,3]` in 0..1 (video = batch of frames + frame rate) |
| `MASK` | float `[B,H,W]` in 0..1 |
| `NOISE`, `GUIDER`, `SAMPLER`, `SIGMAS` | the split sampling pipeline |
| `CONTROL_NET`, `STYLE_MODEL`, `CLIP_VISION`, `CLIP_VISION_OUTPUT`, `UPSCALE_MODEL`, `GLIGEN`, `PHOTOMAKER` | auxiliary models and their outputs |
| `HOOKS`, `HOOK_KEYFRAMES`, `TIMESTEP_KEYFRAMES` | scheduled patches |
| `AUDIO`, `VIDEO`, `MESH`, `VOXEL`, `LOAD3D_CAMERA` | non-image media |
| `INT`, `FLOAT`, `STRING`, `BOOLEAN`, `COMBO` | widget value types (can also travel on links) |
| `*` | any (used by reroute, preview-any, switch) |

### 4.2 Node definition

Nodes are declared once, in C++, with a builder. The same declaration drives validation, the API catalogue, the editor
widgets, the tooltips and the docs generator.

```cpp
registry.add(NodeDef("sampling.sample")
    .display(QT_TR_NOOP("Sampler"))
    .category("sampling")
    .description(QT_TR_NOOP("Denoises a latent with a model, prompts and a sampler. Use for text-to-image and image-to-image."))
    .input(Model("model").tip(tips::kModel))
    .input(Int("seed").range(0, UINT64_MAX).def(0).seedControl().tip(tips::kSeed))
    .input(Int("steps").range(1, 10000).def(20).tip(tips::kSteps))
    .input(Float("cfg").range(0.0, 100.0).step(0.1).round(0.01).def(7.0).tip(tips::kCfg))
    .input(Combo("sampler").options(sampling::samplerNames()).optionTips(sampling::samplerTips()).tip(tips::kSamplingMethod))
    .input(Combo("scheduler").options(sampling::schedulerNames()).optionTips(sampling::schedulerTips()).tip(tips::kScheduler))
    .input(Conditioning("positive").tip(tips::kPositive))
    .input(Conditioning("negative").tip(tips::kNegative))
    .input(Latent("latent").tip(tips::kLatentIn))
    .input(Float("denoise").range(0.0, 1.0).step(0.01).def(1.0).tip(tips::kDenoise))
    .output(Latent("latent").tip(QT_TR_NOOP("The denoised latent. Connect to VAE Decode to get an image.")))
    .run(&sampleNode));
```

Definition flags (our names): `isOutput` (a sink such as Save Image; jobs run the ancestors of these), `lazy(input)`,
`acceptsLists` / `outputIsList(i)`, `fingerprint(fn)` (extra cache key, e.g. file modification time for Load Image, or
"always rerun" for webcam), `validate(fn)` (custom checks), `expands` (may return a sub-graph), `deprecated`,
`experimental`, `searchAliases` (extra words for search, e.g. "denoise", "generate").

Input kinds: required, optional, hidden (filled by the executor: node id, job id, the editor workflow for metadata,
user id). Widgets: `Int`, `Float` (with `display: number | slider`), `String` (single line or multiline, optional
dynamic-prompt processing), `Boolean` (toggle with on/off labels), `Combo` (static list, or a model folder that is
listed live, or an upload target), `Color`, `ImageUpload`, `Seed` (an `Int` with a *control after run* companion).

### 4.3 Editor workflow format (`*.ghwf.json`)

```json
{
  "format": "graphicshost.workflow", "version": 1,
  "id": "uuid", "revision": 12,
  "nodes": [ { "id": 7, "type": "sampling.sample", "title": "Sampler", "pos": [400, 120], "size": [320, 260],
               "mode": "active", "color": null, "collapsed": false, "pinned": false,
               "widgets": { "seed": 42, "seedControl": "randomize", "steps": 20, "cfg": 7.0, "sampler": "euler",
                            "scheduler": "karras", "denoise": 1.0 },
               "inputsAsSockets": ["seed"], "order": 5 } ],
  "links": [ { "id": 3, "from": [4, 0], "to": [7, "model"], "type": "MODEL" } ],
  "reroutes": [ { "id": 1, "pos": [300, 200], "link": 3 } ],
  "groups": [ { "id": 1, "title": "Upscale", "bounds": [0, 0, 800, 400], "color": "#335" } ],
  "subgraphs": [ { "id": "uuid", "name": "Hi-res fix", "inputs": [...], "outputs": [...], "nodes": [...], "links": [...] } ],
  "view": { "offset": [0, 0], "scale": 1.0 },
  "extra": { }
}
```

- Widget values are stored **by name**, not by position, so adding an input to a node never breaks old files.
- `version` + migration list, like `settings.json`. Unknown keys are kept (plugins put their data in `extra`).
- Missing node types are kept as placeholder nodes that show "missing: type id, plugin name" and keep their values.

### 4.4 Job format

What the engine runs, what the API accepts, and what is embedded in outputs next to the editor format:

```json
{ "format": "graphicshost.job", "version": 1,
  "nodes": { "4": { "type": "loader.checkpoint", "inputs": { "checkpoint": "sdxl/base.safetensors" } },
             "7": { "type": "sampling.sample",
                    "inputs": { "model": ["4", 0], "seed": 42, "steps": 20, "cfg": 7.0, "sampler": "euler",
                                "scheduler": "karras", "positive": ["5", 0], "negative": ["6", 0],
                                "latent": ["3", 0], "denoise": 1.0 } } },
  "outputs": ["9"], "client": { "id": "uuid", "label": "my run" } }
```

A link is `[nodeId, outputIndex]`. `outputs` is optional (default: all output nodes).

### 4.5 Metadata in outputs

- PNG: `iTXt` chunks `gh.workflow` (editor format) and `gh.job` (job format), compressed. WebP: XMP packet with the same
  two keys. Video (MP4/WebM): container metadata tag. FLAC/Opus/MP3: Vorbis comment / ID3 `TXXX`.
- Setting `output.embedWorkflow` (default on) and per-save-node override. A job sent through the API embeds only `gh.job`
  unless the client also sends the editor workflow.
- Dropping any such file onto the canvas opens the workflow (W8.17). The main plan's single-shot parameter chunk is kept
  for the Generate tab and is also readable.

---

## 5. Engine: runtime ABI v2 ("components")

ABI v1 calls one function that runs the whole pipeline. Node graphs need the pipeline in pieces: encode text once, run
the model once per step under a host-side sampler, decode with a chosen VAE. ABI v2 is added next to v1 in the same
DLLs.

```c
typedef enum gh_component_kind { GH_DIFFUSION, GH_TEXT_ENCODER, GH_VAE, GH_CONTROL, GH_CLIP_VISION,
                                 GH_UPSCALER, GH_STYLE, GH_PREVIEW_DECODER, GH_PHOTOMAKER, GH_AUDIO_VAE, ... } gh_component_kind;

typedef struct gh_tensor { uint32_t struct_size; int32_t dtype; int32_t ndim; int64_t shape[5]; void* data; } gh_tensor;

GH_API gh_component* gh_component_open(const gh_component_params*);      /* reads headers, detects architecture, no weights yet */
GH_API int32_t gh_component_info(const gh_component*, gh_component_info* out);  /* arch, family, sizes, latent format, prediction type */
GH_API int32_t gh_component_residency(gh_component*, int32_t where, int64_t device_bytes); /* load / partial / offload / free */
GH_API int32_t gh_component_set_patches(gh_component*, const gh_weight_patch*, int32_t n); /* LoRA etc., applied lazily */
GH_API int32_t gh_text_encode(gh_component*, const gh_token_batch*, gh_tensor* hidden, gh_tensor* pooled);
GH_API int32_t gh_denoise(gh_component*, const gh_denoise_args*, gh_tensor* out);  /* one model evaluation */
GH_API int32_t gh_control(gh_component*, const gh_control_args*, gh_control_residuals* out);
GH_API int32_t gh_vae_encode(gh_component*, const gh_tensor* image, const gh_tiling*, gh_tensor* latent);
GH_API int32_t gh_vae_decode(gh_component*, const gh_tensor* latent, const gh_tiling*, gh_tensor* image);
GH_API int32_t gh_clip_vision_encode(gh_component*, const gh_tensor* image, int32_t crop, gh_clip_vision_out* out);
GH_API int32_t gh_upscale(gh_component*, const gh_tensor* image, int32_t tile, gh_tensor* out);
GH_API void    gh_component_close(gh_component*);
```

- `gh_denoise_args`: latent `x`, timestep or sigma per batch item, a list of conditioning blocks (hidden states, pooled
  vector, area, mask, strength), control residuals, reference latents, and a **model patch list** (FreeU factors,
  attention patches for PAG/SAG/SLG, HyperTile, token merging, deep shrink, attention multipliers). The runtime evaluates
  the batched cond/uncond pass in one graph when memory allows, otherwise in sequence.
- **Sampling stays in the host** (`GraphicsHost.Core/sampling`). Every sampler, scheduler and guider is plain C++ over
  `Tensor`; only the model evaluation crosses the ABI. This makes every sampler available to every model family and keeps
  the runtime DLL small. Latents are small enough (an SDXL 1024² latent is 0.5 MB; a 5-second Wan video latent is tens of
  MB) that copying per step costs little next to the model evaluation. Measured in W0.
- **Tokenisers and prompt weighting are in the host** (`text/`), so the runtime gets token ids, per-token weights and
  injected embedding vectors and never parses prompts.
- Implementation inside the runtime: built from stable-diffusion.cpp's model classes (CLIP, T5, UNet, MMDiT, Flux,
  Wan, VAE, TAE, ESRGAN, ControlNet, PhotoMaker) driven through our own shim, plus our own ggml modules for
  architectures the pinned engine version lacks (W10). New ggml code lives in `runtime/arch/` with one file per
  architecture and the same internal interface.
- `struct_size` rules and error handling as ABI v1. `GH_ABI_VERSION` becomes 2; a v1-only DLL is still loadable for the
  Generate tab and reported as "graph features unavailable".

---

## 6. Core subsystems

### 6.1 Tensor library (`tensor/`)
A small owned tensor type: `float` and `f16` storage, shapes up to 5-D, strided views for batch slicing, and the ops the
nodes need (add, mul, lerp, slerp, clamp, mean/std per channel, interpolate nearest/bilinear/bicubic/area/lanczos/
bislerp, gaussian blur, sharpen, morphology, rotate/flip, pad, crop, concat). AVX2 paths through MSVC auto-vectorisation
and `<immintrin.h>` where measured; parallel loops on the CPU pool. Image codecs reuse libpng, libjpeg-turbo, libwebp.

### 6.2 Text (`text/`)
- Tokenisers: CLIP BPE (vocab + merges files embedded or loaded with the encoder), SentencePiece unigram (T5, mT5),
  byte-level BPE for LLM-style encoders (Qwen, Llama, Gemma families used by newer image models).
- Prompt parser, in this order: comments removed (`//` to end of line, `/* ... */`); dynamic choices `{a|b|c}` resolved
  with the node's seed (nested allowed, `\{` escapes); weights `(text:1.3)`, `(text)` = ×1.1, nested multiplies,
  `\(` `\)` escape; `embedding:name` or `embedding:name:0.8` injects a textual-inversion vector. Long prompts are split
  into 75-token chunks for CLIP encoders and concatenated; a `BREAK` keyword forces a new chunk.
- Weight application method: scale the token's hidden state relative to the empty-prompt hidden state (default), or plain
  multiply; a text-encoder option.

### 6.3 Models and memory (`models/`)
- **`ModelFolders`**: one folder per type under the models root, plus any extra roots listed in
  `model-paths.json` (per type: list of folders; a folder may serve several types). Files are listed with relative
  sub-paths (`sdxl/base.safetensors`), so combos stay stable when folders are added.
- **`ModelIndex`**: header scan (safetensors/GGUF/ckpt) gives architecture, family, size and hash (SHA-256 on demand,
  cached in SQLite); preview image next to the file (`name.png`/`name.preview.png`) for the model library sidebar.
- **`ComponentCache`**: an opened component is shared by every node that loads the same file with the same options;
  reference-counted handles in node outputs.
- **`MemoryManager`**: knows the device budget (from `GpuDevices`), every component's resident size, and an estimate of
  the activation memory of the next call (from the component's info and the latent size). Before each runtime call it
  makes room: evicts least-recently-used components to RAM, then from RAM to disk (close), then chooses partial
  residency. Memory modes (setting `memory.mode`): `gpu-only`, `high`, `normal` (default), `low` (partial residency
  allowed), `none` (stream weights each call), `cpu`. `memory.reserveVramMb` keeps room for the desktop.
- **`WeightPatcher`**: LoRA files are parsed into key-mapped low-rank (or full) deltas for the target component. A
  `MODEL`/`TEXT_ENCODER` value carries a list of `{patch, strength, schedule}`; the runtime receives the list and applies
  it when the weights are next used, keeping the originals so a different list can be applied without reloading.
  Supported: LoRA (with alpha), LoCon, LoHa, LoKr, DoRA, full diffs, and "set" patches (for merges).
- Precision: per component `auto | fp32 | fp16 | bf16 | fp8_e4m3 | fp8_e5m2 | q8_0 | q5_0 | q4_0 ...`; fp8 is a storage
  format converted per layer at run time (ggml supports the quantised ones natively).

### 6.4 Sampling (`sampling/`)
All implemented from the papers, as host C++ over `Tensor`:

| Group | Items | Sources |
|---|---|---|
| Model sampling (prediction) | eps, v-prediction, x0, LCM, rectified flow with shift and resolution-dependent shift, EDM, zero-terminal-SNR rescale | Ho 2020; Salimans & Ho 2022 (v); Karras 2022 (EDM); Lin 2023 (zero SNR); Esser 2024 (rectified flow) |
| Samplers | euler, euler ancestral, heun, heun++, dpm2, dpm2 ancestral, lms, dpm fast, dpm adaptive, dpm++ 2S ancestral, dpm++ SDE, dpm++ 2M, dpm++ 2M SDE, dpm++ 3M SDE, ddpm, ddim, lcm, ipndm, ipndm-v, deis, uni-pc (bh1, bh2), res multistep (+ ancestral), gradient estimation, ER-SDE, SEEDS-2/3, SA-solver; CFG++ variants of euler, euler ancestral, dpm++ 2S ancestral, dpm++ 2M, res multistep | Karras 2022; Lu 2022 (DPM-Solver, ++); Zhang & Chen 2022 (DEIS, iPNDM); Zhao 2023 (UniPC); Luo 2023 (LCM); Song 2020 (DDIM); Zhang 2023 (RES); Cui 2023 (ER-SDE); Gonzalez 2023 (SEEDS); Xue 2023 (SA-Solver); Chung 2024 (CFG++); Permenter & Yuan 2023 (gradient estimation) |
| Schedulers | normal (uniform in timestep), karras, exponential, sgm uniform, simple, ddim uniform, beta, linear-quadratic, KL-optimal; plus parametric nodes for karras, exponential, poly-exponential, VP, beta, laplace, align-your-steps, GITS, optimal steps, turbo | Karras 2022; Lee 2024 (beta); Sabour 2024 (AYS); Chen 2024 (GITS); Pham 2025 (optimal steps) |
| Guidance | CFG, rescale CFG, dual CFG (nested and regular), perpendicular negative, CFG-Zero*, adaptive projected guidance (APG), linear and triangle video CFG, skip-layer guidance, PAG, SAG | Ho & Salimans 2022; Lin 2023; Armandpour 2023; Fan 2025; Sadat 2024; Ahn 2024; Hong 2022 |
| Noise / RNG | `cpu` (mt19937 + Box–Muller, matching the CPU generator most tools use, so seeds are comparable), `philox` (counter-based, matches common GPU generators); per-batch-item seeds; ancestral noise device option; SDE noise via Brownian tree | — |

- Sampler interface: `sample(denoiseFn, x, sigmas, extra, callback, rng) -> x`. `denoiseFn` is the **guider**: it wraps
  the model evaluation, conditioning selection by area/mask/timestep, ControlNet application, CFG and CFG patches.
- Partial denoise, start/end step, "return with leftover noise", inpainting noise masks (latent blended back each step),
  and differential diffusion (per-pixel denoise strength from a mask) live in the guider/sampler wrapper so every
  sampler supports them.
- Previews: every `n` steps (setting) the current x0 estimate is turned into a small image, by a per-family 3×C linear
  map (fast, rough), by the tiny preview decoder of that family (TAESD and friends, better), or not at all.

### 6.5 Executor (`graph/`)
1. **Prepare.** Flatten subgraphs, drop muted nodes, rewire bypassed nodes (each output takes the first input of the
   same type; error if none), fold reroutes.
2. **Validate.** Every required input linked or valued; link types match; numbers in range (after rounding to `step`);
   combo values exist (a missing model is a specific error that the editor turns into a "download missing models"
   dialog); custom `validate` hooks. Errors are collected for all nodes, not just the first.
3. **Plan.** Ancestors of the requested outputs, ordered topologically; cycles rejected.
4. **Fingerprint and cache.** A node's cache key = type + widget values + upstream keys + its `fingerprint()`. If the key
   is in the `OutputCache`, the node is skipped and reported as cached. Policies: `last` (keep the outputs of the previous
   job, default), `lru:N`, `ram:<GB>` (evict when free RAM drops under the threshold), `none`.
5. **Run.** In order; lazy inputs are scheduled only when the node asks for them (the node returns "need input X" and is
   re-run once X exists); list handling maps a node over list inputs unless it accepts lists; a node that expands returns
   a sub-graph that is validated and spliced in with ids derived from the parent id (so its cache keys are stable).
6. **Report.** Events per node (started, progress, preview, output summary, cached, error with the input that caused it);
   interrupt checked between nodes and at every sampler step.

### 6.6 Queue and history
- `JobQueue`: job id, number, priority (front/back), client id, graph, status. Operations: add, add-front, cancel
  pending, interrupt running, clear pending, read.
- `History`: status, timings per node, output file list, error, the job graph and editor workflow; kept in the gallery
  SQLite database; size limit (setting `history.maxItems`, default 10 000) with pruning of oldest entries (files stay).
- Seed widgets with *control after run* (`fixed`, `increment`, `decrement`, `randomize`) are applied by the **client**
  after queueing, so the queued graph is exact and history is reproducible. Setting: apply before or after queueing.

---

## 7. Server API

Under the main plan's `/api/v1`, same auth and LAN rules.

| Route | Purpose |
|---|---|
| `POST /api/v1/graph/jobs` | queue a job (`job`, optional `workflow`, `front`, `client.id`, `count`); returns `{id, number}` or validation errors per node and input |
| `GET /api/v1/graph/jobs` | running + pending; `DELETE /api/v1/graph/jobs/{id}` cancels; `DELETE /api/v1/graph/jobs` clears pending |
| `POST /api/v1/graph/interrupt` | stop the running job (optionally only if its id matches) |
| `GET /api/v1/graph/history[?max=&offset=]`, `GET .../history/{id}`, `DELETE ...` | history |
| `GET /api/v1/graph/nodes`, `GET .../nodes/{type}` | node catalogue: inputs, types, ranges, defaults, options, tooltips (localised by `Accept-Language`), category, description |
| `GET /api/v1/graph/models/{folder}`, `GET .../embeddings` | lists for combos; `POST .../models/download` fetches a missing model by URL or catalog key |
| `POST /api/v1/files/upload` | multipart image/mask/video/audio into `input/` (optional subfolder, overwrite flag; mask upload merges into the alpha of an existing image) |
| `GET /api/v1/files/view?name=&folder=&kind=output|input|temp[&preview=webp;80]` | file download / preview |
| `GET /api/v1/graph/system` | versions, devices, VRAM/RAM total and free, memory mode, runtime |
| `POST /api/v1/graph/free` | unload models and/or clear caches |
| `GET/PUT /api/v1/userdata/{path}` | user workflows, settings, templates (per user in multi-user mode) |
| `GET /api/v1/graph/templates` | built-in and plugin workflow templates |
| `WS /api/v1/graph/events?client=` | event stream (below) |

WebSocket events (JSON text frames, our names): `queue` (length), `job.started`, `job.cached` (node ids),
`node.started`, `node.progress` (`value`, `max`, node id), `node.output` (UI summary such as image file names),
`job.error` (node, type, message, input, traceback-free details), `job.interrupted`, `job.finished`. Binary frames carry
preview images: 4-byte type, 4-byte format, then the encoded image; a preview carries the node id.

---

## 8. Desktop editor (Qt 6 Widgets)

A new **Workflows** tab in the main window, built on `QGraphicsView`/`QGraphicsScene` (own implementation; no third-party
node-editor library, so the same code builds for WebAssembly).

- **Canvas:** pan (drag empty space or middle mouse), zoom (wheel, pinch, keyboard), zoom-to-fit, minimap, grid with
  snapping, low-detail rendering when zoomed out, viewport culling for 1 000+ node graphs.
- **Nodes:** title bar with colour, collapse, pin, mode (active/muted/bypassed shown as dimmed/striped), id badge
  (setting), progress bar, error badge; inputs, outputs and widgets laid out from `NodeDef`; resize; image/video/audio
  preview area; right-click menu (rename, colours, convert widget ↔ input, bypass, mute, pin, clone, remove, open
  subgraph, properties).
- **Widgets:** number fields (drag to change, arrows, typed expressions such as `512*2`), sliders, combos with search and
  per-option tooltips, model combos with folder tree and preview thumbnails, multiline text with prompt syntax
  highlighting (weights, embeddings, comments, choices), toggles, colour picker, image upload with paste and drag-drop,
  seed widget with *control after run*.
- **Tooltips:** hover any widget, input socket, output socket, node title or combo option to see its tooltip
  (description + example). Settings `editor.tooltips` (on/off) and `editor.tooltipDelayMs`. Tooltips are translatable.
- **Links:** drag from socket to socket; type-checked with colour; drop on empty canvas opens search filtered to nodes
  with a compatible input (or output); shift-drag moves all links of a socket; link styles spline/linear/straight/
  hidden; reroute dots by double-clicking a link (alt-click on a link inserts a node).
- **Groups:** create from selection, title, colour, fit to contents, move contents with the group, nested groups.
- **Subgraphs:** convert selection to subgraph (boundary links become subgraph inputs/outputs), open/close, edit inputs,
  promote widgets to the subgraph node, save to the node library as a reusable user node, unpack.
- **Search:** double-click canvas: fuzzy search over display names, ids, categories, aliases, descriptions; recent and
  favourite nodes; shows the node's description and inputs as a preview.
- **Sidebars:** node library (categories tree, favourites, user subgraphs), model library (by folder, previews,
  drag a model onto canvas to create its loader node), workflows (saved, open, recent, bookmarks), templates (with
  thumbnails; a template is a workflow with required models listed), queue + history/assets (outputs grid, reopen job
  graph, copy seed, delete), and the existing Logs.
- **Editing:** multi-select (box, ctrl-click), align/distribute, copy/paste (plain keeps values; ctrl+shift+V keeps input
  links), duplicate, undo/redo for every edit (command pattern over the workflow model), workflow tabs with unsaved
  markers, auto-save of open tabs, save/save as/export job format/import.
- **Running:** Queue button with batch count, "queue front", auto-queue mode (off / after each run / when changed),
  "run selected outputs only", interrupt, clear queue; executing node outlined; cached nodes marked; errors focus the
  first bad node and list all errors in a panel.
- **Mask editor:** opened from an image widget: brush (size, hardness, opacity), eraser, fill, invert, clear, zoom,
  paint on alpha; saves as an uploaded mask. Also a crop tool and an A/B image compare view.
- **Missing pieces:** loading a workflow with unknown node types lists the plugins that provide them (from the registry);
  missing models are listed with download buttons when the model is in the catalog or the workflow carries a URL.
- **Settings dialog:** every editor setting listed in the reference document, searchable, each with its tooltip.
- **Keyboard shortcuts:** editable keymap (file `keymap.json`); defaults: Ctrl+Enter queue, Ctrl+Shift+Enter queue front,
  Ctrl+Alt+Enter interrupt, Ctrl+S save, Ctrl+O open, Ctrl+Z/Ctrl+Y undo/redo, Ctrl+B bypass, Ctrl+M mute, Ctrl+G group,
  Ctrl+Shift+E convert to subgraph, Space+drag pan, `.` fit selection, Del delete, Alt+C collapse, Q queue sidebar,
  H history, N node library, M model library.

---

## 9. Browser editor (C++ compiled to WebAssembly)

- The editor sources (`ui/graph/`) are written against an `EditorBackend` interface: the desktop build calls services in
  process; the browser build talks to `/api/v1` over HTTP and WebSocket (`QNetworkAccessManager`, `QWebSocket`).
- `GraphicsHost.Web` builds those sources with **Qt for WebAssembly** (Emscripten). The output (`.wasm`, Qt's generated
  loader) is embedded in the exe as resources and served at `/editor/`. This replaces the hand-written JS browser UI of the
  main plan (§14).
- Limits in the browser build: no native plugins' custom widgets (plugins can still add nodes; their inputs use the
  standard widgets), file access through upload/download, clipboard through the browser permission prompt.
- Risk: building Qt for WebAssembly from MSBuild. W9.1 is a spike; the fallback is a separate Emscripten build invoked
  from a MSBuild custom step (still C++ only).

---

## 10. Plugins

- **ABI:** `plugins/sdk/gh_node_plugin.h`, plain C like the runtime ABI. A plugin DLL exports
  `gh_plugin_info()` and `gh_plugin_register(gh_registry*)`. Through the registry it declares nodes (same features as
  built-ins: types, widgets, tooltips, lazy, lists, expansion, fingerprint), custom value types, model folder types,
  server routes, settings, and (desktop only) custom widget factories. A header-only C++ helper gives the same builder
  syntax as built-in nodes.
- **Tooltip rule applies to plugins:** the loader warns in the log and marks the plugin "incomplete" in the manager if a
  parameter has no tooltip.
- **Isolation:** plugins are native code; the manager shows that clearly. `plugins.safeMode` starts with no plugins.
  A plugin that crashes during load is disabled for the next start automatically.
- **Registry and manager:** a JSON index (URL is a setting; default our own) lists plugin id, versions, ABI version,
  download URL, SHA-256, signature, node types provided. The manager installs, updates, pins versions, disables, removes,
  and resolves "which plugin provides missing node type X". Downloads use the existing `DownloadManager`.
- **Workflow templates** can ship in plugins (`templates/*.ghwf.json` with thumbnails).

---

## 11. Settings

New keys in `settings.json` (main plan §6 conventions; each with a validator and a tooltip in the reference document):
`graph.cache.policy`, `graph.cache.size`, `graph.preview.method`, `graph.preview.everyNSteps`, `graph.preview.maxSize`,
`graph.rng`, `graph.historyMaxItems`, `memory.mode`, `memory.reserveVramMb`, `memory.asyncOffload`,
`precision.diffusion`, `precision.textEncoder`, `precision.vae`, `precision.vaeFallbackFp32`, `paths.input`,
`paths.output`, `paths.temp`, `paths.modelPathsFile`, `output.embedWorkflow`, `output.filenameCounterDigits`,
`plugins.enabled`, `plugins.safeMode`, `plugins.registryUrl`, `users.multiUser`, and editor settings under `editor.*`
(stored per user). The full list with tooltips is §S of the reference document.

---

## 12. Testing

- `[unit]`: tensor ops vs reference values; tokenisers vs reference token ids (fixture vocabularies); prompt parser
  (weights, nesting, escapes, embeddings, choices, comments); each sigma schedule vs closed-form values; each sampler on a
  toy analytic denoiser (Gaussian data, exact x0 known) for convergence order; validator errors; executor ordering,
  caching, lazy, lists, expansion, bypass/mute; workflow ↔ job conversion; file format migrations; **tooltip coverage
  test** (W1.6).
- `[golden]`: fixed seeds on tiny fixture models (SD1-style and DiT-style, a few MB, generated by us) must reproduce
  stored latents within tolerance on CPU; regenerate only on deliberate changes.
- `[integration]`: real models (skipped when absent): SD1.5 and SDXL text-to-image through the graph must match ABI v1
  `gh_generate` output for the same seed/sampler on CPU; LoRA on/off; ControlNet; inpaint.
- `[api]`: server on an ephemeral port: queue, events order, interrupt, uploads, history, node catalogue schema.
- Editor: model-level tests for undo/redo, copy/paste, subgraph convert/unpack, file round trip; screenshot smoke tests
  with `QTest`.

---

## 13. Phases

Prerequisites from the main plan: phases 1–3 (runtime DLLs, `GenerationRequest`, job queue, model library) and 4.1–4.2
(HTTP server, auth). W phases start after main-plan phase 3; W7 needs main-plan 4.1–4.2; W8 needs 4.6 (Qt shell).

### W0 — Spike: the pipeline in pieces (L)
- [ ] W0.1 ABI v2 skeleton in `gh_components.h`; SD1.5 text encoder, UNet and VAE exposed as separate components (M)
- [ ] W0.2 Host-side Euler sampler over `gh_denoise`; same seed and settings as `gh_generate` give the same image on CPU (M)
- [ ] W0.3 Measure per-step overhead of the host loop vs the in-runtime loop (SD1.5 512², SDXL 1024²); budget < 3 % (S)
- [ ] W0.4 SDXL through the same path (dual encoders, pooled + size conditioning) (S)
- [ ] W0.5 Naming check in CI: a script fails on the reference product's name in any tracked file (the pattern is stored as character codes so the script itself does not contain the word) (S)
- [ ] W0.6 **Go/no-go**: host-side sampling confirmed, or fall back to runtime-side samplers with a sampler-id ABI (S)

**Exit:** a C++ test builds an image from three ABI v2 calls plus a host sampler, identical to ABI v1.

### W1 — Graph model (M–L)
- [ ] W1.1 `tensor/` library and image/mask codecs (M)
- [ ] W1.2 Types, `NodeDef` builder, registry, node catalogue JSON (M)
- [ ] W1.3 Workflow (editor) format and job format, conversion (subgraphs, bypass, mute, reroutes, constants), migrations (M)
- [ ] W1.4 Validator with per-node, per-input errors (S)
- [ ] W1.5 Tooltip infrastructure: `tips::` constants, per-option tips, `QT_TR_NOOP` extraction, catalogue export (S)
- [ ] W1.6 Coverage test: every input has a tooltip; every widget, every option tooltip and every setting has one containing an example marker (`Example:` or `e.g.`) (S)
- [ ] W1.7 Docs generator: renders the reference tables from the registry and diffs them with `docs/workflow-node-reference.md` (S)

### W2 — Executor, queue, history (L)
- [ ] W2.1 Planner (ancestors of outputs, partial run), topological order, cycle check (S)
- [ ] W2.2 Fingerprints, `OutputCache` with the four policies (M)
- [ ] W2.3 Lazy inputs, list mapping, node expansion with stable ids (M)
- [ ] W2.4 Events bus; per-node progress; interrupt between nodes and per step (S)
- [ ] W2.5 `JobQueue` (front/back, cancel, clear), `History` in SQLite (M)
- [ ] W2.6 Utility nodes: constants, reroute, note, string tools, compare/switch (lazy), preview-any (S)

### W3 — Models, memory, patches (L)
- [ ] W3.1 `ModelFolders` (all types, extra roots file), `ModelIndex` with previews and hashes (M)
- [ ] W3.2 Loader nodes: checkpoint, diffusion model, text encoders (1–4), VAE, LoRA (+ model-only), embeddings listing (M)
- [ ] W3.3 `ComponentCache`, `MemoryManager` with modes, reserve, LRU eviction, partial residency (L)
- [ ] W3.4 `WeightPatcher`: LoRA/LoCon/LoHa/LoKr/DoRA/full diff key mapping for SD1/SDXL; lazy apply and undo (L)
- [ ] W3.5 Precision options and fp8 storage (M)
- [ ] W3.6 "Free memory" / "unload models" actions (S)

### W4 — Text and sampling (XL)
- [ ] W4.1 CLIP BPE tokeniser, prompt parser (weights, embeddings, choices, comments, BREAK) (M)
- [ ] W4.2 Text encode nodes (generic, SDXL, SDXL refiner), clip-skip node (S)
- [ ] W4.3 RNGs (cpu, philox), noise nodes, Brownian-tree SDE noise (M)
- [ ] W4.4 Model sampling types and their nodes (discrete, continuous EDM/V, flow shift) (M)
- [ ] W4.5 All schedulers and sigma nodes (M)
- [ ] W4.6 All samplers incl. CFG++ variants, toy-denoiser convergence tests (XL)
- [ ] W4.7 Guiders: basic, CFG, dual CFG, perpendicular negative; rescale CFG, CFG-Zero*, APG (M)
- [ ] W4.8 Sampler and Sampler (advanced) nodes, custom-sampling nodes, inpaint noise mask, start/end steps (M)
- [ ] W4.9 Previews: linear latent-to-RGB per family, tiny preview decoders (S)
- [ ] W4.10 Generate tab and `/v1/images/generations` re-implemented as a built-in workflow template (S)

**Exit:** the default text-to-image workflow runs from a job file on CPU and Vulkan; every sampler × scheduler pair
produces an image on SD1.5; golden tests pass.

### W5 — Conditioning and control (L)
- [ ] W5.1 Combine, concat, average, area (px/%), strength, mask, timestep range, zero-out; guider honours all of them (M)
- [ ] W5.2 ControlNet, T2I-Adapter, control-LoRA, union types, diff ControlNet; apply node with strength/start/end and optional VAE (L)
- [ ] W5.3 CLIP-vision encode, style model apply, unCLIP checkpoint + conditioning (M)
- [ ] W5.4 GLIGEN loader + text box (S)
- [ ] W5.5 Inpaint conditioning, inpaint VAE encode, instruct-pix2pix, differential diffusion (M)
- [ ] W5.6 Hooks: LoRA hooks on text encoder and model, hook keyframes, conditioning properties (M)
- [ ] W5.7 PhotoMaker loader + encode (S)

### W6 — Latent, image, mask, I/O nodes (L)
- [ ] W6.1 All latent nodes in the reference document (M)
- [ ] W6.2 All image nodes incl. model upscaler with tiling (M)
- [ ] W6.3 All mask nodes (S)
- [ ] W6.4 Load image/mask/latent, save/preview image (PNG/JPEG/WebP + metadata), save latent, filename tokens and counter (M)
- [ ] W6.5 Tiled VAE encode/decode (spatial + temporal) (S)

### W7 — Server API and events (M)
- [ ] W7.1 Routes of §7, validation errors as structured JSON (M)
- [ ] W7.2 WebSocket events and binary previews; per-client targeting (M)
- [ ] W7.3 Uploads, views (with preview re-encoding), user data (S)
- [ ] W7.4 API contract tests (S)

### W8 — Desktop editor (XL)
- [ ] W8.1 Workflow document model with commands (undo/redo) independent of Qt widgets (M)
- [ ] W8.2 Canvas: pan/zoom, grid, snapping, minimap, culling, LOD (L)
- [ ] W8.3 Node items generated from `NodeDef`; all widget kinds; tooltips everywhere (L)
- [ ] W8.4 Links, type colours, link styles, drop-to-search, reroutes, shift-drag relinking (M)
- [ ] W8.5 Groups, notes, colours, pin, collapse, bypass/mute, id badges (M)
- [ ] W8.6 Node search popup and node library sidebar (M)
- [ ] W8.7 Queue controls, batch count, auto-queue, partial run, interrupt; execution highlight, progress, previews, errors panel (M)
- [ ] W8.8 Queue/history (assets) sidebar, reopen job graph (M)
- [ ] W8.9 Model library sidebar with previews, drag-to-create loaders (S)
- [ ] W8.10 Workflows sidebar, tabs, auto-save, open/save/export/import (M)
- [ ] W8.11 Copy/paste (with and without links), clone, align/distribute, selection toolbox (S)
- [ ] W8.12 Subgraphs: convert, edit, promote widgets, save as user node, unpack (L)
- [ ] W8.13 Templates browser with built-in templates (§13a) (S)
- [ ] W8.14 Mask editor, crop tool, compare view (M)
- [ ] W8.15 Settings dialog, keymap editor (M)
- [ ] W8.16 Missing nodes / missing models dialogs with download (S)
- [ ] W8.17 Drag-and-drop: workflow files, images/video/audio with embedded workflows, images onto Load Image nodes (S)

**Exit:** a user can build, run, save and reopen the default, img2img, inpaint, hi-res-fix, ControlNet and LoRA templates
without touching a file; every parameter shows its tooltip.

### W9 — Browser editor (L)
- [ ] W9.1 Spike: Qt for WebAssembly build of the canvas from MSBuild (M)
- [ ] W9.2 `EditorBackend` over HTTP/WebSocket; uploads, views (M)
- [ ] W9.3 Embed and serve at `/editor/`; tokens; LAN smoke test (S)

### W10 — Model families and patches (XL, split per family)
Each family = detection, ggml architecture (reuse from stable-diffusion.cpp when present, else new in `runtime/arch/`),
text encoders and tokeniser, VAE, latent format and preview factors, model-sampling defaults, family-specific nodes,
LoRA key mapping, a template, a golden test.
- [ ] W10.1 SD 2.x (v-prediction, 768) (S)
- [ ] W10.2 SD 3/3.5 (MMDiT, three encoders, empty-padding option, 16-ch latent) (M)
- [ ] W10.3 FLUX.1 dev/schnell, guidance node, Fill, Kontext reference latents and image scale, Redux style, depth/canny (L)
- [ ] W10.4 Chroma (S) · W10.5 Stable Cascade (stages B/C, two latents) (M) · W10.6 AuraFlow (S) · W10.7 PixArt α/Σ (M)
- [ ] W10.8 Hunyuan-DiT (bert + mT5) (M) · W10.9 Kolors (S) · W10.10 Lumina 2 (M) · W10.11 HiDream (four encoders) (L)
- [ ] W10.12 Qwen-Image and Qwen-Image-Edit (LLM encoder) (L) · W10.13 OmniGen 2 (L) · W10.14 Cosmos-Predict2 (M)
- [ ] W10.15 Model patches: FreeU v1/v2, HyperTile, PAG, SAG, token merging, deep shrink, SLG, attention multiply, epsilon scaling (L)
- [ ] W10.16 Model merge nodes (simple, blocks, add, subtract; text encoder merges), save checkpoint/model/text encoder/VAE, LoRA extraction (M)

### W11 — Video, audio, 3D (XL)
- [ ] W11.1 Video type, load video, save animated WebP/PNG, save video (MP4/WebM via a C++ encoder library; §15 risk) (M)
- [ ] W11.2 SVD image-to-video + video CFG guiders (M)
- [ ] W11.3 Wan 2.x (t2v, i2v, first/last frame) (L) · W11.4 LTX-Video (M) · W11.5 Hunyuan Video (L) · W11.6 Mochi (M) · W11.7 Cosmos video (M)
- [ ] W11.8 Audio type, load/save/preview audio; Stable Audio Open; ACE-Step (L)
- [ ] W11.9 Hunyuan3D 2: latent, decode to voxels, voxel to mesh, save GLB, 3D preview widget (L)
- [ ] W11.10 Webcam capture node (S)

### W12 — Plugins, users, training (L)
- [ ] W12.1 Plugin ABI, loader, safe mode, example plugin (M)
- [ ] W12.2 Registry client and plugin manager UI (M)
- [ ] W12.3 Multi-user mode (S)
- [ ] W12.4 LoRA training nodes *(stretch)* (L)

### W13 — Polish (M)
- [ ] W13.1 Translations of all tooltips (start with English; `.ts` files) (S)
- [ ] W13.2 Accessibility: keyboard-only editing, focus order, screen-reader names for nodes and widgets (M)
- [ ] W13.3 Performance pass: 2 000-node graph at 60 fps, executor overhead < 1 ms per cached node (S)
- [ ] W13.4 User docs generated from the registry (node pages with tooltips and examples) (S)

### 13a. Built-in templates
Text-to-image (SD1.5, SDXL, SDXL + refiner), image-to-image, inpaint, outpaint, hi-res fix (latent and model upscale),
LoRA stack, ControlNet (canny, depth, pose), area composition, unCLIP, SDXL Turbo one-step, then per family as W10/W11
land (SD3.5, FLUX dev/schnell/Fill/Kontext, Qwen-Image edit, Wan text/image-to-video, audio, 3D).

### Ordering

`W0 → W1 → W2 → W3 → W4` is sequential. After W4: W5, W6 and W7 in parallel; W8 needs W7's events contract (it can start
on the document model, W8.1–W8.3, after W2). W9 after W8.3. W10/W11 items are independent of each other once W5–W6 exist.
W12 after W8.

### Rough timeline (one engineer)

| Phase | Effort | Cumulative |
|---|---|---|
| W0 | 2 wk | 2 wk |
| W1–W2 | 4 wk | 6 wk |
| W3 | 2–3 wk | 8–9 wk |
| W4 | 4–5 wk | 12–14 wk |
| W5–W7 | 5–6 wk | 17–20 wk |
| W8 | 7–8 wk | 24–28 wk |
| W9 | 2–3 wk | 26–31 wk |
| W10 | 10–14 wk | 36–45 wk |
| W11 | 8–10 wk | 44–55 wk |
| W12–W13 | 4 wk | 48–59 wk |

---

## 14. Changes to the main plan

| Main plan | Change |
|---|---|
| §5 standard parameter set, Generate tab | kept as the simple UI, implemented as a fixed workflow (W4.10) |
| §7 browser UI in HTML/CSS/JS | replaced by the WebAssembly build of the C++ editor (W9); the Generate/Gallery/Models/Settings pages become editor panels |
| Phase 5.4 LoRA, 5.7 upscaler/ControlNet | delivered by W3.4, W6.2, W5.2 |
| Model families "v1: SD1/SDXL only" | still true for the first release; W10 adds families afterwards |
| §11 vcpkg packages | add: Qt WebSockets (`qtwebsockets`), a video codec library (§15), `sentencepiece`-compatible code written by us (no package), Emscripten toolchain for W9 |

---

## 15. Risks

- **ABI v2 depends on stable-diffusion.cpp internals.** Its model classes are not a public API; pin and wrap only in
  `runtime/`, as for ABI v1. Upgrades are deliberate and covered by golden tests.
- **New architectures in ggml are large work** (W10/W11), and the Vulkan backend may lack ops some models need; each
  family has a CPU-only fallback until ops are added.
- **Video encoding in C++** needs a codec library: libvpx (WebM/VP9, BSD) and/or OpenH264 (BSD, Cisco binary licence) via
  vcpkg; MP4 muxing written by us or via a small muxer library. Decide in W11.1.
- **Qt for WebAssembly from MSBuild** (W9.1 spike, fallback described in §9).
- **Memory estimation** for partial residency is approximate; measure per family and keep a safety margin setting.
- **Native plugins can crash the process.** Safe mode, auto-disable after a crash on load, and (later) an out-of-process
  plugin host using the same ABI over IPC.
- **Clean room discipline** must hold for every contributor; record in each PR template that no reference-product source
  was consulted.
