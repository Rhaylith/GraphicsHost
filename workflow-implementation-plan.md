# GraphicsHost Workflows: implementation plan (C++)

This plan adds a **node-graph workflow system** to GraphicsHost. A user builds a generation pipeline from small nodes (load
a checkpoint, encode a prompt, sample, decode, upscale, save) in a visual editor, wires them together, and queues the
graph. A local engine runs only the parts that changed, keeps loaded models in memory, and streams progress and previews
back. The same graph can be saved as a file, embedded in the images it produced, and sent to an HTTP API.

It builds on [implementation-plan.md](implementation-plan.md). Where the two disagree, this plan wins for the areas it
covers (§14 lists the overrides).

**Starting point (2026-10-25).** The main plan is done except the stretch items (5.7 upscaler/ControlNet, 6.6
auto-update), an MSI installer, and testing on other hardware (6.7). What already exists, and is reused rather than rebuilt:

| Exists | Where | Used by this plan for |
|---|---|---|
| Two runtimes, `sd-cpu-avx2.dll` and `sd-vulkan.dll` (the non-AVX2 one was dropped), ABI v1 with `struct_size` appends and `GH_FEATURE_*` bits (img2img, inpaint, LoRA) | `runtime/gh_runtime.h`, `gh_runtime.cpp`; `ggml/ggml.vcxproj` + `ggml-vulkan.vcxproj` | the component functions of §5 are appended the same way, in `sd-vulkan.dll` only; `sd-cpu-avx2.dll` is left as it is for the existing Generate path and gets no workflow features |
| SD 1.5 and SDXL verified on the RX 7900 XTX (SD1.5 512² 6.9 s, SDXL 1024² 25.6 s), VRAM calibration | `docs/phase0-results.md`, `docs/vram-measurements.md`, `engine/FitEstimator` | golden references (W0), memory estimates (W3) |
| `IImageEngine` / `ILoadedModel`, `JobQueue` (one worker, observer, cancel), `ModelManager` (one model, idle TTL), `GenerationRequest`/`LoadOptions` with validator and JSON, `knownSamplers()`/`knownSchedulers()` | `GraphicsHost.Core/engine/` | the queue is generalised (W2.5); `ModelManager` gives way to the component cache (W3.3) |
| `GenerationService`, `AppHost`, `HardwareMonitor` | `GraphicsHost.Core/host/` | graph jobs go through the same service layer |
| Gallery (dated folders, sidecar JSON, SQLite index), image codecs with embedded parameters | `gallery/GalleryStore`, `engine/ImageCodec` | outputs and history (W2.5, W6.4) |
| Model library, model cards, catalog, downloads, Hugging Face client, fit hints, update check, recommendation | `GraphicsHost.Core/models/` (**not in git, see W0.0**) | model folders and the missing-models flow (W3.1, W8.16) |
| HTTP server with JSON errors, `SseStream`, tokens with `read`/`generate`/`manage`/`admin`, HTTPS, allow-list, rate limit, request log, metrics | `GraphicsHost.Core/server/`, `docs/api.md` | the graph routes and events (§7) |
| Qt tabs Generate, Gallery, My Models, Discover, Settings, Monitor, Logs; `MaskEditor`, `ImageView`, themes with accent, interface modes, translation scaffolding (`graphicshost_de.ts`), first-run wizard, tray | `GraphicsHost/ui/` | the Workflows tab (§8) reuses the mask editor, image view, theme and modes |
| Browser UI in plain HTML/JS (`web/`, about 700 lines) | `web/` | kept as it is until W9 (§9) |
| Presets and per-model defaults, safety word filter (off by default), CLI verbs, portable package (`installer/Package.proj`) | `presets/`, `engine/SafetyFilter`, `cli/`, `installer/` | templates (W8.13), safety on graph prompts (W4.2), `graph` CLI verbs (W7.5), packaging (W13.5) |
| Test tags `[unit]`, `[integration]`, `[model]`, `[api]`, `[ui]`, `[network]` (about 230 cases without models or network) | `GraphicsHost.Tests/` | the same tags (§12) |

So the W phases can start now; nothing in the main plan blocks them.

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
6. **Vulkan only, AVX2 always.** All model computation runs on Vulkan (AMD in v1) through `sd-vulkan.dll`; there is no
   CPU inference path, no CPU fallback and no "CPU only" mode in the workflow system. Low VRAM is handled by keeping
   weights in RAM and streaming them to the GPU (`offload_params_to_cpu`, still Vulkan compute) and by partial
   residency, never by moving a component to the CPU. Host-side C++ (tensor maths, samplers, image nodes, codecs) is
   built with `/arch:AVX2` and assumes AVX2 everywhere: no scalar fallback, no runtime CPU dispatch. A machine without a
   usable Vulkan device gets a clear "a Vulkan GPU is required for workflows" message.

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
| **Memory** | Load on demand, keep in VRAM while useful, offload to RAM, evict by LRU under memory pressure; partial loading for low VRAM; memory modes (*GPU only, high VRAM, normal, low VRAM, streaming*); reserved VRAM; precision per component (fp32, fp16, bf16, fp8 storage, ggml quantisations); "free memory" and "unload models" actions | W3 |
| **Weights and patches** | LoRA and LyCORIS variants (LoCon, LoHa, LoKr, DoRA, full diff) on model and text encoder with separate strengths; patches applied lazily and undone without reloading; scheduled LoRA strength over the sampling run (hooks and keyframes); model merging (simple, per block, add, subtract); save merged checkpoints and extract LoRAs | W3, W10 |
| **Text** | Tokenisers (CLIP BPE, T5/mT5 SentencePiece, LLM BPE for newer text encoders); prompt weighting `(word:1.2)`, nesting `((word))`, escaping `\(`; textual-inversion embeddings `embedding:name`; per-encoder prompts (SDXL `g`/`l`, SD3 `l`/`g`/`t5`); clip-skip; dynamic prompt choices `{red|green|blue}`; `//` and `/* */` comments in prompts | W4 |
| **Sampling** | One-node sampler and the split pipeline (noise source + guider + sampler + sigma schedule); ~35 samplers and 9 schedulers (§6.4); custom sigma tools (split, flip, set first, extend); model sampling types (eps, v-prediction, x0, LCM, flow / rectified flow with shift, EDM, zero-terminal-SNR); CFG variants (rescale, dual CFG, perpendicular negative, CFG++, APG, CFG-Zero*); reproducible seeds with a CPU-compatible RNG and a Philox RNG, both run on the host | W4, W10 |
| **Conditioning** | Combine, concat, average; area (pixels or percent) and strength; mask conditioning; timestep ranges; zero-out; SDXL size/crop conditioning; ControlNet, T2I-Adapter, control-LoRA, union ControlNet types, diff ControlNet, with strength and start/end; style models; CLIP-vision encode; unCLIP; GLIGEN boxes; inpaint conditioning; instruct-pix2pix; reference latents for edit models; PhotoMaker | W5 |
| **Latents, images, masks** | Empty latents per family; VAE encode/decode, tiled and temporal-tiled; inpaint encode with mask growth; latent upscale (several methods), composite, blend, arithmetic, batch tools, crop, flip, rotate, noise masks; image scale/crop/pad/blend/blur/sharpen/quantise/invert/stitch/morphology; masks from images and colours, mask maths, grow, feather, threshold; model upscalers (ESRGAN family) with tiling | W6 |
| **Inputs and outputs** | Load image (with alpha as mask), load mask, load latent, load video, load audio, webcam capture; save/preview image (PNG with embedded workflow, JPEG, WebP), animated WebP/PNG, video, audio, latents; filename prefixes with date and node-value tokens | W6, W11 |
| **Model families** | SD 1.x, SD 2.x (incl. 768 v-prediction), SDXL base/refiner/Turbo/Lightning, SD 3/3.5, FLUX.1 (dev, schnell, Fill, Kontext, Redux, depth/canny), Chroma, Stable Cascade, AuraFlow, PixArt α/Σ, Hunyuan-DiT, Kolors, Lumina 2, HiDream, Qwen-Image (incl. edit), OmniGen 2, Cosmos-Predict2, SDXL edit/instruct models | W4 (SD1/SDXL), W10 |
| **Model patches** | FreeU v1/v2, HyperTile, perturbed-attention guidance, self-attention guidance, token merging, deep shrink, differential diffusion, skip-layer guidance, attention multiply, epsilon scaling, latent/ CFG post-processing | W10 |
| **Video, audio, 3D** | Image-to-video (SVD), text/image-to-video (Wan 2.x, LTX-Video, Hunyuan Video, Mochi, Cosmos); video CFG schedules; audio generation (Stable Audio Open, ACE-Step) with audio VAE and save as FLAC/MP3/Opus; image-to-3D (Hunyuan3D 2) to voxels and mesh, save GLB, 3D preview | W11 |
| **Editor** | Infinite canvas, pan/zoom, minimap, grid snapping, alignment tools; node search (double-click canvas, or drop a link on empty canvas for a type-filtered search); node library sidebar; model library sidebar with previews; workflow browser; templates browser; queue and history (assets) sidebar; image previews in nodes; per-node progress and execution highlight; error badges on nodes and inputs; link styles; selection toolbox; copy/paste with or without links; undo/redo; workflow tabs; mask editor; image crop and compare views; settings dialog; keyboard shortcut editor; light/dark themes; translations | W8 |
| **Browser editor** | The same editor in a browser (WebAssembly), talking to the local server; works over LAN with tokens | W9 |
| **API** | Queue a job, read queue, cancel, interrupt, history, download outputs, upload inputs, node catalogue (all node definitions with types, ranges, defaults, tooltips), model lists, embeddings list, system stats, free memory; Server-Sent Events stream with status, start, cached nodes, executing node, step progress, previews, node outputs, errors, interruption, completion; `graph` CLI verbs | W7 |
| **Extensibility** | Native plugin DLLs with a C ABI for new nodes, new widget kinds (desktop), new model formats and new server routes; plugin registry and in-app plugin manager (install, update, disable, remove, safe mode); per-plugin settings | W12 |
| **Multi-user** | Optional user accounts on a shared server: own settings, workflows, inputs/outputs subfolders | W12 |
| **Training** | Train a LoRA from a folder of images inside a workflow (stretch) | W12 |

Out of scope: cloud "partner API" nodes that call paid remote services (this is a local host), and Python of any kind.

---

## 3. Architecture

New folders are marked **new**; everything else exists and is extended.

```
GraphicsHost/
├─ runtime/                       sd-vulkan.dll (workflows); sd-cpu-avx2.dll unchanged for the Generate path
│  ├─ gh_runtime.h                ABI v1 + the component functions appended under GH_FEATURE_COMPONENTS (§5)
│  └─ arch/                       new: own ggml modules for architectures the pinned engine lacks (W10/W11)
├─ GraphicsHost.Core/
│  ├─ engine/     existing; JobQueue generalised to any job kind; ImageCodec gains metadata chunks for workflows
│  ├─ models/     existing library/catalog/downloads; adds ComponentCache, MemoryManager, WeightPatcher
│  ├─ gallery/    existing; adds the history tables and workflow/job columns
│  ├─ host/       existing; adds GraphService next to GenerationService
│  ├─ server/     existing; adds GraphRoutes, FileRoutes (uploads, views), SSE graph events
│  ├─ cli/        existing; adds GraphCommands (`graph run`, `graph nodes`, `graph validate`)
│  ├─ tensor/     new: Tensor (CPU float32/f16, up to 5-D), elementwise ops, resize, blur, morphology
│  ├─ text/       new: tokenisers (CLIP BPE, SentencePiece, LLM BPE), prompt parser (weights, embeddings, choices, comments)
│  ├─ sampling/   new: noise/RNG, sigma schedules, model-sampling types, samplers, guiders, CFG patches
│  ├─ graph/      new: types, NodeDef + registry, Workflow (editor format), JobGraph (job format), Validator,
│  │              Executor, OutputCache, Events
│  ├─ nodes/      new: built-in nodes, one .cpp per category (loaders, conditioning, sampling, latent, image, mask, ...)
│  └─ plugins/    new: plugin loader, registry client, plugin manager
├─ GraphicsHost/ui/               existing tabs; MaskEditor and ImageView reused by the editor
│  └─ graph/                      new: Workflows tab: Canvas, NodeItem, LinkItem, widgets, panels (§8)
├─ GraphicsHost.Web/              new: the same editor sources built with Qt for WebAssembly (§9)
├─ web/                           existing plain HTML/JS browser UI, unchanged until W9
├─ sdk/                           new: gh_node_plugin.h, C++ helper header, example plugin project
└─ GraphicsHost.Tests/            existing tags [unit] [integration] [model] [api] [ui] [network]
```

No new folder may be named `models/`, `logs/`, `build/` or `bin/`: `.gitignore` patterns for data folders must stay
anchored (W0.0).

**Data flow of one job.** Editor (desktop or browser) → `Workflow` → converted to a `JobGraph` (muted nodes removed,
bypassed nodes short-circuited, subgraphs flattened, reroutes and constants folded) → `POST /api/v1/graph/jobs`, the
`graph run` verb, or `GraphService` directly → `Validator` → the shared `JobQueue` → `Executor` on the queue's worker →
nodes call `ComponentCache` / `MemoryManager` / `sampling` / the runtime → events to the SSE stream and to the Qt editor
→ outputs into `GalleryStore` → history.

**Threads.** The existing `JobQueue` worker runs every job, generation or graph, one at a time (the GPU is one resource).
Node code may use a CPU thread pool for image maths. The runtime DLL is called only from that worker (ABI v1 rule:
`gh_generate` is not re-entrant, progress and log callbacks are process-wide). The UI thread never blocks on the engine;
it receives events through `QMetaObject::invokeMethod`, as the Generate tab does today.

**Relationship to the existing single-shot generator.** The Generate tab, the browser UI, `gen`, `POST /api/v1/generate`
and `/v1/images/generations` keep working on ABI v1 unchanged. Once the graph path matches it image for image (W4.10),
`GenerationService` builds a fixed workflow from a `GenerationRequest` instead of calling `gh_generate`, so there is one
engine path and one place where sampling is implemented. `gh_generate` stays in the DLLs as the reference for the
`[model]` parity tests.

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

## 5. Engine: component functions in the runtime ABI

`gh_generate` runs the whole pipeline in one call (with img2img, inpaint and LoRA appended in main-plan phase 5). Node
graphs need the pipeline in pieces: encode text once, run the model once per step under a host-side sampler, decode with
a chosen VAE. The component functions are **appended to `gh_runtime.h`** the way phase 5 extended it: new functions and
types, a new feature bit `GH_FEATURE_COMPONENTS` in `gh_runtime_info.features`, and `GH_ABI_VERSION` stays 1 because
nothing existing changes. The host looks the new functions up with `GetProcAddress`, so an older DLL without them still
loads and simply reports "graph features unavailable".

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
  Wan, VAE, TAE, ESRGAN, ControlNet, PhotoMaker) driven through our own shim next to the existing `gh_runtime.cpp`, plus
  our own ggml modules for architectures the pinned engine version (`fd21ecf`) lacks (W10). New ggml code lives in
  `runtime/arch/` with one file per architecture and the same internal interface. Both DLL projects compile the same
  sources, as today.
- LoRA today is passed per generation (`gh_generate_params.loras`) and applied by the engine. The component path moves
  this to `gh_component_set_patches` so a patched model stays patched across steps and jobs; the per-generation path
  stays for ABI v1 callers.
- `struct_size` rules, `gh_last_error()` and the `gh_status` codes as today; `EngineErrorCode` (incl. `OutOfMemory`,
  `Cancelled`) maps component errors to the CLI exit codes and API error codes that already exist.
- The pinned engine commit is the one Phase 0 verified. Upgrading it (needed for several W10 families) is a deliberate
  step with the `[model]` parity tests and `tools/measure-vram.ps1` rerun.

---

## 6. Core subsystems

### 6.1 Tensor library (`tensor/`)
A small owned tensor type: `float` and `f16` storage, shapes up to 5-D, strided views for batch slicing, and the ops the
nodes need (add, mul, lerp, slerp, clamp, mean/std per channel, interpolate nearest/bilinear/bicubic/area/lanczos/
bislerp, gaussian blur, sharpen, morphology, rotate/flip, pad, crop, concat). Compiled with `/arch:AVX2` (AVX2 and FMA
are assumed, so intrinsics from `<immintrin.h>` are used directly where measured to help, with no fallback path);
parallel loops on the CPU pool. These are small host-side operations on latents and images, not model inference.
Image codecs reuse libpng, libjpeg-turbo, libwebp.

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
- **Model kinds in the existing `ModelLibrary`.** Today the library scans one models folder (`modelsDir`, default
  `DownloadedModels` in the working directory) in the `publisher/repo/file` layout, reads safetensors/GGUF headers
  (`ModelCard`: family, defaults, companions) and links a VAE through `<file>.source.json`. It is extended with a
  **kind** per file (checkpoint, diffusion model, text encoder, VAE, LoRA, ControlNet/adapter, CLIP vision, style model,
  upscaler, embedding, hypernetwork, GLIGEN, PhotoMaker, preview decoder), detected from the header and overridable by
  placing the file under a kind subfolder (`loras/`, `vae/`, ...). Extra roots per kind come from
  `model-paths.json` in the data folder. Combos list files by their path relative to the root, so they stay stable when
  roots are added. Downloads from the catalog and Hugging Face keep working as they are.
- **Previews and hashes:** a preview image next to the file (`name.preview.png`) for the model library sidebar; SHA-256
  from `core/Sha256` (already used for downloads), cached with the card.
- **`ComponentCache`**: an opened component is shared by every node that loads the same file with the same options;
  reference-counted handles in node outputs. It replaces `ModelManager`'s one-checkpoint policy for graph jobs and keeps
  its idle unload (`LoadOptions.ttlMinutes`, default 10) per component. The Generate path keeps using `ModelManager`
  until W4.10, and the two never hold the same file twice: `ModelManager` asks the cache first.
- **`MemoryManager`**: knows the device budget (from `GpuDevices`/`HardwareMonitor`), every component's resident size,
  and an estimate of the activation memory of the next call. The estimate starts from `FitEstimator`'s measured
  constants (SD 1.x ≈ 1450 MiB + 3130 MiB/MP, SDXL ≈ 1100 MiB + 2540 MiB/MP, `docs/vram-measurements.md`) and gets a
  per-component measurement step for every new family (`tools/measure-vram.ps1`). Before each runtime call it makes room:
  evicts least-recently-used components to RAM, then closes them, then chooses partial residency. Memory modes
  (setting `memory.mode`): `gpu-only`, `high`, `normal` (default), `low` (partial residency allowed), `stream` (weights
  stay in RAM and are streamed to the GPU each call). Every mode computes on the GPU. `memory.reserveVramMb` keeps room for the desktop (the measured idle desktop on the dev
  machine is about 1.24 GiB). Weight offload already exists in the runtime (`offload_params_to_cpu`, measured nearly
  free on SDXL) and is the first tool before partial residency; flash attention stays on by default for SD 1.x above
  512² on Vulkan, where it was measured 5× faster.
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
| Noise / RNG | `cpu-compatible` (mt19937 + Box–Muller, matching the CPU generator most tools use, so seeds are comparable; always run on the host), `philox` (counter-based, matches common GPU generators); per-batch-item seeds; noise generator selectable per sampler; SDE noise via Brownian tree | — |

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
- The existing `JobQueue` (one worker thread, `Job` with state/progress/result/cancel via `std::stop_source`, an
  observer, finished jobs trimmed) is generalised: a `Job` holds a `JobWork` (either a `GenerationRequest` on a loaded
  model, as now, or a `JobGraph`) so generation and graph jobs share one queue and one GPU. Added: priority (front/back),
  client id, a queue number, "clear pending", and node-level progress (`StepProgress` gains the node id).
- History: status, timings per node, output list, error, the job graph and editor workflow. Stored in the gallery's
  SQLite database (new tables plus a schema migration; `GalleryStore::rescan` learns to read the workflow chunk). Every
  saved output is a normal gallery entry, so the Gallery tab, `/api/v1/gallery` and "reuse parameters" see graph outputs
  too ("reuse" opens the workflow when the entry has one). Size limit `graph.historyMaxItems` (default 10 000) prunes
  history rows, never files.
- The safety filter applies to graph jobs: when `safety.enabled` (or the job's own flag) is on, every prompt string that
  reaches a text-encode node is checked by `ISafetyFilter::checkPrompt` during validation, and a refusal is reported as
  a validation error on that node (`EngineErrorCode::Blocked`).
- Seed widgets with *control after run* (`fixed`, `increment`, `decrement`, `randomize`) are applied by the **client**
  after queueing, so the queued graph is exact and history is reproducible. Setting: apply before or after queueing.

---

## 7. Server API

Added to the existing native API (`server/`, documented in `docs/api.md`): same error shape
(`{"error": {message, type, code}}`), same tokens and permissions, LAN/HTTPS/allow-list/rate-limit rules, and the
existing 64 MB body cap checked before parsing (uploads get their own cap, `server.maxUploadMb`). New route files
`GraphRoutes.cpp` and `FileRoutes.cpp` register through `RouteContext` like the others.

| Route | Permission | Purpose |
|---|---|---|
| `POST /api/v1/graph/jobs` | generate | queue a job (`job`, optional `workflow`, `front`, `client`, `count`); 202 with `{id, number}`, or 400 `invalid_parameters` with errors per node and input |
| `GET /api/v1/graph/jobs`, `GET .../jobs/{id}` | read | running + pending (the existing `GET jobs` lists them too, with a `kind` field) |
| `POST .../jobs/{id}/cancel`, `POST .../jobs/clear` | generate | cancel one (running or pending), clear pending; same verbs as the existing `jobs/{id}/cancel` |
| `GET /api/v1/graph/history?limit=&offset=`, `GET/DELETE .../history/{id}` | read, manage | history |
| `GET /api/v1/graph/nodes`, `GET .../nodes/{type}` | read | node catalogue: inputs, types, ranges, defaults, options, tooltips (localised by `Accept-Language`), interface mode, category, description |
| `GET /api/v1/graph/models/{kind}`, `GET .../embeddings` | read | lists for combos; missing models are fetched with the existing `POST models/get` (catalog key or repository + file) |
| `POST /api/v1/files/upload` | generate | multipart image/mask/video/audio into the input folder (optional subfolder, overwrite flag; a mask upload merges into the alpha of an existing image) |
| `GET /api/v1/files/view?name=&folder=&kind=input\|temp[&preview=webp;80]` | read | inputs and temp previews; outputs keep using `GET /outputs/<path>` and `gallery/{id}/image\|thumbnail` |
| `POST /api/v1/graph/free` | manage | unload models and/or clear caches (extends `models/unload`) |
| `GET/PUT/DELETE /api/v1/userdata/{path}` | read, manage | user workflows, editor settings, keymap, subgraphs (per user in multi-user mode) |
| `GET /api/v1/graph/templates` | read | built-in and plugin workflow templates |
| `GET /api/v1/graph/events?client=` | read | **Server-Sent Events** stream (below) |

System information stays on the existing `GET status`, `hardware` and `hardware/usage`, with memory mode and resident
components added.

**Events use SSE, not WebSocket.** The server already streams job progress with `SseStream`
(`GET jobs/{id}/events`), cpp-httplib serves SSE without extra dependencies, and every client direction that matters
(queue, cancel, upload) is a normal request. Event names (ours): `queue` (length), `job.started`, `job.cached` (node
ids), `node.started`, `node.progress` (`value`, `max`, node id), `node.preview` (node id + URL of a small WebP in the temp
folder, at most `graph.preview.everyNSteps`), `node.output` (UI summary such as gallery ids), `job.error` (node, type,
message, input), `job.interrupted`, `job.finished`. A client reconnecting with `Last-Event-ID` gets missed events from
a short ring buffer. A WebSocket transport can be added later behind the same event bus if preview latency demands it.

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
- **Mask editor:** the existing `ui/MaskEditor` (built for inpainting in main-plan 5.3) opened from an image widget,
  extended with hardness, opacity, fill, invert and painting on alpha; saves as an uploaded mask. Also a crop tool and
  an A/B image compare view built on `ui/ImageView`.
- **Interface modes:** every widget and setting carries the existing interface-mode tag (`user`, `power`,
  `developer`; setting `interfaceMode`). The Workflows tab itself is `power`; in `user` mode it shows templates with
  only their `user` widgets (prompt, size, seed), so the tab is usable without wiring nodes.
- **Look and feel:** link and type colours come from the existing `ui/Theme` (light/dark, accent colour); all strings go
  through `tr()` into the existing translation files.
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
  process; the browser build talks to `/api/v1` over HTTP and reads the SSE event stream (`QNetworkAccessManager`).
- `GraphicsHost.Web` builds those sources with **Qt for WebAssembly** (Emscripten). The output (`.wasm`, Qt's generated
  loader) is copied next to the exe like `web/` is today (`props/Data.targets`), included by `installer/Package.proj`,
  and served at `/editor/`.
- The existing plain-JS browser UI in `web/` (Generate, Gallery, My Models, Discover, Settings, Logs) predates this plan
  and stays as it is; no new JS is written. Once W9 ships, its pages can be rebuilt as panels of the WebAssembly app and
  `web/` retired; that is a separate decision (§14).
- Limits in the browser build: no native plugins' custom widgets (plugins can still add nodes; their inputs use the
  standard widgets), file access through upload/download, clipboard through the browser permission prompt.
- Risk: building Qt for WebAssembly from MSBuild. W9.1 is a spike; the fallback is a separate Emscripten build invoked
  from a MSBuild custom step (still C++ only).

---

## 10. Plugins

- **ABI:** `sdk/gh_node_plugin.h`, plain C like the runtime ABI, with the same `struct_size` and feature-bit rules. A plugin DLL exports
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

Each new option follows `docs/adding-options.md` (declare, `toJson`/`settingsFromJson`, `kKnownKeys`, `validate()`, CLI
`config set`, Settings tab, tests). Adding keys with defaults needs no migration; the gallery database change (W2.5)
does.

- **Reused, unchanged:** `backend`, `gpuDevice`, `threads`, `modelsDir`, `interfaceMode`, `theme`, `accent`,
  `notifications`, `server.*` (port, LAN, tokens, HTTPS, allow-list, rate limit), `logging.level`, `safety.enabled`.
  Load options `flashAttention`, `offloadParamsToCpu` and `ttlMinutes` stay per load, as loader node widgets and
  per-model defaults. `clipOnCpu` and `vaeOnCpu` (CPU compute) are not offered by workflows. `backend` must resolve to
  Vulkan for graph jobs; with `backend = cpu`, or no usable Vulkan device, the Workflows tab and `/api/v1/graph/jobs`
  report that a Vulkan GPU is required.
- **New, camelCase as the conventions require:** `graph.cachePolicy`, `graph.cacheSize`, `graph.previewMethod`,
  `graph.previewEveryNSteps`, `graph.previewMaxSize`, `graph.rng`, `graph.historyMaxItems`, `graph.seedControlTiming`,
  `memory.mode`, `memory.reserveVramMb`, `memory.asyncOffload`, `precision.diffusion`, `precision.textEncoder`,
  `precision.vae`, `precision.vaeFallbackFp32`, `paths.inputDir`, `paths.tempDir`, `output.embedWorkflow`,
  `output.filenameCounterDigits`, `server.maxUploadMb`, `plugins.enabled`, `plugins.safeMode`, `plugins.registryUrl`,
  `users.multiUser`, `network.downloadsAllowed`. Outputs stay in the gallery's outputs folder.
- **Editor settings** (`editor.*`) are per user, stored under user data, not in `settings.json`.

The full list with tooltips is §S of the reference document.

---

## 12. Testing

- `[unit]`: tensor ops vs reference values; tokenisers vs reference token ids (fixture vocabularies); prompt parser
  (weights, nesting, escapes, embeddings, choices, comments); each sigma schedule vs closed-form values; each sampler on a
  toy analytic denoiser (Gaussian data, exact x0 known) for convergence order; validator errors; executor ordering,
  caching, lazy, lists, expansion, bypass/mute; workflow ↔ job conversion; file format migrations; **tooltip coverage
  test** (W1.6).
The existing tags and fixtures are reused (`TestUtil.h`, `ModelFixture.h`, `SafetensorsFixture.h`, `ApiTestUtil.h`,
`TestHttpServer.h`); the "no model, no network" run (`~[model]~[network]`) must stay fast and green in CI.

- `[integration]`: the component functions against a built runtime DLL with tiny fixture weights (SD1-style and
  DiT-style, a few MB, generated by us with `SafetensorsFixture`): fixed seeds reproduce stored latents within a
  tolerance on `sd-vulkan.dll`; regenerate only on deliberate changes. They need a Vulkan device: on the development
  machine the RX 7900 XTX; in CI (GitHub runners have no GPU) a software Vulkan driver (SwiftShader or Mesa lavapipe
  for Windows) installed by the workflow, with a looser tolerance because results differ slightly between drivers.
  Tests that need Vulkan are tagged `[vulkan]` as well and skip themselves cleanly when no device exists.
- `[model]`: real checkpoints (`models\v1-5-pruned-emaonly.safetensors`, `models\sd_xl_base_1.0.safetensors`, skipped
  when absent): the graph's text-to-image, img2img, inpaint and LoRA results must match `gh_generate` on
  `sd-vulkan.dll` for the same seed, sampler and scheduler (this is what W4.10 depends on); later ControlNet and each
  new family. The existing CPU-runtime `[model]` tests stay for the Generate path.
- `[api]`: server on an ephemeral port with the fake engine, as `ApiContractTests`: queue, SSE event order, cancel,
  uploads, history, node catalogue schema, permissions per route.
- `[ui]`: the existing offscreen smoke test extended to the Workflows tab (`tools/ui-shots.ps1`); editor model-level
  tests for undo/redo, copy/paste, subgraph convert/unpack and file round trips are `[unit]`.

---

## 13. Phases

Every prerequisite in the main plan is done (phases 0–6, except its stretch items, the MSI and other-hardware checks,
none of which this plan needs). Main-plan step 5.7 (upscaler, ControlNet) is delivered here by W6.2 and W5.2 instead.

### W0 — Spike: the pipeline in pieces (L)
- [x] W0.0 **Repository fix first:** (already in place when this plan started: the patterns are anchored and `GraphicsHost.Core/models/` is tracked; verified with `git status --ignored`, no source file is hidden) `.gitignore`'s unanchored `models/` (meant for the spike's model folder) also
  ignores `GraphicsHost.Core/models/`, so `ModelLibrary`, `ModelCard`, `Catalog`, `DownloadManager`, `HuggingFace`,
  `ModelService`, `FitHints`, `ModelUpdates` and `Recommendation` were never committed, although nine headers are
  included from there and `GraphicsHost.Core.vcxproj` compiles `models\**\*.cpp`. Anchor the data patterns (`/models/`,
  `/DownloadedModels/`, `/logs/`), add the sources from the machine that has them, and check a clean clone builds
  (CI does). The same anchoring keeps new folders of this plan from being ignored (S)
- [x] W0.1 Component functions appended to `gh_runtime.h` under `GH_FEATURE_COMPONENTS`; SD1.5 text encoder, UNet and VAE exposed as separate components in both DLLs (M)
  - Spike shape: the pieces of an already loaded model (`gh_session_begin/end`, `gh_text_encode`, `gh_sigma_schedule`, `gh_initial_latent`, `gh_denoise`, `gh_vae_decode`, `gh_tensor`), implemented in `runtime/gh_components.cpp`, which includes the engine's internal headers (engine include paths and C++17 for that file only). The standalone `gh_component_open` per file arrives with W3. Text still goes through the engine's own tokeniser (host tokenisers are W4.1). Checked by `[model][vulkan][components]`.
- [x] W0.2 Host-side Euler sampler over `gh_denoise`; same seed and settings as `gh_generate` give the same image on Vulkan (M)
  - `sampling/Euler.h` (plain C++ over a float latent). SD 1.5, 512², 6 steps, euler/discrete, cfg 7.5, seed 42: mean difference to `gh_generate` 0.23 of 255 (the engine repeats itself to < 0.05; a different seed differs by > 10). `gh_denoise` batches the prompt and negative prompt into one UNet call like the engine does; evaluated one after the other the difference was 1.9.
- [x] W0.3 Measure per-step overhead of the host loop vs the in-runtime loop on the RX 7900 XTX (SD1.5 512² at 0.35 s/step, SDXL 1024² at 2.89 s/step as measured); budget < 3 % (S)
  - Measured by `[model][vulkan][components][perf]` (slope between a 6-step and a 26-step run, best of two): SD 1.5 512², flash attention on, RX 7900 XTX: 0.0915 s/step in the engine's loop, 0.0920 s/step in the host loop, **0.5 % overhead** (budget 3 %). The "0.35 s/step" quoted above is the older measurement without flash attention.
- [x] W0.4 SDXL through the same path (dual encoders, pooled + size conditioning) (S)
  - `[sdxl]` test: 2048-wide hidden states and a 2816-wide pooled vector (pooled CLIP-G plus the size and crop embeddings); SDXL base 1024², 6 steps, euler/discrete: mean difference to `gh_generate` 0.25 of 255. One host rule was needed: an empty negative prompt on SDXL is encoded as all zeros (`zero_out` argument of `gh_text_encode`); without it the images differed by 7.6.
- [x] W0.5 (done as the `[unit][naming]` test `NamingTests.cpp`, which CI runs with the other unit tests, instead of a script: PowerShell scripts are disabled on the development machine) Naming check in CI (`.github/workflows/build.yml`): a script in `tools/` fails on the reference product's name in any tracked file (the pattern is stored as character codes so the script itself does not contain the word) (S)
- [x] W0.6 **Go/no-go**: host-side sampling confirmed, or fall back to runtime-side samplers with a sampler-id ABI (S)
  - **Go.** Evidence and the notes for later steps in [docs/workflow-w0-results.md](docs/workflow-w0-results.md).
- [x] W0.7 (both halves written; the CI half has never run on GitHub, see below) CI: install a software Vulkan driver on the Windows runner, run `[integration][vulkan]` tests; host code built with `/arch:AVX2` in `props/Core.props` (S)
  - `/arch:AVX2` is set for all first-party code in `props/GraphicsHost.props`. CI (`.github/workflows/build.yml`) downloads Mesa lavapipe (software Vulkan), points `VK_DRIVER_FILES` at it and runs `[integration][vulkan]` (new checkpoint-free case: `sd-vulkan.dll` announces the component functions); `continue-on-error` on the download so a failed install only makes the tests skip. **Not verified:** that step has not run on GitHub, and the fixture-model tests it is meant for (tiny SD-style weights generated with `SafetensorsFixture`) do not exist yet; they come with W1 to W4.

**Exit:** a `[model]` test builds an image from component calls plus a host sampler, identical to `gh_generate`, on
Vulkan. A software Vulkan driver runs the fixture tests in CI (W0.7).

### W1 — Graph model (M–L)
- [ ] W1.1 `tensor/` library; conversion to and from `engine::Image` and the existing `ImageCodec` (S)
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
- [ ] W2.5 Generalise `JobQueue` (`JobWork`, front/back, clear, node progress) without changing the Generate path's behaviour (its `JobQueueTests` stay green); history tables and migration in the gallery database (M)
- [ ] W2.6 Utility nodes: constants, reroute, note, string tools, compare/switch (lazy), preview-any (S)
- [ ] W2.7 `GraphService` in `host/` (validate, submit, history, events) wired into `AppHost` (S)

### W3 — Models, memory, patches (L)
- [ ] W3.1 Model kinds in `ModelLibrary`/`ModelCard` (header detection, kind subfolders, `model-paths.json` extra roots), previews, cached hashes (M)
- [ ] W3.2 Loader nodes: checkpoint, diffusion model, text encoders (1–4), VAE, LoRA (+ model-only), embeddings listing; LoRA names resolve the way `<lora:name:w>` does today (M)
- [ ] W3.3 `ComponentCache` (shared with `ModelManager`), `MemoryManager` with modes, reserve, LRU eviction, partial residency; estimates seeded from `FitEstimator` (L)
- [ ] W3.4 `WeightPatcher`: LoRA/LoCon/LoHa/LoKr/DoRA/full diff key mapping for SD1/SDXL; lazy apply and undo; results match today's per-generation LoRA path (`LoraTests`) (L)
- [ ] W3.5 Precision options and fp8 storage (M)
- [ ] W3.6 "Free memory" / "unload models" actions (S)

### W4 — Text and sampling (XL)
- [ ] W4.1 CLIP BPE tokeniser, prompt parser (weights, embeddings, choices, comments, BREAK) (M)
- [ ] W4.2 Text encode nodes (generic, SDXL, SDXL refiner), clip-skip node; safety filter on graph prompts (S)
- [ ] W4.3 RNGs (cpu-compatible, philox), noise nodes, Brownian-tree SDE noise (M)
- [ ] W4.4 Model sampling types and their nodes (discrete, continuous EDM/V, flow shift) (M)
- [ ] W4.5 All schedulers and sigma nodes (M)
- [ ] W4.6 All samplers incl. CFG++ variants, toy-denoiser convergence tests (XL)
- [ ] W4.7 Guiders: basic, CFG, dual CFG, perpendicular negative; rescale CFG, CFG-Zero*, APG (M)
- [ ] W4.8 Sampler and Sampler (advanced) nodes, custom-sampling nodes, inpaint noise mask, start/end steps (M)
- [ ] W4.9 Previews: linear latent-to-RGB per family, tiny preview decoders (S)
- [ ] W4.10 `GenerationService` builds a fixed workflow from a `GenerationRequest` (txt2img, img2img, inpaint, LoRA) instead of calling `gh_generate` on the Vulkan runtime (the CPU runtime keeps `gh_generate`), behind a setting until the `[model]` parity tests pass for every sampler/scheduler in `knownSamplers()`/`knownSchedulers()`; then it becomes the default. Graph sampler ids map to the existing names (`euler_a`, `dpm++2m`, ...) so presets, per-model defaults and API requests keep working (M)

**Exit:** the default text-to-image workflow runs from a job file on Vulkan; every sampler × scheduler pair
produces an image on SD1.5; the `[integration]` fixture tests and the `[model]` parity tests pass.

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
- [ ] W6.4 Load image/mask/latent, save/preview image (PNG/JPEG/WebP via `ImageCodec`, workflow chunks next to the existing parameter chunk, saved through `GalleryStore`), save latent, filename tokens and counter (M)
- [ ] W6.5 Tiled VAE encode/decode (spatial + temporal) (S)

### W7 — Server API and events (M)
- [ ] W7.1 `GraphRoutes`: routes of §7 with permissions, validation errors in the existing error shape (M)
- [ ] W7.2 SSE graph events on `SseStream`, preview files in temp, `Last-Event-ID` replay, per-client filter (S)
- [ ] W7.3 `FileRoutes`: uploads with their own size cap, views (with preview re-encoding), user data (S)
- [ ] W7.4 API contract tests in `ApiContractTests` style; `docs/api.md` section for the graph API (S)
- [ ] W7.5 CLI: `graph run <file> [--set node.input=value] [--json]` (prints progress on stderr like `gen`, same exit codes), `graph validate`, `graph nodes [type]` (S)

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
- [ ] W8.14 Extend the existing `MaskEditor`; crop tool and compare view on `ImageView` (S)
- [ ] W8.18 Gallery integration: "open workflow" on gallery entries that carry one; Generate tab "open as workflow" (S)
- [ ] W8.15 Settings dialog, keymap editor (M)
- [ ] W8.16 Missing nodes / missing models dialogs with download (S)
- [ ] W8.17 Drag-and-drop: workflow files, images/video/audio with embedded workflows, images onto Load Image nodes (S)

**Exit:** a user can build, run, save and reopen the default, img2img, inpaint, hi-res-fix, ControlNet and LoRA templates
without touching a file; every parameter shows its tooltip.

### W9 — Browser editor (L)
- [ ] W9.1 Spike: Qt for WebAssembly build of the canvas from MSBuild (M)
- [ ] W9.2 `EditorBackend` over HTTP and SSE; uploads, views (M)
- [ ] W9.3 Copy next to the exe, add to `Package.proj`, serve at `/editor/`; tokens; LAN smoke test (S)

### W10 — Model families and patches (XL, split per family)
Each family = detection, ggml architecture (reuse from stable-diffusion.cpp when present, else new in `runtime/arch/`),
text encoders and tokeniser, VAE, latent format and preview factors, model-sampling defaults, family-specific nodes,
LoRA key mapping, catalog entries (`data/catalog.json`) with companions and fit hints, a template, a fixture test,
a `[model]` test, and a VRAM measurement row in `docs/vram-measurements.md`. Families that need a newer
stable-diffusion.cpp start with the submodule upgrade step (§5).
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
- [ ] W13.1 Tooltips into the existing translation files (`GraphicsHost/translations/`, German first as today) (S)
- [ ] W13.2 Accessibility: keyboard-only editing, focus order, screen-reader names for nodes and widgets (M)
- [ ] W13.3 Performance pass: 2 000-node graph at 60 fps, executor overhead < 1 ms per cached node (S)
- [ ] W13.4 User docs generated from the registry (node pages with tooltips and examples); README feature table (S)
- [ ] W13.5 Package: the WebAssembly editor, templates and the plugin SDK in the portable zip; hardware checklist rows for graph jobs (S)

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
| §5 standard parameter set, Generate tab, `gen`, OpenAI routes | unchanged for users; implemented on a fixed workflow once parity holds (W4.10) |
| `ModelManager` one-model policy | stays for the Generate path; graph jobs use `ComponentCache`, which `ModelManager` consults (W3.3) |
| `JobQueue` | generalised to graph jobs, one queue for both (W2.5) |
| Step 5.7 upscaler / ControlNet (open, stretch) | delivered by W6.2 and W5.2; tick 5.7 with a pointer when they land |
| §7 browser UI in HTML/CSS/JS (done, `web/`) | kept; the workflow editor in the browser is the WebAssembly build (W9). Retiring `web/` afterwards is a later decision |
| Model families "v1: SD1/SDXL only" | still true for the first release; W10 adds families afterwards |
| Hardware "CPU and Vulkan on AMD" | workflows are Vulkan only and assume AVX2; `sd-cpu-avx2.dll` and the CPU device options remain only for the existing Generate path, which keeps calling `gh_generate` when the CPU runtime is selected (also after W4.10) |
| `.gitignore` | data-folder patterns anchored so source folders are never ignored (W0.0) |
| §11 vcpkg packages | add: a video codec library (§15) and the Emscripten toolchain + Qt for WebAssembly for W9 (not from vcpkg); tokenisers are our own code. No WebSocket library is needed (§7) |

---

## 15. Risks

- **The component functions depend on stable-diffusion.cpp internals.** Its model classes are not a public API; pin and wrap only in
  `runtime/`, as for ABI v1. Upgrades are deliberate and covered by golden tests.
- **New architectures in ggml are large work** (W10/W11), and the Vulkan backend may lack ops some models need. With
  no CPU path there is no fallback: a missing op is a Vulkan shader we write (in ggml's style, in the pinned submodule's
  shader set or a local patch) before that family ships. Each W10/W11 item starts with an op inventory against
  `ggml-vulkan` so this shows up early.
- **Software Vulkan in CI** may be slow, may lack features (cooperative matrices, some fp16 paths) and differs
  numerically from the GPU; fixture models stay tiny, and GPU-only checks run on the development machine before a
  phase is ticked.
- **Video encoding in C++** needs a codec library: libvpx (WebM/VP9, BSD) and/or OpenH264 (BSD, Cisco binary licence) via
  vcpkg; MP4 muxing written by us or via a small muxer library. Decide in W11.1.
- **Qt for WebAssembly from MSBuild** (W9.1 spike, fallback described in §9).
- **Memory estimation** for partial residency is approximate; measure per family and keep a safety margin setting.
- **Native plugins can crash the process.** Safe mode, auto-disable after a crash on load, and (later) an out-of-process
  plugin host using the same ABI over IPC.
- **Clean room discipline** must hold for every contributor; record in each PR template that no reference-product source
  was consulted.
