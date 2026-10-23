# GraphicsHost — implementation plan (C++)

A local **image-generation host**, modelled on [LLMHost](../LLMHost/README.md): one native executable that discovers and
downloads models, runs them in-process on **CPU or GPU (Vulkan)**, exposes one standard set of generation parameters, and
serves them through a desktop window, a CLI and an HTTP API.

LLMHost is C# (LLamaSharp over llama.cpp). Here the whole program is **C++20**, which removes the wrapper layer: we link
**stable-diffusion.cpp** (the ggml-based sibling of llama.cpp) directly and call its C API.

Effort: **S** ≤ 2 d · **M** 3–5 d · **L** 6–10 d · **XL** > 10 d (one engineer, includes tests).

---

## 1. What we copy from LLMHost, and what changes

| Concern | LLMHost | GraphicsHost (C++) |
|---|---|---|
| Engine | llama.cpp via LLamaSharp (P/Invoke) | **stable-diffusion.cpp** linked directly (`stable-diffusion.h`), built from source with ggml CPU + Vulkan |
| Model format | GGUF | GGUF first; also `.safetensors` / `.ckpt` single-file checkpoints, which stable-diffusion.cpp reads and can quantise |
| Model families | chat LLMs | **v1: SD 1.x and SDXL (incl. Turbo/Lightning) only**, plus VAE, LoRA. SD3 / FLUX are post-v1 |
| Output | token stream | PNG images + progress callbacks per sampling step |
| UI | WinForms + WebView2 + web UI | **Qt 6 Widgets desktop UI + a browser UI** served by the built-in HTTP server |
| Hardware (v1) | CPU or Vulkan (AMD, NVIDIA, Intel) | **CPU and Vulkan on AMD only.** Other vendors may work but are not tested or supported |
| API | ASP.NET minimal API (OpenAI / Anthropic / native) | C++ HTTP server: OpenAI `/v1/images/generations` + `/v1/images/edits`, plus native `/api/v1` |
| Discovery | `catalog.json` + Hugging Face search | same, filtered to diffusion models (§6) |

### Why C++ simplifies the engine, and what it costs

- **Gain:** no binding question. No marshalling, no wrapper lagging behind upstream; we can track stable-diffusion.cpp
  master, use its progress/log callbacks directly, and compile exactly the ggml backends we want.
- **Cost:** we own the build (MSBuild projects for ggml, Vulkan shader compilation with `glslc`), memory safety, and every piece LLMHost got for
  free from the .NET BCL and ASP.NET (HTTP, JSON, async, DI, settings binding). Those are covered by libraries in §2.

---

## 2. Toolchain, build and dependencies

- **Language / compiler:** C++20, MSVC from Visual Studio 2026, `/W4 /permissive-`, x64 only (ARM64 later).
- **Build system: Visual Studio 2026 solution and project files, no CMake.** `GraphicsHost.slnx` holds MSBuild `.vcxproj`
  projects (§3 lists them). The empty C# WPF project currently in this repo is removed in Phase 1.
- **Packages:** **vcpkg in manifest mode** (`vcpkg.json` at the repo root, `vcpkg integrate install` once). All vcpkg
  libraries are **dynamic**: the stock triplet `x64-windows` (DLLs, dynamic CRT `/MD`), with no custom triplet. The complete
  package list is in §11. A vcpkg checkout already exists at `D:\code\vcpkg`.
- **Deployment consequence of all-dynamic.** The vcpkg DLLs (OpenSSL, curl, libpng, spdlog, SQLite, …) and the Qt DLLs ship
  next to `GraphicsHost.exe`. vcpkg's MSBuild integration copies the DLLs to the output folder on build
  (`VcpkgApplocalDeps`); `windeployqt` adds the Qt plugins. The installer (Phase 6) packages that folder, plus the VC++
  Redistributable or an app-local copy. This also keeps Qt LGPL-compliant.
- The `sd-*.dll` runtimes are separate DLLs by design. They expose a plain C ABI and link ggml/stable-diffusion statically
  (our own `.vcxproj` static libs), so they do not need the vcpkg DLLs.
- **Qt in MSBuild:** use the **Qt Visual Studio Tools** extension, which runs `moc`, `uic`, `rcc` and `lupdate` as normal MSBuild
  steps and points the project at the vcpkg Qt (`QtInstall` = `D:\code\vcpkg\installed\x64-windows`). The alternative, plain
  custom build steps calling `moc.exe`, works but is more to maintain.
- **stable-diffusion.cpp is not consumed through vcpkg.** It is a pinned git submodule (`extern/stable-diffusion.cpp`, with its
  `ggml` submodule), compiled by our own `ggml.vcxproj` + `stable-diffusion.vcxproj` (static libraries). This is the hard part
  of choosing MSBuild over CMake: ggml's Vulkan backend needs shader sources compiled at build time. See the Phase 0 spike and
  the first risk in §10.

| Need | Source | Notes |
|---|---|---|
| Diffusion engine | **stable-diffusion.cpp** submodule, pinned commit | ggml CPU backend always, ggml-vulkan for the GPU runtime |
| Vulkan build tools | Vulkan SDK **or** vcpkg `shaderc` | `glslc` compiles ggml's `.comp` shaders; runtime needs only the AMD driver (Adrenalin) |
| Everything else | vcpkg packages | §11 |

- **Backends shipped as separate DLLs** (the C++ equivalent of LLMHost's separate CPU/Vulkan native folders):
  `sd-cpu-avx2.dll`, `sd-cpu-noavx.dll` and `sd-vulkan.dll` (AVX-512 optional later), each exporting the same small C ABI
  (`gh_load`, `gh_generate`, `gh_free`, …) implemented by a thin shim around `stable-diffusion.h`. The host loads exactly one
  with `LoadLibrary` at startup. This keeps the host independent of ggml's CPU-feature selection and lets a user-installed
  newer runtime replace a bundled one without recompiling the app. Each is its own `.vcxproj` using the same shim source with
  different preprocessor defines.
- **Error policy:** exceptions inside our code, `noexcept` C ABI at the DLL boundary (errors returned as codes + message).
  RAII everywhere; no raw `new`/`delete`. Run MSVC `/analyze`, clang-tidy, and AddressSanitizer builds in CI.

---

## 3. Architecture

```
GraphicsHost/
├─ GraphicsHost.slnx, vcpkg.json, Directory.Build.props   shared C++20 / warning / vcpkg settings
├─ extern/stable-diffusion.cpp          pinned submodule (+ ggml)
├─ ggml/                 ggml.vcxproj            static lib: ggml core + CPU backend (+ Vulkan backend, shader step)
├─ sdcpp/                stable-diffusion.vcxproj static lib
├─ runtime/              sd-cpu-avx2 / sd-cpu-noavx / sd-vulkan .vcxproj -> sd-*.dll   (only code that includes ggml headers)
├─ GraphicsHost/         GraphicsHost.vcxproj    the application (Qt Widgets, CLI, server) - single exe
│  ├─ core/      settings, paths, logging, migrations, hardware, AppHost (composition root)
│  ├─ engine/    IImageEngine, RuntimeLoader, GpuDevices, ModelManager, JobQueue, FitEstimator
│  ├─ models/    Catalog, HuggingFace, DownloadManager, ModelLibrary, ModelCard, Gguf/Safetensors readers
│  ├─ server/    HTTP routes, auth, request log, metrics, static web UI hosting
│  ├─ cli/       verbs, --json, exit codes
│  ├─ ui/        Qt desktop application (view models + views)
│  └─ web/       browser UI (HTML/CSS/JS), embedded as resources
└─ GraphicsHost.Tests/   GraphicsHost.Tests.vcxproj   unit / api / integration (Catch2)
```

- **One process, one engine host.** UI, CLI and HTTP server all call the same services from `AppHost`; nothing UI-specific
  lives below `ui/`.
- **Threading model:** one **engine thread per loaded model** (stable-diffusion.cpp contexts are not thread-safe); a
  `JobQueue` feeds it (FIFO, cancellation token, progress callback). HTTP handlers and UI never call into the engine
  directly. A small `asio`-style thread pool (or `std::jthread` + a work queue) serves HTTP and downloads.
- **Cancellation:** the shim checks an atomic flag in the sampling-step callback, so Cancel takes effect at the next step.

---

## 4. Engine layer (hosting + Vulkan/CPU)

**Reference:** LLMHost `Inference/ModelRuntime.cs`, `ModelManager.cs`, `NativeBackend.cs`, `RuntimeManager.cs`, `GpuDevices.cs`.

```cpp
struct LoadOptions;          // see below
struct GenerationRequest;    // §5
struct GenerationResult { std::vector<Image> images; std::vector<int64_t> seeds; std::vector<std::string> warnings; Timings t; };

class ILoadedModel {
public:
    virtual ~ILoadedModel() = default;
    virtual const ModelInfo& info() const = 0;
    virtual GenerationResult generate(const GenerationRequest&, ProgressSink&, std::stop_token) = 0;
};
class IImageEngine {
public:
    virtual std::unique_ptr<ILoadedModel> load(const LoadOptions&, std::stop_token) = 0;
};
```

- **`LoadOptions`** (per load): model path, optional VAE / CLIP-L / CLIP-G / T5 paths (resolved from the model card, §6),
  `device` (`cpu` | `vulkan:<index>`), `threads`, `weight_type` override (quantise on load: f16, q8_0, q5_0, q4_0),
  `vae_tiling`, `clip_on_cpu`, `vae_on_cpu`, `offload_params_to_cpu` (low-VRAM mode), `flash_attention`, LoRA directory,
  idle-unload TTL.
- **Runtime selection (CPU vs Vulkan).** `RuntimeLoader` enumerates bundled + user-installed runtime folders, marks
  compatibility (CPUID for AVX/AVX2/AVX-512; Vulkan loader + a usable device for GPU), persists the choice, applies it on the
  next start. Setting `backend = auto | vulkan | cpu`; `auto` = Vulkan if a usable device exists, else CPU.
- **GPU discovery.** `GpuDevices` calls `vkEnumeratePhysicalDevices` / `vkGetPhysicalDeviceMemoryProperties` (and DXGI for live
  VRAM use on Windows) to list name, vendor, VRAM; feeds the device picker, the hardware monitor and `FitEstimator`.
- **Fit estimation.** Weights + text encoders + VAE + activation memory at the requested width × height × batch; warn before
  loading ("fits VRAM" / "needs offload" / "CPU only, expect minutes").
- **Fallback.** If a Vulkan load fails (OOM, driver), *offer* a retry on CPU; never switch silently; log the native error.
- **Several models at once** (only if memory allows), JIT load with eviction and idle TTL (LLMHost §6.1 behaviour). Default: one loaded model, the
  previous one evicted, because image models are large.
- **v1 hardware scope:** CPU and **Vulkan on AMD** only. `GpuDevices` still lists every Vulkan device, but non-AMD devices are
  labelled "unsupported in this version" and `backend = auto` never picks them. No ROCm/HIP/CUDA anywhere. Test on at least
  one RDNA card and, if available, one older GCN card, plus the integrated Radeon in APUs (shared memory: VRAM numbers
  from Vulkan heaps are misleading there, so `FitEstimator` treats an iGPU's budget as a fraction of system RAM).
- **Risks:** ggml Vulkan behaviour differs per driver (AMD Adrenalin vs. Mesa is out of scope on Windows); CPU is slow for
  SDXL. Both go on the model card as "expected speed". SDXL at 1024² on a card with < 8 GB VRAM needs `vae_tiling` and
  `offload_params_to_cpu`; the fit estimator proposes them automatically.

**Effort:** L.

---

## 5. Standard parameter set

One `GenerationRequest` struct, independent of model family and API. Fields a family does not support are ignored and reported
in `warnings`.

| Group | Parameter | Notes |
|---|---|---|
| Core | `prompt`, `negative_prompt` | Turbo-style SDXL models use CFG ~1 and ignore it |
| | `width`, `height` | multiples of 8 (validated); defaults per family |
| | `steps`, `cfg_scale` | defaults from the model card |
| | `seed` (−1 = random), `batch_count` (`n`) | the returned seed is always the real one |
| Sampler | `sampler` | euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm, ipndm, … |
| | `scheduler` | discrete, karras, exponential, ays, gits, … |
| | `clip_skip` | |
| Image-to-image | `init_image`, `strength`, `mask_image` | inpainting |
| Extras | `loras: [{name, weight}]` | also parsed from `<lora:name:0.8>` in the prompt |
| | `control_image`, `control_strength` | ControlNet, phase 5 |
| | `vae_tiling`, `upscale` | ESRGAN upscaler optional, phase 5 |
| Output | `format` (png / jpeg / webp), `embed_metadata` | PNG text chunk with the full parameter set so any image can be re-created |
| Safety | `safety_filter` (bool) | optional; **off by default** (setting `safety.enabled`, overridable per request). See §7 |

**Plumbing** (copy LLMHost `docs/adding-options.md`): declare in the struct → validator returning `{field, message}` list →
per-model defaults → presets → JSON (de)serialisation (`to_json`/`from_json` in one place) → API mapping → CLI flag → UI
parameter panel → tests. Write the checklist as `docs/adding-options.md` in Phase 1. Each option carries an interface-mode
tag (User / Power user / Developer) so advanced samplers and offload switches can be hidden.

**Effort:** M.

---

## 6. Settings, storage, model discovery and download

### Settings and storage
**Reference:** LLMHost `Core/Settings.cs`, `Migrations.cs`, `SettingsValidator.cs`, `PresetFiles.cs`.

- `%USERPROFILE%\.graphicshost\`: `settings.json` (with `schemaVersion` + a migration list, backups `.bak-vN`, from day
  one), `presets/*.json`, `model-defaults.json`, `runtimes/`, `logs/`, `outputs/YYYY/MM/DD/*.png`.
- Files are written atomically (write temp + `MoveFileEx` replace); config reads tolerate unknown keys.
- **Gallery store:** every image stored with its parameters (PNG text chunk + sidecar JSON); UI gallery, "reuse settings" and
  API history read this one source. Index in SQLite (`sqlite3` via vcpkg) so search by prompt is fast.
- Presets: named `GenerationRequest` templates, live reload (directory watcher), import/export.

### Discovery and download
**Reference:** LLMHost `Models/Catalog.cs`, `catalog.json`, `HuggingFace.cs`, `DownloadManager.cs`, `ModelLibrary.cs`,
`ModelDirectories.cs`, `Gguf.cs`, `ModelCard.cs`, `ModelUpdates.cs`.

- **Library scan.** Watch model folders (`publisher/repo/file` layout). Detect family from metadata: GGUF header reader
  (port `Gguf.cs`), and for `.safetensors` parse the 8-byte-length-prefixed JSON header for tensor names/shapes to tell SD1.5 /
  SDXL apart (SD3 / FLUX are only detected, to report "not supported in this version"). Result: a `ModelCard` with family, default size/steps/cfg/sampler and required companions.
  Files are memory-mapped (`CreateFileMapping`) for the header read only.
- **Companion files.** Some SD1.5/SDXL checkpoints need a separate VAE (e.g. an fp16-safe SDXL VAE). The card lists it; the
  library resolves it and **downloads it together with the main model** (the equivalent of LLMHost's auto-fetched `mmproj`).
  The mechanism is general, so FLUX/SD3 text encoders can be added after v1.
- **Curated catalog** (`catalog.json` embedded as a resource, dated): `key`, `family`, `repo`, `file`, `quant`,
  `companions[]`, `tags` (`txt2img`, `fast`, `inpaint`, `sd15`, `sdxl`…), approximate size, minimum VRAM, licence note.
- **Live Hugging Face search** (automatic discovery): HF API for repos tagged `text-to-image` that contain GGUF or
  safetensors diffusion files; filter by family, size, "fits this GPU/RAM"; sort by downloads/likes; show licence and gating.
- **Gated models:** HF-token + accept-terms flow (token kept in Windows Credential Manager, not in `settings.json`).
- **Downloads.** Port `DownloadManager` onto libcurl: HTTP range resume, queue, pause/resume, free-space check, SHA-256,
  split files, progress events to UI/CLI/API.
- **Update check** against the repo's latest revision (`ModelUpdates.cs`).
- **First run wizard:** survey hardware (GPU, VRAM, RAM, AVX), recommend a model that fits.

**Effort:** M (settings/storage) + L (models).

---

## 7. API surface

**Reference:** LLMHost `Server/ApiServer.cs`, `ApiAuth.cs`, `NativeApi.cs`, `NetworkSecurity.cs`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/images/generations` | OpenAI-compatible: `prompt`, `n`, `size` ("1024x1024"), `response_format` (`b64_json`/`url`); our extra parameters accepted as top-level fields |
| `POST /v1/images/edits` | img2img / inpainting (multipart `image`, `mask`) |
| `GET /v1/models` | loaded and installed image models |
| `POST /api/v1/generate` | native: full parameter set, returns a job id; `GET /api/v1/jobs/{id}` (poll), `/stream` (SSE step progress), `DELETE` cancels |
| `GET/POST /api/v1/models`, `/load`, `/unload` | library; load/unload with `LoadOptions` |
| `GET/POST /api/v1/downloads`, `GET /api/v1/discover` | catalog, HF search, queue downloads |
| `GET /api/v1/gallery`, `GET /outputs/...` | history and image files |
| `GET /metrics`, `GET /api/requests` | Prometheus text + request log |

Hardening as in LLMHost: API tokens with permissions (stored hashed), localhost-only by default, `--lan`, optional HTTPS
(OpenSSL; generated self-signed or user certificate), IP allow-list, body-size and rate limits, JIT load policy. Request bodies
are size-capped *before* parsing, since base64 images are large.

**Safety filter (optional, off by default).** A post-generation classifier hook (`ISafetyFilter`) that runs on each output
image when `safety.enabled` is true or the request sets `safety_filter: true`. Flagged images are replaced by a blurred
placeholder and the result carries `"safety": {"flagged": true}`. The classifier is a small ONNX/ggml model downloaded on
first enable (not bundled); the interface exists in v1 and the first implementation is picked in Phase 5. Because it is off
by default, no v1 path depends on it.

**Browser UI.** The same server hosts the web UI at `/` (embedded static files, vanilla HTML/CSS/JS with no build step or
CDN, so it works offline and over LAN). It talks only to `/api/v1` (the same endpoints the Qt UI's services use) and covers
the Generate, Gallery, My Models, Discover, and Settings tabs with feature parity as the goal; the first release may omit
inpaint-mask drawing. The server is localhost-only unless `--lan` is set, and the browser UI uses the same tokens as the API.

**Effort:** L.

---

## 8. Desktop UI

**Qt 6 Widgets** (decided), dynamically linked for LGPL compliance (ship the licence texts, no static Qt). Charts for the
hardware monitor use **Qt Charts**; the gallery uses a `QListView` with a lazy thumbnail model; context menus are plain
`QMenu`. UI code stays in `ui/` behind view-model-style classes, and the browser UI (§7) is a second client of the same
services, so features are built as services first and screens second. Qt's event loop runs on the main thread; engine,
download and server callbacks post to it with `QMetaObject::invokeMethod`.

| Tab | Contents |
|---|---|
| **Generate** | Prompt + negative prompt, parameter panel (mode-gated), model + device picker, Generate/Cancel, live step progress and preview, result grid, batch, "send to img2img", seed lock, save/copy/open folder, context menus on images |
| **Gallery** | Thumbnails (lazy model/view), search by prompt, "reuse parameters", delete, export |
| **My Models** | Library with family/size/fit badges, load dialog (`LoadOptions`), per-model defaults, companion-file status, disk usage, folders, import |
| **Discover** | Catalog + live HF search, filters, fit hints, gated flow, download queue |
| **Requests / Logs** | API request log, application log with level filter |
| **Settings** | Server (port, LAN, tokens, HTTPS), runtimes + backend `auto/vulkan/cpu`, GPU device, hardware monitor graphs (GPU util / VRAM), interface mode, theme, models folder |

Also: tray icon, single instance (named mutex), remembered window state, light/dark/accent themes, all strings through
Qt's translation system from the start.

**Effort:** XL.

---

## 9. CLI

`graphicshost` with LLMHost's conventions: `--json`, `--quiet`, and the same exit-code table (0 ok, 1 failure, 2 usage,
3 not found, 4 not authorised, 5 server unreachable, 6 out of memory, 7 cancelled, 8 conflict). Console build as a second
executable (`graphicshost.com`-style) or `AttachConsole` in the GUI exe so output reaches the terminal.

Verbs: `catalog`, `search`, `get`, `downloads`, `ls`, `load` / `unload` / `ps`, `gen "<prompt>" [--model --steps --seed --size …]`,
`serve`, `daemon` (Windows service via the SCM API), `preset`, `config`, `token`, `runtime`, `hardware`, `log`.
No verb = open the desktop window with the API running.

**Effort:** M.

---

## 10. Phasing, testing and decisions

Each step is a checkbox (`- [ ]`); tick it and add the commit hash when done. A phase is finished only when its **exit
criterion** holds. Effort tags: **S** ≤ 2 d · **M** 3–5 d · **L** 6–10 d · **XL** > 10 d. Steps marked *(AMD)* need the AMD test
machine; steps marked *(stretch)* can slip past v1 without breaking it.

### Phase 0 — Spike: prove the engine builds and runs without CMake (M–L)

Goal: retire the biggest risk (§10) before any application code exists.

- [x] 0.1 Install the toolchain: VS2026 C++ workload, Vulkan SDK (or vcpkg `shaderc`), Git; bootstrap the existing `D:\code\vcpkg` (S)
- [x] 0.2 Add `extern/stable-diffusion.cpp` (+ `ggml`) as a pinned submodule; record the commit (S) — pinned `fd21ecf` (master-995), ggml `c1f80e5`
- [x] 0.3 `ggml.vcxproj`: static library, ggml core + CPU backend, builds clean with `/W3` (M) — CPU variant avx2; `ggml/ggml.vcxproj`, Release builds clean
- [x] 0.4 `stable-diffusion.vcxproj`: static library on top of `ggml.vcxproj` (S) — `stable-diffusion/stable-diffusion.vcxproj`, also builds zip, utf8proc, oniguruma; hand-written oniguruma `config.h`
- [x] 0.5 Console spike project: load an SD1.5 model, generate a 512×512 image on **CPU**, save PNG (S) — `spike/spike.vcxproj`; 18.2 s on CPU, see `docs/phase0-results.md`
- [x] 0.6 Vulkan backend: build `vulkan-shaders-gen` as its own `.vcxproj`, run it with `glslc` in a pre-build step, add the generated sources to `ggml.vcxproj` (L) — builds: `/p:GgmlVulkan=true`, ~1m50 clean; GPU runtime check blocked by the RX 7900 XTX driver error (code 31), see `docs/phase0-results.md`
- [x] 0.7 Generate the same SD1.5 prompt on the AMD GPU *(AMD)*; then one SDXL 1024² image on CPU and GPU; record time, VRAM and any driver errors (M) — done 2026-10-23: SD1.5 512: CPU 18.2 s, GPU 6.9 s; SDXL 1024: CPU 103 s, GPU 25.6 s; see `docs/phase0-results.md`
- [x] 0.8 **Go/no-go:** if 0.6 is too fragile, switch `sd-vulkan.dll` to the prebuilt-release fallback and note it in §10 (S) — go: the MSBuild shader path works and generates correct images on the RX 7900 XTX, no prebuilt-release fallback needed

**Exit:** one SD1.5 and one SDXL image generated on CPU and on AMD Vulkan, timings recorded, shader build step documented. *Met on 2026-10-23 after the GPU driver was repaired: CPU and Vulkan images for both models, see docs/phase0-results.md.*

### Phase 1 — Foundations: solution, dependencies, plumbing (M–L)

- [x] 1.1 Remove the empty C# WPF project and `GraphicsHost.slnx`; create the new `.slnx` with the project set from §3 (S) — `GraphicsHost.slnx` builds the native projects; more projects are added by later steps
- [x] 1.2 `Directory.Build.props`: C++20, `/W4 /permissive-`, output folders, vcpkg switches; `vcpkg.json` with the §11 packages; `vcpkg integrate install` (S) — `props/GraphicsHost.props` (first-party only), `vcpkg.json` documents the set; MSBuild stays in classic mode (`VcpkgEnableManifest=false`) so Qt is never rebuilt from source
- [x] 1.3 Move the spike's static libs into the solution; add the three runtime projects `sd-cpu-avx2`, `sd-cpu-noavx`, `sd-vulkan` exporting the C ABI shim (`gh_load`, `gh_generate`, `gh_free`, cancel) (M) — `runtime/gh_runtime.h` C ABI v1; three DLLs build and export it; noavx and vulkan load, avx2 was first refused by Application Control then loaded after a rebuild (docs/environment-notes.md)
- [x] 1.4 `GraphicsHost.vcxproj` with Qt Visual Studio Tools: a main window that opens and closes; `windeployqt` post-build step (M) — own moc/plugin-deploy MSBuild integration (`props/Qt.props`, `Qt.targets`) instead of the Qt VS Tools extension; builds and deploys; `GraphicsHost.exe --smoke-test` opens the window and exits 0 with both the offscreen and the Windows platform plugin
- [x] 1.5 `core/`: paths (`%USERPROFILE%\.graphicshost`), `Settings` with `schemaVersion` + migrations + validator, atomic file writes (M) — in `GraphicsHost.Core` (new static library shared by app, CLI and tests): `core/Paths`, `AtomicFile`, `Settings`, `Migrations`, validator; unit tests follow in 1.9
- [x] 1.6 `core/`: spdlog logging (JSON-lines file sink, ring buffer for the UI), native log callback routed in (S) — `core/Log`: JSON-lines rotating file + 2000-entry ring buffer for the UI, engine log hook (`nativeLog`); tests follow in 1.9
- [x] 1.7 `core/`: `AppHost` composition root; command-line entry (`CLI11`) and GUI/CLI mode switch (S) — `host/AppHost` (composition root), `cli/Cli` (CLI11: version, hardware, runtime info, config show|path|set, `--json`, exit-code table), GUI by default and CLI when arguments are given; verified by running `hardware`, `--json hardware`, `runtime info` and the CLI test cases
- [x] 1.8 `engine/`: `GpuDevices` (Vulkan enumeration, AMD flag, VRAM), CPU feature detection, `RuntimeLoader` choosing and loading one `sd-*.dll` (M) — `engine/CpuFeatures`, `GpuDevices` (dynamic vulkan-1.dll, AMD-only support flag), `RuntimeLoader` (discovery, pure `chooseRuntime`, DLL wrapper with Application Control hint); tests follow in 1.9
- [x] 1.9 `GraphicsHost.Tests` project with Catch2 and the `[unit]/[api]/[integration]` tags; first tests for settings and migrations (S) — settings, migrations, log, engine selection, CLI: 39 test cases / 167 assertions pass, including the `[integration]` cases that load a real runtime DLL (2026-10-23)
- [x] 1.10 `docs/adding-options.md` (option-plumbing checklist) and a README with the build instructions (S) — `docs/adding-options.md`, README with build, run, test and layout
- [x] 1.11 CI: build the solution, run the unit tests, `/analyze`, vcpkg binary cache (M) — `.github/workflows/build.yml` (build, unit and integration tests, /analyze via `-p:GhAnalyze=true`, vcpkg binary cache); /analyze verified locally on Core, the workflow itself has NOT run on GitHub yet

**Exit:** a clean checkout builds in one step; the app starts, lists GPUs, picks CPU or Vulkan, loads a runtime DLL; unit tests pass. *Met on 2026-10-23: the whole solution builds in one step at /W4 with no warnings, /analyze is clean on the first-party code, all 39 test cases pass, the app starts, lists the RX 7900 XTX, auto-selects the Vulkan runtime and loads it (`runtime info` shows `Vulkan0 AMD Radeon RX 7900 XTX`). The GitHub workflow has not run yet.*

### Phase 2 — Engine: generate from the command line (L)

- [x] 2.1 `GenerationRequest` struct (§5), defaults, JSON (de)serialisation in one place, validator returning `{field, message}` (M) — `engine/GenerationRequest` (+ `LoadOptions`): defaults mean "model default", `resolve()` fills from family defaults, partial-JSON apply for presets, validator with field names; 10 test cases
- [x] 2.2 `IImageEngine` / `ILoadedModel` implemented over the runtime DLL: load with `LoadOptions`, unload (M) — `engine/ImageEngine.h` (`IImageEngine`, `ILoadedModel`, typed `EngineError`), `RuntimeEngine.cpp` over the runtime DLL; native error capture explains OOM; progress is reported as steps 1..N; 7 real-model test cases
- [x] 2.3 Engine thread per loaded model, `JobQueue`, progress callback per step, cancellation via the step callback (M) — `engine/JobQueue`: one worker thread (the runtime ABI allows one generation at a time per process, so the queue is per process, not per model), FIFO, observable progress, cancel of queued and running jobs, last 100 finished jobs kept; 9 test cases against a fake model, run 8 times without a flake
- [x] 2.4 txt2img for SD1.5 and SDXL with sampler/scheduler/seed/steps/CFG/size/batch; returned seed is the real seed (M) — all 15 samplers and 11 schedulers in `knownSamplers()/knownSchedulers()` run on SD1.5, SDXL runs with its family defaults, odd sizes, VAE tiling, real seed reported; tests tagged `[model]` (skipped without checkpoints)
- [x] 2.5 Output: PNG with parameter text chunk (`libpng`), JPEG, WebP; reproducibility test (same seed gives the same image on CPU) (S) — `engine/ImageCodec`: PNG (tEXt chunk `graphicshost`), JPEG (COM marker), WebP (XMP chunk, lossy and lossless) with decoders and `readImageMetadata`; same-seed reproducibility is covered by the 2.2 tests; 7 test cases
- [x] 2.6 Gallery store: `outputs/YYYY/MM/DD`, sidecar JSON, SQLite index (M) — `gallery/GalleryStore`: dated folders, sidecar JSON, SQLite index (WAL), list/search/get/remove, `rescan()` rebuilds the index from the sidecars; 10 test cases incl. concurrency and collisions
- [x] 2.7 `ModelManager`: load, unload, idle TTL, one-model-loaded policy with eviction (S) — `engine/ModelManager`: one model loaded, reuse on identical options, eviction only while the queue is idle, idle TTL via injectable clock plus background sweeper, owns the JobQueue; 12 test cases, stable over 5 runs
- [x] 2.8 Fit estimator v1 (weights + VAE + activations at W×H×batch) with the low-VRAM suggestions (`vae_tiling`, `offload_params_to_cpu`) (M) — `engine/FitEstimator`: constants calibrated on 9 measured runs (`docs/vram-measurements.md`, `tools/measure-vram.ps1`); suggests offload and flash attention (found: SD1.5 at 768 is 5x faster with it); auto falls back to CPU, an explicit GPU never does; 12 test cases
- [x] 2.9 CPU→Vulkan fallback *offer* on load failure; error codes and the exit-code table (S) — `engine/LoadAdvice` (offers offload, then CPU; never retries by itself), `cli::exitCodeFor` maps `EngineErrorCode` to the exit-code table; 5 test cases. The CLI prints the offers in 2.10
- [x] 2.10 CLI verbs `gen`, `ls`, `load`/`unload`/`ps`, `hardware`, `config`, `--json` (M) — `gen` (all parameters as flags, `--preset`, `--dry-run`, `-o`, Ctrl+C cancels), `ls`, `preset`, `defaults`, `gallery`, `--json`, exit codes, automatic offload/CPU choice only with a visible note, load failures print offers. **Deviation:** `load`/`unload`/`ps` are not in this phase: they only make sense with a long-lived process, so they come with the server in phase 4
- [x] 2.11 Per-model defaults and presets (files, live reload) (S) — `presets/PresetStore` (files re-read on every call = live reload, safe names, validation incl. field paths, import/export) and `ModelDefaultsStore` (per checkpoint file name), `layerRequest/layerLoadOptions` (model defaults < preset < explicit); 8 test cases. CLI `preset` verbs come with 2.10

**Exit:** `graphicshost gen "a lighthouse" --model sd15 --steps 20 --seed 1` writes an image on CPU and on AMD Vulkan; cancel works; the engine integration test passes. *Met on 2026-10-23: the command (`gen "a lighthouse..." -m v1-5 --steps 20 --seed 1`) writes a correct image on `--device cpu` in 18.1 s and on `--device vulkan0` in 5.9 s through the real app; cancel is tested at engine, queue and manager level (Ctrl+C in the CLI is wired but not exercised by a test); the full suite (135 test cases) passes.*

### Phase 3 — Models: discover, download, library (L)

- [x] 3.1 Safetensors header reader and GGUF header reader (port of LLMHost `Gguf.cs`); family detection SD1.5 / SDXL (SD3/FLUX detected and reported unsupported) (M) — `models/CheckpointHeader` (safetensors + GGUF, header only, damaged files refused with a reason), `models/ModelCard` (`detectFamily`: SD1.x/2.x/SDXL, inpainting, VAE, LoRA; FLUX/SD3 detected and reported unsupported; weight-size estimate from tensor types, calibrated on the two reference checkpoints); 9 test cases
- [x] 3.2 `ModelLibrary`: folder watching, `publisher/repo/file` layout, `ModelCard` (family, default size/steps/CFG/sampler, companions) (M) — `models/ModelLibrary`: recursive scan of any layout, header-analysis cache by path/size/mtime, VAE linking (recorded companions or a `<name>…vae…` file next to it), `.source.json` sidecars, `resolve()` by path/name/unique prefix, `ReadDirectoryChangesW`-style watcher with settle delay; CLI `ls`/`gen` use it (family and weights now come from the header); 7 test cases + the CLI tests moved to valid fake checkpoints
- [ ] 3.3 `DownloadManager` on libcurl: queue, range resume, pause/resume, free-space check, SHA-256 (CNG), progress events (L)
- [ ] 3.4 Embedded `catalog.json` (curated SD1.5 / SDXL / Turbo / Lightning entries) and `Catalog` loader (S)
- [ ] 3.5 `HuggingFace` client: search by `text-to-image` tag, file listing, licence and gating info; HF token in Windows Credential Manager; gated-model flow (M)
- [ ] 3.6 Companion resolution: a separate VAE is downloaded and linked automatically with the model (S)
- [ ] 3.7 Fit hints in search results and catalog ("fits this GPU / needs offload / CPU only") (S)
- [ ] 3.8 Model update check (repo revision) (S)
- [ ] 3.9 CLI verbs `catalog`, `search`, `get`, `downloads` (S)

**Exit:** `graphicshost get <catalog key>` downloads, verifies and registers a model, with its VAE if needed; `gen` can use it; downloads resume after being killed.

### Phase 4 — API and both UIs (XL)

- [ ] 4.1 HTTP server (`cpp-httplib`): routing, JSON errors, SSE helper, body-size caps before parsing (M)
- [ ] 4.2 Auth: API tokens (hashed, permissions), localhost-only default, `--lan` flag (M)
- [ ] 4.3 OpenAI endpoints: `/v1/images/generations`, `/v1/images/edits` (stub until img2img), `/v1/models` (M)
- [ ] 4.4 Native `/api/v1`: generate + job poll/stream/cancel, models load/unload, downloads, discover, gallery, outputs (L)
- [ ] 4.5 Request log, `/api/requests`, `/metrics` (S)
- [ ] 4.6 Qt shell: main window, tabs, tray icon, single instance, window-state memory, services → view-model wiring (M)
- [ ] 4.7 Qt **Generate** tab: prompt boxes, parameter panel with interface-mode gating, model/device picker, live progress and preview, results, context menus (L)
- [ ] 4.8 Qt **My Models** and **Discover** tabs, download queue (L)
- [ ] 4.9 Qt **Settings** tab (server, tokens, runtime/backend, GPU device, models folder) and **Logs** tab (M)
- [ ] 4.10 Browser UI: embedded static files served at `/`, with Generate, Gallery, My Models, Discover and Settings against `/api/v1` (L)
- [ ] 4.11 API contract tests (server on an ephemeral port) and UI smoke checks (M)

**Exit:** the same generation can be started from the Qt window, the browser UI, an OpenAI client and the CLI; models can be discovered and downloaded from either UI.

### Phase 5 — Depth: more of what an image host needs (L)

- [ ] 5.1 Qt and browser **Gallery** tab: lazy thumbnails, search by prompt, reuse parameters, delete, export (M)
- [ ] 5.2 img2img (`init_image`, `strength`) and `/v1/images/edits` (M)
- [ ] 5.3 Inpainting with `mask_image`; mask drawing in the Qt UI (browser UI may follow later) (M)
- [ ] 5.4 LoRA support: library scan, `loras[]` parameter, `<lora:name:w>` prompt syntax (M)
- [ ] 5.5 Presets import/export; hardware monitor graphs (GPU utilisation, VRAM) with Qt Charts (M)
- [ ] 5.6 Optional safety filter: `ISafetyFilter`, first implementation, per-request override, off by default (M)
- [ ] 5.7 ESRGAN upscaler *(stretch)*; ControlNet *(stretch)*

**Exit:** the standard parameter set from §5 is fully usable from the API and both UIs, except the stretch items.

### Phase 6 — Polish and release (L)

- [ ] 6.1 First-run wizard: hardware survey, recommended model, models folder (M)
- [ ] 6.2 Themes (light/dark/accent), interface modes (User / Power user / Developer), translation scaffolding (`.ts`) (M)
- [ ] 6.3 HTTPS (OpenSSL, generated or user certificate), IP allow-list, rate limits (M)
- [ ] 6.4 Notifications (download done, generation done), daemon / Windows service mode (M)
- [ ] 6.5 Installer (WiX or MSIX) packaging the exe, vcpkg and Qt DLLs, runtime DLLs and the VC++ Redistributable (M)
- [ ] 6.6 Auto-update (feed, signed package, rollback) *(stretch)*
- [ ] 6.7 Hardware cross-check: RDNA GPU, an older GCN card if available, an APU; a CPU-only machine without AVX2; clean-VM install test *(AMD)* (M)
- [ ] 6.8 Docs: README with feature table, per-feature docs as in LLMHost, API reference (S)

**Exit:** an installer installs on a clean Windows machine and passes the smoke checklist on CPU and AMD Vulkan.

### Critical path and ordering

`0 → 1 → 2 → 3` is strictly sequential (each needs the previous layer). Inside phase 4, `4.1–4.5` (server) come before `4.10`
(browser UI) and can run in parallel with `4.6–4.9` (Qt UI) once the services exist. Phase 5 items are independent of each
other. Phase 6 can start in part once Phase 4 is done (installer, wizard).

### Rough timeline (one engineer)

| Phase | Effort | Cumulative |
|---|---|---|
| 0 Spike | 2–3 wk | 2–3 wk |
| 1 Foundations | 2 wk | 4–5 wk |
| 2 Engine | 2 wk | 6–7 wk |
| 3 Models | 2 wk | 8–9 wk |
| 4 API + UIs | 5–6 wk | 13–15 wk |
| 5 Depth | 2–3 wk | 15–18 wk |
| 6 Polish | 2–3 wk | 17–21 wk |

Phase 0 dominates the uncertainty. If it ends in the prebuilt-Vulkan fallback, subtract roughly one week; if shader
generation proves intractable, CPU-only ships first and Vulkan moves to a fast follow-up.

**Tests** (Catch2, run from Test Explorer; tags `[unit]`, `[api]`, `[integration]`): validators, JSON round-trips, family detection from
header fixtures, catalog parsing, download resume against a local test server, API contract tests (HTTP server started on an
ephemeral port), and an integration test that generates a 256×256 image on CPU with a tiny fixture model (skipped when
absent). An opt-in Vulkan test checks only that an image of the right size returns (outputs differ between backends).
CI builds with `/analyze`, clang-tidy and an ASan configuration.

### Risks specific to C++

- ggml/stable-diffusion.cpp API changes often → pin a commit, wrap it only in `runtime/`, upgrade deliberately.
- **ggml's Vulkan shaders have no MSBuild path upstream.** The Vulkan backend normally generates its shader sources through
  CMake (`vulkan-shaders-gen`). Phase 0 must solve this: build `vulkan-shaders-gen` as its own `.vcxproj`, run it from a
  pre-build step with `glslc`, and add the generated sources to `ggml.vcxproj`. If that proves too fragile, the fallback is to
  use prebuilt Vulkan releases of stable-diffusion.cpp as the `sd-vulkan.dll` runtime, which the DLL design already allows.
  **Phase 0 result:** the MSBuild path works (clean Vulkan build in about 2 minutes, no CMake), so the prebuilt-release fallback is
  not needed so far; this still has to be confirmed on a working AMD GPU (step 0.7).
- Vulkan needs `glslc` on the build machine and an AMD driver at run time; verify on AMD only (other vendors are out of v1 scope).
- Qt through vcpkg compiles from source the first time (hours); enable vcpkg binary caching.
- Native crashes take the whole process down → keep the engine in-process for v1, but design `IImageEngine` so a
  **worker-process** implementation (crash isolation, as in LLMHost plan §2.5) can be added without touching callers.

### Decisions (settled)

| Question | Decision |
|---|---|
| Build system | Visual Studio 2026 solution + `.vcxproj` files. **No CMake.** |
| UI | Qt 6 Widgets desktop UI **and** a browser UI served by the built-in HTTP server |
| Model families, v1 | SD 1.x and SDXL only |
| Safety filter | Optional, off by default |
| Hardware, v1 | CPU and Vulkan on AMD only |
| LLMHost | Nothing shared: own models folder, own Hugging Face token, own settings |

## 11. vcpkg packages

Manifest (`vcpkg.json`) for the stock triplet `x64-windows` (§2): every library is dynamic. Package and feature
names are from memory of the vcpkg registry; run `vcpkg search <name>` once to confirm each before the first install.

| Package (features) | Used for | Installed in the vcpkg checkout (checked 2026-10-22, all present) |
|---|---|---|
| `qtbase` | Qt Core, Gui, Widgets, Network; also provides `moc`, `uic`, `rcc` | ✅ x64-windows |
| `qtcharts` | hardware-monitor graphs | ✅ x64-windows |
| `qtsvg` | SVG icons | ✅ x64-windows |
| `qttools` | Linguist (`lupdate`/`lrelease`), `windeployqt` | ✅ x64-windows |
| `qtimageformats` | WebP / TIFF in the gallery and previews | ✅ x64-windows |
| `cpp-httplib` (`openssl`) | HTTP/HTTPS server, SSE, hosting the browser UI | ✅ x64-windows |
| `openssl` | HTTPS certificates, TLS for the server | ✅ x64-windows |
| `curl` (`ssl`; Schannel on Windows) | resumable downloads, Hugging Face API | ✅ x64-windows |
| `nlohmann-json` | settings, presets, catalog, API payloads | ✅ x64-windows |
| `spdlog` (brings `fmt`) | logging, JSON-lines sink | ✅ x64-windows |
| `cli11` | command-line parsing | ✅ x64-windows |
| `sqlite3` | gallery index and search | ✅ x64-windows |
| `libpng` (brings `zlib`) | PNG with parameter text chunks | ✅ x64-windows |
| `libjpeg-turbo` | JPEG output | ✅ x64-windows |
| `libwebp` | WebP output | ✅ x64-windows |
| `catch2` | tests | ✅ x64-windows |
| `vulkan-headers`, `vulkan-loader` | Vulkan build headers and loader for the ggml Vulkan backend and `GpuDevices` | ✅ x64-windows |
| `shaderc` | provides `glslc` to compile ggml's Vulkan shaders (not needed if the Vulkan SDK is installed) | ✅ x64-windows |

Not from vcpkg (install separately):

- **Visual Studio 2026** with the *Desktop development with C++* workload (MSVC, Windows SDK, MSBuild).
- **Qt Visual Studio Tools** extension (runs `moc`/`uic`/`rcc` inside MSBuild). Point it at the vcpkg install of the
  `x64-windows` triplet; `windeployqt` copies the Qt plugins next to the exe.
- **Vulkan SDK** (LunarG): the alternative to `shaderc`, and includes validation layers useful in Phase 0.
- **AMD Adrenalin driver** on the test machines (runtime Vulkan driver).
- **stable-diffusion.cpp** and its **ggml** submodule (git submodules, pinned).
- **Git**, and later **WiX Toolset** for the installer (Phase 6).
- SHA-256 uses Windows CNG (`bcrypt`), and the browser UI is plain HTML/CSS/JS, so neither needs a package.

### Sources

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — the engine (CPU and Vulkan backends via ggml)
- LLMHost, `D:\code\LLMHost` — the reference design
