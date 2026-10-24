# GraphicsHost

A local host for AI image-generation models (Stable Diffusion 1.x and SDXL in v1): discovers and downloads models,
runs them in-process on **CPU or Vulkan (AMD GPUs)**, exposes one standard set of generation parameters, and serves
them through a Qt desktop window, a browser UI, a CLI and an HTTP API. Modelled on LLMHost; the engine is
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

Status: **Phases 2 to 6 are done except the stretch items (upscaler, ControlNet, auto-update), an MSI installer and testing on other hardware**: images are generated from the Qt window, the
browser, the command line and the HTTP API (OpenAI-compatible and native) on CPU and on the AMD GPU (Vulkan), for
SD 1.x and SDXL, with all samplers and schedulers, a job queue with cancellation, PNG/JPEG/WebP output with the
parameters embedded, a gallery, presets, per-model defaults and a memory estimate, plus a model library, a verified catalog,
Hugging Face search and resumable downloads, API tokens and a request log, image to image, inpainting, LoRA, a hardware monitor and an
optional prompt safety filter. See [implementation-plan.md](implementation-plan.md) for the plan and what is ticked off, and
[docs/api.md](docs/api.md) for the HTTP API.

## What it does

| Area | Features |
|---|---|
| Generation | SD 1.x and SDXL on CPU (AVX2 / plain) or Vulkan (AMD); samplers and schedulers; batches; seeds; image to image; inpainting with a painted mask; LoRA; PNG/JPEG/WebP with the parameters embedded |
| Models | library of the models folder; curated catalog with a fit hint for this computer; Hugging Face search; resumable, checksummed downloads; update check; per-model defaults; presets (import/export) |
| Interfaces | Qt window (Generate, Gallery, My Models, Discover, Settings, Monitor, Logs), browser UI, CLI, OpenAI-compatible and native HTTP API ([docs/api.md](docs/api.md)) |
| Server | access tokens with permissions, HTTPS (own or generated certificate), address allow-list, rate limit, request log, `/metrics`, Windows service mode |
| Comfort | first-run wizard, light/dark themes with an accent colour, tray icon and notifications, hardware monitor, translation scaffolding |
| Safety | optional prompt word filter, off by default ([docs/api.md](docs/api.md#safety-filter)) |

Packaging: `msbuild installer\Package.proj` makes `build\dist\GraphicsHost-<version>-win-x64.zip` (a portable package, not an
installer; see [implementation-plan.md](implementation-plan.md) step 6.5). Hardware coverage and what is untested:
[docs/hardware-checklist.md](docs/hardware-checklist.md).

## Build

Requirements (all on Windows, x64):

* Visual Studio 2026 (18.x) with the **Desktop development with C++** workload. Everything builds with MSBuild
  (`.slnx` + `.vcxproj`), there is no CMake.
* [vcpkg](https://github.com/microsoft/vcpkg) with MSBuild integration (`vcpkg integrate install`) and the packages of
  [vcpkg.json](vcpkg.json) installed **as the `x64-windows` triplet** (dynamic): classic mode, e.g.

  ```
  vcpkg install --triplet x64-windows qtbase qtcharts qtsvg qttools qtimageformats cpp-httplib[openssl] curl[ssl] nlohmann-json spdlog cli11 sqlite3 libpng libjpeg-turbo libwebp catch2 vulkan-headers vulkan-loader shaderc
  ```

  The build looks for vcpkg in `D:\code\vcpkg\installed\x64-windows` by default; set `VCPKG_INSTALLED` to another
  `installed\x64-windows` folder to change that (see [Directory.Build.props](Directory.Build.props)).
* Git, with submodules: `git submodule update --init --recursive`.

```
msbuild GraphicsHost.slnx -p:Configuration=Release -p:Platform=x64 -m
```

Output goes to `build\bin\x64\Release`:

| File | What |
|---|---|
| `GraphicsHost.exe` | the application: window with no arguments, command line with arguments |
| `runtimes\sd-cpu-avx2.dll`, `sd-vulkan.dll` | the inference runtimes (one is loaded at start-up) |
| `GraphicsHost.Tests.exe` | the tests |
| `spike.exe`, `spike-vulkan.exe` | Phase 0 spike programs (removed in phase 2) |

The first full build compiles ggml and stable-diffusion.cpp (several minutes); the Vulkan runtime also generates about 190
shader files with `glslc` from vcpkg's `shaderc` (about 2 minutes). Later builds are incremental.

## Run

```
GraphicsHost.exe                       # desktop window
GraphicsHost.exe hardware              # CPU, GPUs, runtimes, which runtime the backend setting selects
GraphicsHost.exe --json hardware       # same as JSON
GraphicsHost.exe runtime info          # load the selected runtime and show what it reports
GraphicsHost.exe config show           # settings (config set <key> <value> changes one)
GraphicsHost.exe --home D:\tmp\gh ...  # use another data folder
```

Generating images (checkpoints are looked for in the models folder, `config set modelsDir <folder>`; a path also works):

```
GraphicsHost.exe ls                                          # checkpoints found
GraphicsHost.exe gen "a lighthouse at sunset" -m v1-5        # name, file name or unique prefix; result goes to the gallery
GraphicsHost.exe gen "a red fox" -m sdxl --size 1024x768 --steps 30 --sampler dpm++2m --scheduler karras -b 4 -o D:\out
GraphicsHost.exe gen "..." -m v1-5 --device cpu              # auto (default) | cpu | vulkan0
GraphicsHost.exe gen "..." -m v1-5 --size 768x768 --flash-attn   # SD 1.x above 512 is about 5x faster with it on the GPU
GraphicsHost.exe gen "..." -m sdxl --dry-run                 # resolved parameters and memory estimate, nothing is generated
GraphicsHost.exe preset set portrait width 512 height 768 steps 30 load.flashAttention true
GraphicsHost.exe gen "..." -m v1-5 --preset portrait --steps 40   # model defaults < preset < flags
GraphicsHost.exe defaults set v1-5-pruned-emaonly.safetensors sampler euler_a
GraphicsHost.exe gallery list lighthouse                     # also: show ID, rm ID, rescan
```

Finding and downloading models:

```
GraphicsHost.exe catalog                       # the curated list, with "fits the GPU / needs offload / CPU only" for this computer
GraphicsHost.exe get sdxl                      # downloads SDXL base + its VAE, checks the SHA-256, records where it came from
GraphicsHost.exe get sdxl --dry-run            # what would be downloaded, where, and how it fits
GraphicsHost.exe search "dreamshaper"          # live search on Hugging Face (text-to-image models)
GraphicsHost.exe info Lykon/DreamShaper        # the usable files of a repository, with sizes
GraphicsHost.exe get Lykon/DreamShaper DreamShaper_8_pruned.safetensors
GraphicsHost.exe login                         # a token for gated models; kept in the Windows Credential Manager
GraphicsHost.exe updates                       # have the downloaded models changed on Hugging Face?
GraphicsHost.exe downloads                     # partial files of interrupted downloads (run `get` again to continue)
```

Downloads resume where they stopped (also after killing the program or a crash) and a checksum mismatch deletes the
file. Models land in `<models folder>\<publisher>\<repository>\<file>`; a VAE that belongs to a checkpoint is linked
through the record next to it (`<file>.source.json`), or by naming it `<checkpoint name>...vae....safetensors` in the same
folder. `<data folder>\catalog.json` adds to or replaces entries of the shipped `data\catalog.json`.

`gen` prints progress on stderr and the result on stdout (`--json` for scripts). Ctrl+C cancels at the next sampling
step. The seed that was actually used is always reported; the same prompt, parameters and seed give the same image on
the same runtime. A model that does not fit the GPU is run with its weights in RAM (`--offload`) or on the CPU when the
device is `auto`, with a note; an explicit `--device vulkan0` is never moved, and a failed load prints what you could
try. The engine's own start-up lines (`ggml_vulkan: ...`) also appear on stderr.

Data lives in `%USERPROFILE%\.graphicshost` (`GRAPHICSHOST_HOME` overrides it): `settings.json`, `logs\`, `runtimes\` for
user-installed runtimes, `presets\`, `outputs\`. Downloaded models go to `DownloadedModels` in the working directory the program
was started from (change it with the `modelsDir` setting); with `GRAPHICSHOST_HOME` set they stay in `<home>\models` instead.

Exit codes: 0 ok, 1 failure, 2 usage, 3 not found, 4 not authorised, 5 server unreachable, 6 out of memory, 7 cancelled,
8 conflict.

The server and the UIs:

```
GraphicsHost.exe                               # the window; it also runs the API server (settings server.enabled / server.port)
GraphicsHost.exe serve [--port 1234]           # the same server without a window, until Ctrl+C
GraphicsHost.exe token create "my phone" --perm read,generate   # an access token (shown once)
curl http://127.0.0.1:1234/v1/images/generations -H "Content-Type: application/json" -d "{\"model\":\"v1-5\",\"prompt\":\"a lighthouse\"}"
```

Open `http://127.0.0.1:1234/` for the browser UI (the `web` folder next to the exe). Closing the window leaves GraphicsHost
in the tray while the server runs; Quit is in the tray menu. Without a token the server answers only programs on this
computer; letting other computers connect (`server.lan`) is refused until a token exists. The OpenAI-style routes and
the native `/api/v1` routes are listed in [docs/api.md](docs/api.md).

## Test

```
build\bin\x64\Release\GraphicsHost.Tests.exe "[unit]"          # pure logic, settings, CLI, queue, gallery, codecs
build\bin\x64\Release\GraphicsHost.Tests.exe "[integration]"   # needs a built runtime DLL next to the exe
build\bin\x64\Release\GraphicsHost.Tests.exe "[model]"         # real generation; needs checkpoints in models\ (see below)
build\bin\x64\Release\GraphicsHost.Tests.exe "[api]"           # the HTTP API against a fake engine (OpenAI, native, auth, metrics, contract)
build\bin\x64\Release\GraphicsHost.Tests.exe "[ui]"            # starts GraphicsHost.exe offscreen and checks that every tab renders
build\bin\x64\Release\GraphicsHost.Tests.exe "~[model]~[network]"   # everything that needs neither a checkpoint nor the internet (about 230 cases)
build\bin\x64\Release\GraphicsHost.Tests.exe                   # everything, including the real-model and network tests
```

The `[model]` tests look for `models\v1-5-pruned-emaonly.safetensors` (SD 1.5) and `models\sd_xl_base_1.0.safetensors`
(SDXL) in the repository (or the file in `GRAPHICSHOST_TEST_MODEL`) and skip themselves when they are missing. They run
on the CPU runtime, so no GPU is needed. `tools\measure-vram.ps1` measures GPU memory for the estimator
([docs/vram-measurements.md](docs/vram-measurements.md)).

## Layout

| Folder | What |
|---|---|
| `GraphicsHost/` | the application: `main.cpp`, Qt UI (`ui/`: Generate, My Models, Discover, Settings, Logs tabs) |
| `web/` | the browser UI (plain HTML/JS/CSS, copied next to the exe) |
| `GraphicsHost.Core/` | everything that is not UI: `core/` settings, logging, paths; `engine/` hardware, runtime loader, `ImageEngine`, `JobQueue`, `ModelManager`, image codecs, fit estimator; `gallery/`; `presets/`; `host/` the composition root `AppHost` and `GenerationService`; `models/` library, catalog, downloads; `server/` the HTTP server, auth, routes; `cli/` |
| `GraphicsHost.Tests/` | Catch2 tests |
| `runtime/` | the C ABI (`gh_runtime.h`) and the three runtime DLL projects |
| `ggml/`, `stable-diffusion/` | MSBuild projects that build the engine from the submodule in `extern/` |
| `props/` | shared MSBuild settings, including the Qt moc/deploy integration |
| `docs/` | [adding an option](docs/adding-options.md), [Phase 0 results](docs/phase0-results.md), [environment notes](docs/environment-notes.md), [HTTP API](docs/api.md), [hardware checklist](docs/hardware-checklist.md) |
| `spike/` | Phase 0 spike program |

## Known environment issues

* Windows Application Control can refuse to load freshly built unsigned binaries; see
  [docs/environment-notes.md](docs/environment-notes.md).
* The offscreen UI checks (`tools/ui-shots.ps1`, the `[ui]` test) need no display; the tray icon and the single-instance behaviour need a real desktop and are not covered by automated checks.
