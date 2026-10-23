# GraphicsHost

A local host for AI image-generation models (Stable Diffusion 1.x and SDXL in v1): discovers and downloads models,
runs them in-process on **CPU or Vulkan (AMD GPUs)**, exposes one standard set of generation parameters, and serves
them through a Qt desktop window, a browser UI, a CLI and an HTTP API. Modelled on LLMHost; the engine is
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

Status: **Phase 1 (foundations)**. Done: the engine builds without CMake and generates on CPU, the runtime DLLs and their
C ABI, settings and logging, hardware detection, the runtime loader, a CLI skeleton and an empty Qt window. Image
generation from the app, model management and the HTTP API come in phases 2 to 4. See
[implementation-plan.md](implementation-plan.md) for the plan and what is ticked off.

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
| `runtimes\sd-cpu-avx2.dll`, `sd-cpu-noavx.dll`, `sd-vulkan.dll` | the inference runtimes (one is loaded at start-up) |
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

Data lives in `%USERPROFILE%\.graphicshost` (`GRAPHICSHOST_HOME` overrides it): `settings.json`, `logs\`, `runtimes\` for
user-installed runtimes, `presets\`, `outputs\`, `models\`.

Exit codes: 0 ok, 1 failure, 2 usage, 3 not found, 4 not authorised, 5 server unreachable, 6 out of memory, 7 cancelled,
8 conflict.

## Test

```
build\bin\x64\Release\GraphicsHost.Tests.exe "[unit]"          # pure logic, settings, CLI
build\bin\x64\Release\GraphicsHost.Tests.exe "[integration]"   # needs a built runtime DLL next to the exe
```

## Layout

| Folder | What |
|---|---|
| `GraphicsHost/` | the application: `main.cpp`, Qt UI (`ui/`) |
| `GraphicsHost.Core/` | everything that is not UI: `core/` settings, logging, paths; `engine/` hardware and runtime loader; `host/` the composition root `AppHost`; `cli/` |
| `GraphicsHost.Tests/` | Catch2 tests |
| `runtime/` | the C ABI (`gh_runtime.h`) and the three runtime DLL projects |
| `ggml/`, `stable-diffusion/` | MSBuild projects that build the engine from the submodule in `extern/` |
| `props/` | shared MSBuild settings, including the Qt moc/deploy integration |
| `docs/` | [adding an option](docs/adding-options.md), [Phase 0 results](docs/phase0-results.md), [environment notes](docs/environment-notes.md) |
| `spike/` | Phase 0 spike program |

## Known environment issues

* Windows Application Control can refuse to load freshly built unsigned binaries; see
  [docs/environment-notes.md](docs/environment-notes.md).
* GPU runs are untested until the development machine's AMD driver is repaired (Phase 0 results).
