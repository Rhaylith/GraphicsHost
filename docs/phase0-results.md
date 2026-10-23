# Phase 0 results

Machine: AMD Ryzen Threadripper PRO 7995WX (96 cores), AMD Radeon RX 7900 XTX, Windows 11.
Engine: stable-diffusion.cpp `fd21ecf` (master-995), ggml `c1f80e5`, built by `ggml.vcxproj` + `stable-diffusion.vcxproj`
(MSVC v145, Release, CPU variant AVX2). Driver program: `spike/spike.exe`.

Prompt: "a lighthouse on a cliff at sunset, detailed, oil painting", seed 1, euler_a default sampler, CFG 7.

| Step | Model | Backend | Size | Steps | Load | Generate | s/step | Notes |
|---|---|---|---|---|---|---|---|---|
| 0.5 | SD1.5 (v1-5-pruned-emaonly) | cpu (AVX2) | 512x512 | 20 | 0.12 s | 18.17 s | 0.73 (sampling) | image checked by eye; VAE decode 1.95 s |

## Step 0.6: Vulkan backend without CMake

`msbuild spike\spike.vcxproj /p:Configuration=Release /p:Platform=x64 /p:GgmlVulkan=true` builds the Vulkan variant:

1. `ggml\vulkan-shaders-gen.vcxproj` builds the shader generator (host tool, always Release, `build\tools\`).
2. `ggml\ggml.vcxproj` (with `GgmlVulkan=true`) runs it before compiling: once for `ggml-vulkan-shaders.hpp`, then once per
   `.comp` file with `glslc` (vcpkg `shaderc` 2026.2; no Vulkan SDK needed). The ~190 generated `*.comp.cpp` files are
   incremental and compiled into `ggml-vulkan.lib`.
3. The glslc feature tests CMake runs at configure time are done by `tools\probe-glslc.ps1`; all seven extensions are
   supported, and the matching `GGML_VULKAN_*_GLSLC_SUPPORT` defines are set in the two vcxproj files.

Clean Vulkan build (all shaders): about 1 min 50 s on the 96-core machine. Output: `ggml-vulkan.lib` (65 MB) and
`spike-vulkan.exe` (links `vulkan-1.lib`; uses the driver's `vulkan-1.dll` from System32).
`/p:GgmlVulkanDebug=true` enables ggml's Vulkan debug log (instance errors are otherwise swallowed).

**Runtime check on this machine: failed, not because of the build.** `spike-vulkan --list-devices` shows only the CPU.
With the debug log: `vk::createInstance: ErrorIncompatibleDriver`, with both the vcpkg and the System32 loader.
Windows reports the AMD Radeon RX 7900 XTX (driver 32.0.21030.2001) in error state
`CM_PROB_FAILED_ADD` (Device Manager code 31: "Windows cannot load the drivers required for this device"), so no Vulkan
ICD can initialise. The session also runs on a Remote Display Adapter (RDP). The GPU needs a driver reinstall or reboot
before step 0.7 can measure Vulkan.

## Step 0.7 (partial): SDXL on CPU; GPU runs blocked

| Model | Backend | Size | Steps | Load | Generate | s/step | Notes |
|---|---|---|---|---|---|---|---|
| SD1.5 | cpu (AVX2) | 512x512 | 20 | 0.12 s | 18.17 s | 0.73 | from step 0.5 |
| SDXL base 1.0 (fp16 safetensors) | cpu (AVX2) | 1024x1024 | 20 | 0.11 s | 102.95 s | 4.51 (sampling) | VAE decode 6.8 s; image checked by eye |
| SD1.5 / SDXL | vulkan (RX 7900 XTX) | - | - | - | - | - | **not measured**: Vulkan cannot initialise, GPU in Device Manager code 31 |

Memory/VRAM figures for the GPU runs are still missing for the same reason.

Phase 0 closed as CPU-verified (user decision): GPU measurements are skipped for now. Open item: run `spike-vulkan` on a working RX 7900 XTX and fill in the table above.

**Update 2026-10-23:** the RX 7900 XTX now initialises in Vulkan (`runtime info` lists `Vulkan0 AMD Radeon RX 7900 XTX`), so the GPU rows above can be measured with `spike-vulkan.exe`; not done yet.

## Step 0.7 completed: GPU measurements (2026-10-23)

Driver repaired (the card initialises in Vulkan: `ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (AMD proprietary driver) | fp16: 1 | bf16: 1 | warp size: 64 | matrix cores: KHR_coopmat`, 24 GiB).
`spike-vulkan.exe --backend vulkan0`, same prompt, seed 1, 20 steps, CFG 7 as the CPU runs.

| Model | Backend | Size | Load | Generate | Sampling | VAE decode | s/step (total) | Weights in VRAM |
|---|---|---|---|---|---|---|---|---|
| SD1.5 | cpu (AVX2) | 512x512 | 0.12 s | 18.17 s | 14.5 s | 1.95 s | 0.909 | - |
| SD1.5 | **vulkan0** | 512x512 | 1.22 s | **6.94 s** | 5.64 s | 0.73 s | 0.347 | 2 784 MB |
| SDXL base 1.0 | cpu (AVX2) | 1024x1024 | 0.11 s | 102.95 s | 94.8 s | 6.8 s | 5.148 | - |
| SDXL base 1.0 | **vulkan0** | 1024x1024 | 0.21 s | **25.57 s** | 18.25 s | 5.40 s | 1.279 | 6 624 MB |

* Speed-up on the 7900 XTX over the 96-core CPU: SD1.5 2.6x, SDXL 4.0x. The CPU is strong here (AVX2 on 192 threads), so
  the ratio is smaller than on a typical desktop.
* Both GPU images were checked by eye and match the CPU images (same composition for the same seed).
* The first SD1.5 run on the GPU takes longer to load (1.2 s) than later runs: shader pipelines are built on first use.
* Peak VRAM including activations was not measured; only the weight sizes the engine prints. Both models fit in 24 GiB
  with a lot of room; the estimator in step 2.8 needs a measured activation figure (see the TODO there).
* ggml printed `Failed to allocate pinned memory` twice during the SDXL VAE decode and carried on (the result is correct);
  it means a host-visible staging buffer was refused and a slower path was used. Worth watching in later phases.
* `conv3d direct auto: Vulkan0 -> off (probe output check failed)`: the engine's own probe disables one optimisation on
  this GPU/driver. Harmless for image models.
