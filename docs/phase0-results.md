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
