# Phase 0 results

Machine: AMD Ryzen Threadripper PRO 7995WX (96 cores), AMD Radeon RX 7900 XTX, Windows 11.
Engine: stable-diffusion.cpp `fd21ecf` (master-995), ggml `c1f80e5`, built by `ggml.vcxproj` + `stable-diffusion.vcxproj`
(MSVC v145, Release, CPU variant AVX2). Driver program: `spike/spike.exe`.

Prompt: "a lighthouse on a cliff at sunset, detailed, oil painting", seed 1, euler_a default sampler, CFG 7.

| Step | Model | Backend | Size | Steps | Load | Generate | s/step | Notes |
|---|---|---|---|---|---|---|---|---|
| 0.5 | SD1.5 (v1-5-pruned-emaonly) | cpu (AVX2) | 512x512 | 20 | 0.12 s | 18.17 s | 0.73 (sampling) | image checked by eye; VAE decode 1.95 s |
