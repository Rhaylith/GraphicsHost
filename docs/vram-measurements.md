# GPU memory measurements (2026-10-23)

Source of the constants in `GraphicsHost.Core/engine/FitEstimator.cpp`. Measured with `tools/measure-vram.ps1`:
peak "GPU Adapter Memory / Dedicated Usage" of the whole machine during one `spike-vulkan.exe` run, minus the idle level
before it (about 1.24 GiB, the desktop). Radeon RX 7900 XTX, 24 GiB, Vulkan runtime, Release build, 4 to 8 steps,
fp16 weights as converted by the engine.

**Units:** the counter reports MiB. The engine prints its weight sizes in decimal MB (`total params memory size =
2784.45MB`), which are 2655 MiB for SD1.5 and 6317 MiB for SDXL. "Activations" below is growth (MiB) minus the weights
converted to MiB, i.e. compute buffers, VAE and driver overhead.

| Model | Size | Mode | Growth over idle | Weights | Activations | s/step |
|---|---|---|---|---|---|---|
| SD1.5 | 512x512 | normal | 4 893 MiB | 2 655 MiB | 2 238 MiB | 0.35 |
| SD1.5 | 768x768 | normal | 5 872 MiB | 2 655 MiB | 3 217 MiB | 4.41 |
| SD1.5 | 768x768 | flash attention | 5 989 MiB | 2 655 MiB | 3 334 MiB | 0.89 |
| SD1.5 | 1024x1024 | normal | 5 345 MiB | 2 655 MiB | 2 690 MiB | (very slow: 12.5) |
| SD1.5 | 512x512 | weights offloaded to RAM | 4 422 MiB | 0 in VRAM | 4 422 MiB | 0.76 |
| SDXL | 768x768 | normal | 8 842 MiB | 6 317 MiB | 2 525 MiB | - |
| SDXL | 1024x1024 | normal | 9 952 MiB | 6 317 MiB | 3 635 MiB | 2.89 |
| SDXL | 1024x1024 | flash attention | 9 972 MiB | 6 317 MiB | 3 655 MiB | 2.85 |
| SDXL | 1024x1024 | weights offloaded to RAM | 6 829 MiB | 0 in VRAM | 6 829 MiB | 3.04 |

## What the numbers say

* Activations grow roughly linearly with the pixel count (two points per family, so a line, not a proven law):
  about 1450 MiB + 3130 MiB per megapixel for SD 1.x and 1100 MiB + 2540 MiB per megapixel for SDXL. SD1.5 at
  1024x1024 is *lower* than the line predicts because the engine switches to segmented compute when the attention
  buffers get too big; the estimator keeps the line (conservative) and stops growing it at 1.3 MP.
* Offloading the weights to system RAM (`offloadParamsToCpu`) keeps the diffusion model's share of the weights in VRAM
  (82 % for SD 1.x, 51 % for SDXL) and costs almost no speed here (SDXL 3.04 vs 2.89 s/step).
* **Flash attention matters for SD 1.x above 512x512**: 768x768 takes 0.89 s/step with it and 4.41 s/step without
  (5x), and 1024x1024 is far worse without it. SDXL is not affected. The estimator suggests it; making it the default for
  SD 1.x on Vulkan is a candidate for a later phase (check image quality first).
* The engine prints `Failed to allocate pinned memory` for the SDXL VAE decode; harmless, see `phase0-results.md`.
* File size is not the weight size: the SD1.5 checkpoint is 4.27 GB on disk but 2.78 GB in VRAM (stored as fp32,
  loaded as fp16: ratio 0.653); the SDXL one is 6.94 GB on disk and 6.62 GB in VRAM (ratio 0.955). The estimator uses
  these ratios until phase 3 reads the real tensor types from the file header.
* One GPU, one driver, two checkpoints: treat the constants as a first calibration. Other cards (smaller caches, other
  drivers) may differ; rerun `tools/measure-vram.ps1` there.
