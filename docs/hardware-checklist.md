# Hardware cross-check (step 6.7)

What has been run, and what still needs another computer. Run the checklist on each machine and add a row.

## Verified so far

| Machine | CPU runtime | Vulkan runtime | Notes |
|---|---|---|---|
| Ryzen Threadripper PRO 7995WX, 512 GB RAM, Radeon RX 7900 XTX (RDNA 3, 24 GB), Windows 11 | `cpu-avx2`: all `[model]` tests (real SD 1.5 generations, image to image, inpainting, LoRA) | `sd-vulkan`: SD 1.5 and SDXL, 128 to 1024 px, VRAM measurements in [vram-measurements.md](vram-measurements.md); the packaged build was run from another folder with a bare PATH | the only machine so far |

## Not verified (needs hardware this project does not have)

* An older AMD card (GCN / Polaris / Vega) and an RDNA 1 or 2 card: does `sd-vulkan` start, does the fit estimator's
  VRAM arithmetic hold, does flash attention work?
* An APU (shared memory): the estimator treats an integrated GPU's budget as a share of free RAM, which is untested.
* A clean Windows machine (or VM) without Visual Studio, vcpkg or the Vulkan SDK: the package carries the VC++ runtime and
  was run with a bare PATH here, but a real clean install is the only proof.
* The tray icon, single instance, the first-run wizard's Finish and download path, and the Windows service
  (`service install`) on a real desktop.

## Checklist for a new machine

1. Unzip the package, run `GraphicsHost.exe hardware`: the CPU features, the GPU and the selected runtime look right.
2. `GraphicsHost.exe` shows the first-run wizard; Finish downloads the suggested model.
3. Generate 512x512 with SD 1.5 on the selected runtime; then with `--device cpu` (Settings: Backend CPU).
4. SDXL at 1024x1024 if the memory suggests it fits; otherwise check that the "needs offload / CPU only" hint is right.
5. `GraphicsHost.exe serve` and the browser UI at `http://127.0.0.1:1234/`: generate an image, open the Gallery.
6. Closing the window with the server on leaves the tray icon; Quit from it ends the process.
7. `GraphicsHost.Tests.exe "~[model]~[network]"` if a build tree is available.
