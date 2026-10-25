# W0 results: the pipeline in pieces

Measured on the development machine (Radeon RX 7900 XTX, `sd-vulkan.dll`, engine commit `fd21ecf`). The tests that produce
these numbers are `[model][vulkan][components]` in `GraphicsHost.Tests/ComponentTests.cpp`.

| Question | Result |
|---|---|
| Can the engine's text encoder, UNet and VAE be called separately from our runtime DLL? | Yes. `runtime/gh_components.cpp` includes the engine's internal headers (engine include paths, C++17 for that file only) and drives `StableDiffusionGGML` directly. No change to the submodule. |
| Does a host-side Euler loop make the same image as `gh_generate`? | SD 1.5, 512², 6 steps, euler/discrete, cfg 7.5, seed 42: mean difference **0.23 of 255**. SDXL base, 1024², same settings: **0.25 of 255**. `gh_generate` against itself: < 0.05. A different seed: > 10. |
| What did parity need? | (1) The prompt and the negative prompt as **one batch of two** in the UNet call, as the engine does (sequential calls: 1.9). (2) For SDXL an **empty negative prompt encoded as zeros** (without it: 7.6). Both are now part of the component functions / the host rules. |
| What does the host loop cost? | SD 1.5, 512², flash attention on: engine loop 0.0915 s/step, host loop 0.0920 s/step, **0.5 %** (budget 3 %). The per-step host work (copying a 64×64×4 latent in and out, the Euler update) is a fraction of a millisecond; it does not grow with the model's step time, so SDXL (about 1 s/step and more) sits far below the budget (estimated, not separately measured). |

## Decision (W0.6): go

Sampling stays in the host (`GraphicsHost.Core/sampling`); only the model evaluation crosses the ABI. The fallback
(runtime-side samplers behind a sampler-id ABI) is not needed.

## Things the later steps must remember

* The engine has an "execution scope" it opens around every call (it refreshes the model sources and releases the
  runners at the end). `gh_session_begin/end` keeps one scope open across many calls; `gh_generate` cannot run while a
  session is open (it says the engine is busy). The graph executor must hold a session per job and end it before anything
  else calls `gh_generate`.
* `gh_text_encode` still tokenises inside the engine. Host-side tokenisers and prompt weighting are W4.1.
* The component functions work on an already loaded model (`gh_load`). Opening checkpoints, text encoders and VAEs as
  separate components from separate files is W3.
* The initial noise is drawn by the engine's RNG for the seed (`gh_initial_latent`); the host RNGs are W4.3.
