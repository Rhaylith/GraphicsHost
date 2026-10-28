#pragma once

#include "sampling/Euler.h"

#include <functional>
#include <string>
#include <vector>

namespace gh::sampling {

// The samplers (workflow plan, section 6.4): host-side loops over a latent that call the guided model for the denoised
// estimate. The first-order ones, the DPM-Solver family, linear multistep, Adams-Bashforth, RES and the ancestral and SDE
// variants are written from the papers. `sigmas` has steps + 1 entries and ends at 0.

// Standard normal noise for step `step` (1-based) going from `sigma` to `sigmaNext`; the sampler scales it.
using NoiseFn = std::function<void(float* out, size_t count, int step, float sigma, float sigmaNext)>;
// Brownian noise for an SDE sampler: the increment of the motion between two noise levels, normalised to unit variance.
using BrownianFn = std::function<void(float* out, size_t count, float sigma, float sigmaNext)>;

struct SamplerEnv {
    DenoiseFn denoise;
    StepFn onStep;
    NoiseFn noise;           // sequential noise for the ancestral and SDE samplers (missing: they run without noise)
    BrownianFn brownian;     // for the SDE samplers; missing: `noise` is used
    // The CFG++ samplers read the unguided estimate of the call that just returned; the guider fills it.
    const Latent* uncond = nullptr;
};

struct SamplerOptions {
    double eta = 1.0;         // amount of fresh noise (ancestral and SDE samplers)
    double sNoise = 1.0;      // multiplier on that noise
    std::string solver = "midpoint";   // dpm++ 2M SDE: midpoint or heun
    double r = 0.5;           // dpm++ SDE: position of the intermediate step
    int order = 4;            // lms
};

// The sampler ids the nodes offer, in the order of the reference (tooltip table O.sampler).
const std::vector<std::string>& allSamplerNames();
// The ones this program can run; the rest are listed so graphs that name them load, and fail with a clear message when run.
bool samplerAvailable(const std::string& name);
bool samplerIsStochastic(const std::string& name);

// Runs a sampler over x (updated in place). Returns false when the model call asked to stop or the name is not available.
bool runSampler(const std::string& name, Latent& x, const std::vector<float>& sigmas, SamplerEnv& env, const SamplerOptions& options = {});

}  // namespace gh::sampling
