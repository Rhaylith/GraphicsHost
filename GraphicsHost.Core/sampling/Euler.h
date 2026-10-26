#pragma once

#include <functional>
#include <vector>

namespace gh::sampling {

// A latent as a flat float array (the layout the model uses; the sampler does not care about the shape).
using Latent = std::vector<float>;

// The model seen from the sampler: the denoised estimate of x at noise level sigma. `step` counts from 1. Return false to
// stop (cancelled, or the model failed); the sampler then returns false as well.
using DenoiseFn = std::function<bool(const Latent& x, float sigma, int step, Latent& denoised)>;
// Called after each finished step with (step, steps).
using StepFn = std::function<void(int step, int steps)>;

// Euler's method on the probability-flow ODE (Karras et al. 2022, algorithm 2 without the stochastic part):
//   d = (x - denoised) / sigma,  x += d * (sigma_next - sigma)
// sigmas has steps + 1 entries and ends at 0. x is updated in place. Equal neighbouring sigmas are skipped. The
// arithmetic is plain loops over floats, which the AVX2 build vectorises.
bool euler(Latent& x, const std::vector<float>& sigmas, const DenoiseFn& denoise, const StepFn& onStep = {});

}  // namespace gh::sampling
