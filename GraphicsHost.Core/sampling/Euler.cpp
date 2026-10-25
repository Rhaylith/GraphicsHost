#include "sampling/Euler.h"

#include <cstddef>

namespace gh::sampling {

bool euler(Latent& x, const std::vector<float>& sigmas, const DenoiseFn& denoise, const StepFn& onStep) {
    if (sigmas.size() < 2) return true;
    const int steps = static_cast<int>(sigmas.size()) - 1;
    Latent denoised;
    for (int i = 0; i < steps; ++i) {
        const float sigma = sigmas[i];
        if (sigma == sigmas[i + 1]) continue;
        if (!denoise(x, sigma, i + 1, denoised) || denoised.size() != x.size()) return false;
        const float dt = sigmas[i + 1] - sigma;

        const size_t n = x.size();
        float* __restrict px = x.data();
        const float* __restrict pd = denoised.data();
        for (size_t k = 0; k < n; ++k) px[k] += ((px[k] - pd[k]) / sigma) * dt;   // same operation order as the engine's own Euler
        if (onStep) onStep(i + 1, steps);
    }
    return true;
}

}  // namespace gh::sampling
