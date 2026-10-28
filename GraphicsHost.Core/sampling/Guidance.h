#pragma once

#include <cstddef>
#include <vector>

namespace gh::sampling {

// How the predictions of a model for the positive and the negative prompt are combined into the one the sampler uses (workflow
// plan, section 6.4). All functions work on denoised estimates (x0), batch item by batch item: `perSample` values each.

// Optional refinements of classifier-free guidance, set by the model patch nodes.
struct GuidanceTweaks {
    double rescale = 0.0;          // Rescale CFG: 0 = off, else the multiplier (Lin et al. 2023)
    bool zeroStar = false;         // CFG-Zero*: optimised scale of the negative prediction (Fan et al. 2025)
    int zeroInitSteps = 0;         // CFG-Zero*: the first steps return the input unchanged (a zero noise prediction)
    bool apg = false;              // adaptive projected guidance (Sadat et al. 2024)
    double apgEta = 1.0;           // weight of the part of the update parallel to the positive prediction
    double apgNormThreshold = 5.0; // cap on the size of the update (0 = none)
    double apgMomentum = 0.0;      // average of the update over steps (negative sharpens)
    bool operator==(const GuidanceTweaks&) const = default;
};

// What the combination remembers between steps of one run.
struct GuidanceState {
    int step = 0;
    std::vector<float> apgAverage;
};

// Classifier-free guidance: uncond + scale * (cond - uncond). `out` may alias either input.
void cfgCombine(const float* cond, const float* uncond, double scale, float* out, size_t count);

// CFG with the tweaks applied. `x` is the noisy input the model saw (the rescale and zero-init work on the noise prediction x - denoised).
void guide(const GuidanceTweaks& tweaks, GuidanceState& state, const float* x, const float* cond, const float* uncond, double scale, size_t perSample, size_t batch, float* out);

// Two positive prompts at separate strengths. regular: neg + cfgB (condB - neg) + cfgA (condA - condB);
// nested: neg + cfgB (condB + cfgA (condA - condB) - neg).
void dualCfg(const float* condA, const float* condB, const float* negative, double cfgA, double cfgB, bool nested, float* out, size_t count);

// Perpendicular negative guidance (Armandpour et al. 2023): the negative prompt only removes the part of itself that is
// perpendicular to the positive one. `empty` is the prediction for an empty prompt.
void perpNegCfg(const float* positive, const float* negative, const float* empty, double cfg, double negativeScale, size_t perSample, size_t batch, float* out);

}  // namespace gh::sampling
