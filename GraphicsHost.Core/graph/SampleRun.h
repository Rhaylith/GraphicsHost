#pragma once

#include "graph/Handles.h"
#include "graph/Inference.h"
#include "graph/NodeApi.h"

#include <functional>
#include <optional>
#include <vector>

namespace gh::graph {

// One run of a sampler over a latent: what the Sampler and Sample (custom) nodes do (workflow plan, section 6.4).
struct SampleRequest {
    GuiderHandle guider;
    SamplerHandle sampler;
    NoiseHandle noise;
    std::vector<float> sigmas;               // the whole schedule; [firstStep, lastStep] of it is run
    int firstStep = 0;
    int lastStep = -1;                       // < 0 or beyond the end: to the end
    bool returnWithLeftoverNoise = false;    // otherwise a run that ends early finishes at noise level 0
    tensor::Tensor latent;                   // [batch, channels, height, width]
    std::optional<tensor::Tensor> noiseMask; // inpainting: 1 = change, 0 = keep; the same element count as the latent, or one channel of it
    InferenceService* inference = nullptr;
    std::string defaultRng = "cpu-compatible";
    std::function<void(int step, int steps)> progress;
    std::stop_token stop;
};

struct SampleResult {
    tensor::Tensor output;
    tensor::Tensor denoised;   // the last estimate of the clean latent
};

// Throws NodeError for a sampler that is not available or a missing service, Interrupted when the stop token is set.
SampleResult runSampling(const SampleRequest& request);

}  // namespace gh::graph
