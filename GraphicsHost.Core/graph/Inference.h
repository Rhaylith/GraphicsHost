#pragma once

#include "graph/Value.h"
#include "text/PromptParser.h"

#include <string>
#include <vector>

namespace gh::graph {

struct ModelHandle;
struct TextEncoderHandle;
struct VaeHandle;

// What a latent is to the inference service: a flat array in the layout of the model. Shape is [batch, channels, height, width]
// (row-major, which is the same memory as the runtime's {width, height, channels, batch}).
struct LatentView {
    const float* data = nullptr;
    int64_t batch = 1, channels = 4, height = 0, width = 0;
    size_t count() const { return static_cast<size_t>(batch * channels * height * width); }
};

// The model side of the sampling nodes, provided by the host (backed by the runtime), faked in tests. Calls come from the
// worker thread of the job, one at a time. Failures throw NodeError (or Interrupted).
class InferenceService {
public:
    virtual ~InferenceService() = default;

    // (The parsed prompt carries the weights, embeddings and BREAKs; a backend that can only encode plain text uses plainText().)
    // The conditioning for a prompt: the output of the text encoders as a CONDITIONING value. `zeroOut` gives the all-zero
    // encoding SDXL uses for an empty negative prompt.
    virtual Value encodeText(const TextEncoderHandle& encoder, const text::ParsedPrompt& prompt, int clipSkip, int width, int height, bool zeroOut) = 0;

    // The model's estimate of the clean latent for x at noise level sigma, for one conditioning (no guidance). `out` has the size of x.
    virtual void denoise(const ModelHandle& model, const Value& conditioning, const LatentView& x, float sigma, std::vector<float>& out) = 0;

    // Two conditionings in one call where the backend can batch them. The default makes two calls.
    virtual void denoisePair(const ModelHandle& model, const Value& first, const Value& second, const LatentView& x, float sigma, std::vector<float>& outFirst, std::vector<float>& outSecond) {
        denoise(model, first, x, sigma, outFirst);
        denoise(model, second, x, sigma, outSecond);
    }

    // The guided estimate uncond + cfg * (cond - uncond), in one batched call where the backend can. Used when nothing needs the two
    // estimates apart (no CFG++ sampler, no guidance tweaks).
    virtual void denoiseGuided(const ModelHandle& model, const Value& cond, const Value& uncond, float cfg, const LatentView& x, float sigma, std::vector<float>& out) {
        std::vector<float> c, u;
        denoisePair(model, cond, uncond, x, sigma, c, u);
        out.resize(c.size());
        for (size_t i = 0; i < c.size(); ++i) out[i] = u[i] + cfg * (c[i] - u[i]);
    }

    // Image from latent.
    virtual tensor::Tensor decode(const VaeHandle& vae, const ModelHandle& model, const LatentView& latent) = 0;
};

}  // namespace gh::graph
