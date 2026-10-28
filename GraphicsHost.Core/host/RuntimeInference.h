#pragma once

#include "engine/RuntimeLoader.h"
#include "graph/Handles.h"
#include "graph/Inference.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace gh::host {

// The model side of the graph nodes, on the component calls of the Vulkan runtime (GH_FEATURE_COMPONENTS): it opens a checkpoint
// when a node first needs it, keeps it open for the following nodes, and encodes, denoises and decodes through gh_text_encode,
// gh_denoise and gh_vae_decode. The sampling itself runs on the host (graph/SampleRun).
//
// Limits of this version, all reported as node errors: only the model, text encoders and VAE inside one checkpoint (separate
// text encoder or VAE files and pixel space need runtime support that does not exist yet), LoRAs and weight patches are not
// applied (the weight patcher has no runtime-backed target yet), and the prompt reaches the encoder as plain text (the weights
// and BREAK chunks of the prompt syntax are not applied).
class RuntimeInference final : public graph::InferenceService {
public:
    // Gives the runtime to use; called when the first node needs the model. Returns null and fills `error` when there is none.
    using RuntimeProvider = std::function<std::shared_ptr<engine::Runtime>(std::string& error)>;
    explicit RuntimeInference(RuntimeProvider provider);
    ~RuntimeInference() override;

    graph::Value encodeText(const graph::TextEncoderHandle& encoder, const text::ParsedPrompt& prompt, int clipSkip, int width, int height, bool zeroOut) override;
    void denoise(const graph::ModelHandle& model, const graph::Value& conditioning, const graph::LatentView& x, float sigma, std::vector<float>& out) override;
    void denoiseGuided(const graph::ModelHandle& model, const graph::Value& cond, const graph::Value& uncond, float cfg, const graph::LatentView& x, float sigma, std::vector<float>& out) override;
    tensor::Tensor decode(const graph::VaeHandle& vae, const graph::ModelHandle& model, const graph::LatentView& latent) override;

    // Closes the open checkpoint (the "free memory" action).
    void release();
    bool loaded() const;

private:
    struct Open;
    Open& open(const graph::ModelFileRef& file, const graph::LoadSettings& load);

    RuntimeProvider provider_;
    mutable std::mutex mutex_;
    std::unique_ptr<Open> open_;
};

}  // namespace gh::host
