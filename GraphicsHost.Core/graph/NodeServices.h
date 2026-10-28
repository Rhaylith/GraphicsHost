#pragma once

#include <optional>
#include <string>
#include <vector>

namespace gh::graph {

class InferenceService;

// A model file as the nodes see it: where it is and what it is.
struct ModelFileRef {
    std::string kind;     // checkpoint, diffusion_model, text_encoder, vae, lora, ... (models::toString(Kind))
    std::string name;     // the combo value: the path below its model folder, forward slashes
    std::string path;     // the file
    std::string family;   // "SD 1.x", "SDXL", ... or "unknown"
    bool supported = true;
};

// The number formats the settings ask for (precision.*): "auto" keeps the format of the file.
struct PrecisionPolicy {
    std::string diffusion = "auto";
    std::string textEncoder = "auto";
    std::string vae = "auto";
    bool vaeFallbackFp32 = true;   // decode again in fp32 when the result has NaN or infinite values
};

// Live previews while sampling (settings graph.preview*).
struct PreviewPolicy {
    std::string method = "fast";   // none | fast | decoder
    int everyNSteps = 1;
    int maxSize = 512;
};

// What the host offers to nodes while a job runs. The executor hands it to every node through NodeContext::services; the
// built-in implementation (host/GraphService.cpp) is backed by the model library.
class NodeServices {
public:
    virtual ~NodeServices() = default;

    // The model of a kind with this name: the exact combo value first, then the loose rules the prompt tag <lora:name:w> has
    // always used (case-insensitive file name or name without extension, then a unique prefix). `problem` says why not
    // (no such file, several matches).
    virtual std::optional<ModelFileRef> findModel(const std::string& kind, const std::string& name, std::string* problem = nullptr) const = 0;
    // The values a combo of this kind offers, sorted.
    virtual std::vector<std::string> modelNames(const std::string& kind) const = 0;
    virtual PrecisionPolicy precisionPolicy() const { return {}; }
    // The random generator the settings ask for when a node says "setting default": "cpu-compatible" or "philox".
    virtual PreviewPolicy previewPolicy() const { return {}; }
    virtual std::string defaultRng() const { return "cpu-compatible"; }
    // The model side of the sampling nodes (text encoding, guided model calls, VAE decoding); null where no runtime is available.
    virtual InferenceService* inference() const { return nullptr; }
    // Empty when the prompt may be used; otherwise why the safety filter refuses it.
    virtual std::string refusePrompt(const std::string& prompt) const { (void)prompt; return {}; }
    // Names of textual-inversion embeddings, as written after "embedding:" in a prompt.
    virtual std::vector<std::string> embeddingNames() const = 0;
};

}  // namespace gh::graph
