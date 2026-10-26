#pragma once

#include <optional>
#include <string>
#include <vector>

namespace gh::graph {

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
    // Names of textual-inversion embeddings, as written after "embedding:" in a prompt.
    virtual std::vector<std::string> embeddingNames() const = 0;
};

}  // namespace gh::graph
