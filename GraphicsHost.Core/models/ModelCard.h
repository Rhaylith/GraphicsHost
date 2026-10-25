#pragma once

#include "engine/GenerationRequest.h"
#include "models/CheckpointHeader.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gh::models {

enum class Family { Sd1, Sd2, SdXl, Sd3, Flux, Unknown };
enum class Kind { Checkpoint, Vae, Lora, Embedding, Other };

const char* toString(Family family);   // "SD 1.x", "SD 2.x", "SDXL", "SD3", "FLUX", "unknown"
const char* toString(Kind kind);       // "checkpoint", "vae", "lora", ...

struct FamilyInfo {
    Kind kind = Kind::Other;
    Family family = Family::Unknown;
    bool supported = false;       // can this version run it?
    bool inpainting = false;      // a checkpoint trained for inpainting (9 input channels)
    std::string note;             // why it is unsupported, or a caveat
};

// Classifies a file from its tensor names and shapes.
FamilyInfo detectFamily(const CheckpointHeader& header);

// Memory the weights occupy once loaded (empirical ratios per tensor type, see the implementation).
uint64_t estimateWeightsBytes(const CheckpointHeader& header);

// Everything the application knows about one model file.
struct ModelCard {
    std::string name;               // file name without the extension
    std::string file;               // file name
    std::filesystem::path path;
    uint64_t sizeBytes = 0;
    Kind kind = Kind::Other;
    Family family = Family::Unknown;
    std::string familyName;
    bool supported = false;
    bool inpainting = false;
    std::string note;
    uint64_t weightsBytes = 0;      // in memory when loaded
    bool headerRead = false;        // false for .ckpt and unreadable files: then the family is a guess
    std::string problem;            // why the header could not be read
    engine::FamilyDefaults defaults;
    std::vector<std::string> companions;  // separate files this model needs or can use (for example a VAE file name)
    std::string vae;                // a VAE file the library linked to this checkpoint (file name), if any
    nlohmann::json source = nullptr;  // where it was downloaded from (repo, revision, sha256), from the .source.json sidecar
};

// Builds the card of a file: reads the header (safetensors, gguf) or guesses from the size (.ckpt, unreadable).
ModelCard describeFile(const std::filesystem::path& file);

nlohmann::json toJson(const ModelCard& card);

}  // namespace gh::models
