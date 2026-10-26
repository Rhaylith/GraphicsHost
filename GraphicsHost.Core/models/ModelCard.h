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
// What a model file is for. The first five existed before node graphs; the others are the kinds the loader nodes list.
enum class Kind {
    Checkpoint, Vae, Lora, Embedding, Other,
    DiffusionModel, TextEncoder, ControlNet, ClipVision, StyleModel, UpscaleModel, Hypernetwork, Gligen, PhotoMaker, PreviewDecoder,
};

const char* toString(Family family);   // "SD 1.x", "SD 2.x", "SDXL", "SD3", "FLUX", "unknown"
const char* toString(Kind kind);       // "checkpoint", "vae", "lora", "diffusion_model", ...

// The folder name that makes files below it that kind, and that names the kind in model-paths.json and in the node
// catalogue: checkpoints, diffusion_models, text_encoders, vae, loras, controlnet, clip_vision, style_models, upscale_models,
// embeddings, hypernetworks, gligen, photomaker, vae_approx. Empty for Kind::Other.
const char* folderName(Kind kind);
std::optional<Kind> kindFromFolder(const std::string& name);   // case-insensitive; also accepts the toString names
const std::vector<Kind>& libraryKinds();                       // every kind except Other

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
    std::string relative;           // path below the root it was found in, forward slashes: this is the name node combos show
    std::filesystem::path root;     // the folder it was found in
    std::string kindSource = "header";  // header | folder (a kind subfolder such as loras/) | root (a model-paths.json folder)
    std::filesystem::path preview;  // name.preview.png (or .jpg, .webp) next to the file, if there is one
    std::string sha256;             // when known without reading the file: from the download record or the hash cache
};

// Builds the card of a file: reads the header (safetensors, gguf) or guesses from the size (.ckpt, unreadable).
// `forced`: the kind is known from where the file lives (a kind folder), so a file whose header cannot be read (.pth, .pt, .bin,
// .ckpt) is not taken for a checkpoint and a header that says otherwise is not believed.
ModelCard describeFile(const std::filesystem::path& file, std::optional<Kind> forced = std::nullopt);

nlohmann::json toJson(const ModelCard& card);

}  // namespace gh::models
