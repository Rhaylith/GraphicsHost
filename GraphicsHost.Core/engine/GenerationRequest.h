#pragma once

#include "core/Settings.h"  // FieldError

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gh::engine {

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 3;       // 3 = RGB, 4 = RGBA
    std::vector<uint8_t> data;   // width * height * channels bytes, row by row
};

// The standard parameter set, independent of model family and of the API it arrived through (see implementation plan,
// section 5). A field a family does not support is ignored and reported in GenerationResult::warnings.
// "Unset" values (0, -1, empty string) mean: use the model's default; resolve() fills them in.
struct GenerationRequest {
    std::string prompt;
    std::string negativePrompt;
    int width = 0;             // 0 = family default; otherwise a multiple of 8
    int height = 0;
    int steps = 0;             // 0 = family default
    float cfgScale = -1.0f;    // < 0 = family default
    int64_t seed = -1;         // -1 = random; the real seed is always reported back
    int batchCount = 1;        // images per request
    std::string sampler;       // empty = model default, see knownSamplers()
    std::string scheduler;     // empty = model default, see knownSchedulers()
    int clipSkip = -1;         // -1 = model default
    bool vaeTiling = false;    // decode the image in tiles: slower, needs much less memory
    std::string format = "png";  // png | jpeg | webp
    bool embedMetadata = true;   // write the parameters into the image file
    bool safetyFilter = false;   // per request override of safety.enabled; off by default
    // image to image: start from this picture instead of noise. It is not part of the JSON form (pictures travel as
    // files); the JSON form only says whether there is one.
    std::shared_ptr<const Image> initImage;
    float strength = 0.75f;      // with initImage: how much of it is replaced, (0..1]; 1 ignores the picture
    std::shared_ptr<const Image> maskImage;   // with initImage: white = repaint, black = keep (inpainting); same size as the result
};

// How a model is put in memory. Changing it needs a reload. Mirrors gh_load_params in runtime/gh_runtime.h.
struct LoadOptions {
    std::string modelPath;
    std::string vaePath;           // optional separate VAE
    std::string device = "auto";   // auto | cpu | vulkan<N>
    int threads = 0;               // CPU threads, 0 = number of physical cores
    bool flashAttention = false;
    bool offloadParamsToCpu = false;  // keep weights in RAM and stream them to the GPU (low VRAM)
    bool clipOnCpu = false;
    bool vaeOnCpu = false;
    int ttlMinutes = 10;           // unload after this many idle minutes, 0 = keep loaded

    bool operator==(const LoadOptions&) const = default;
};

// Defaults that depend on the model family ("SDXL" -> 1024x1024 ...). The family string is what the runtime reports.
struct FamilyDefaults {
    int width = 512;
    int height = 512;
    int steps = 20;
    float cfgScale = 7.5f;
};
FamilyDefaults familyDefaults(std::string_view family);

// Replaces every unset field by the family default.
GenerationRequest resolve(GenerationRequest request, const FamilyDefaults& defaults);

const std::vector<std::string>& knownSamplers();
const std::vector<std::string>& knownSchedulers();

std::vector<core::FieldError> validate(const GenerationRequest& request);
std::vector<core::FieldError> validate(const LoadOptions& options);

nlohmann::json toJson(const GenerationRequest& request);
// Missing keys keep their defaults; unknown keys are ignored; a wrong type for a known key throws
// nlohmann::json::exception.
GenerationRequest requestFromJson(const nlohmann::json& json);
// Applies only the keys present in json on top of base (presets and per-model defaults are partial documents).
GenerationRequest applyRequestJson(GenerationRequest base, const nlohmann::json& json);

nlohmann::json toJson(const LoadOptions& options);
LoadOptions loadOptionsFromJson(const nlohmann::json& json);
LoadOptions applyLoadOptionsJson(LoadOptions base, const nlohmann::json& json);

}  // namespace gh::engine
