#include "engine/GenerationRequest.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace gh::engine {

namespace {

using Json = nlohmann::json;

bool contains(const std::vector<std::string>& list, const std::string& value) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

std::string joined(const std::vector<std::string>& list) {
    std::string out;
    for (const auto& item : list) {
        out += (out.empty() ? "" : ", ") + item;
    }
    return out;
}

bool validDevice(const std::string& device) {
    if (device == "auto" || device == "cpu") {
        return true;
    }
    if (device.rfind("vulkan", 0) != 0 || device.size() <= 6) {
        return false;
    }
    return std::all_of(device.begin() + 6, device.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

// Copies json[key] into target when present; throws json::type_error on a wrong type.
template <typename T>
void takeIf(const Json& json, const char* key, T& target) {
    if (const auto it = json.find(key); it != json.end() && !it->is_null()) {
        target = it->get<T>();
    }
}

}  // namespace

FamilyDefaults familyDefaults(std::string_view family) {
    FamilyDefaults d;
    if (family.find("XL") != std::string_view::npos) {
        d.width = d.height = 1024;
        d.steps = 25;
        d.cfgScale = 7.0f;
    }
    return d;
}

GenerationRequest resolve(GenerationRequest r, const FamilyDefaults& d) {
    if (r.width <= 0) r.width = d.width;
    if (r.height <= 0) r.height = d.height;
    if (r.steps <= 0) r.steps = d.steps;
    if (r.cfgScale < 0) r.cfgScale = d.cfgScale;
    return r;
}

const std::vector<std::string>& knownSamplers() {
    static const std::vector<std::string> list = {
        "euler",   "euler_a",  "heun",          "dpm2",           "dpm++2s_a",
        "dpm++2m", "dpm++2mv2", "ipndm",        "ipndm_v",        "lcm",
        "ddim_trailing", "tcd", "res_multistep", "res_2s",        "lms",
    };
    return list;
}

const std::vector<std::string>& knownSchedulers() {
    static const std::vector<std::string> list = {
        "discrete", "karras", "exponential", "ays",         "gits",        "sgm_uniform",
        "simple",   "smoothstep", "kl_optimal", "lcm",      "bong_tangent",
    };
    return list;
}

std::vector<core::FieldError> validate(const GenerationRequest& r) {
    std::vector<core::FieldError> errors;
    if (r.prompt.empty()) {
        errors.push_back({"prompt", "must not be empty"});
    }
    auto checkSize = [&](const char* field, int value) {
        if (value != 0 && (value < 64 || value > 4096 || value % 8 != 0)) {
            errors.push_back({field, "must be 0 (model default) or a multiple of 8 between 64 and 4096"});
        }
    };
    checkSize("width", r.width);
    checkSize("height", r.height);
    if (r.steps < 0 || r.steps > 150) {
        errors.push_back({"steps", "must be between 0 (model default) and 150"});
    }
    if (r.cfgScale >= 0 && r.cfgScale > 30.0f) {
        errors.push_back({"cfgScale", "must be at most 30"});
    }
    if (r.seed < -1) {
        errors.push_back({"seed", "must be -1 (random) or 0 or greater"});
    }
    if (r.batchCount < 1 || r.batchCount > 16) {
        errors.push_back({"batchCount", "must be between 1 and 16"});
    }
    if (!r.sampler.empty() && !contains(knownSamplers(), r.sampler)) {
        errors.push_back({"sampler", "must be one of: " + joined(knownSamplers())});
    }
    if (!r.scheduler.empty() && !contains(knownSchedulers(), r.scheduler)) {
        errors.push_back({"scheduler", "must be one of: " + joined(knownSchedulers())});
    }
    if (r.clipSkip < -1 || r.clipSkip > 12) {
        errors.push_back({"clipSkip", "must be -1 (model default) or between 0 and 12"});
    }
    if (r.initImage && !(r.strength > 0.0f && r.strength <= 1.0f)) {
        errors.push_back({"strength", "must be above 0 and at most 1"});
    }
    if (r.format != "png" && r.format != "jpeg" && r.format != "webp") {
        errors.push_back({"format", "must be png, jpeg or webp"});
    }
    return errors;
}

std::vector<core::FieldError> validate(const LoadOptions& o) {
    std::vector<core::FieldError> errors;
    if (o.modelPath.empty()) {
        errors.push_back({"modelPath", "must not be empty"});
    }
    if (!validDevice(o.device)) {
        errors.push_back({"device", "must be auto, cpu or vulkan<N> (for example vulkan0)"});
    }
    if (o.threads < 0 || o.threads > 1024) {
        errors.push_back({"threads", "must be between 0 (automatic) and 1024"});
    }
    if (o.ttlMinutes < 0 || o.ttlMinutes > 1440) {
        errors.push_back({"ttlMinutes", "must be between 0 (never unload) and 1440"});
    }
    return errors;
}

Json toJson(const GenerationRequest& r) {
    return {{"prompt", r.prompt},
            {"negativePrompt", r.negativePrompt},
            {"width", r.width},
            {"height", r.height},
            {"steps", r.steps},
            {"cfgScale", r.cfgScale},
            {"seed", r.seed},
            {"batchCount", r.batchCount},
            {"sampler", r.sampler},
            {"scheduler", r.scheduler},
            {"clipSkip", r.clipSkip},
            {"vaeTiling", r.vaeTiling},
            {"format", r.format},
            {"embedMetadata", r.embedMetadata},
            {"safetyFilter", r.safetyFilter},
            {"strength", r.strength},
            {"hasInitImage", r.initImage != nullptr}};
}

GenerationRequest applyRequestJson(GenerationRequest r, const Json& json) {
    if (!json.is_object()) {
        throw std::runtime_error("generation parameters must be a JSON object");
    }
    takeIf(json, "prompt", r.prompt);
    takeIf(json, "negativePrompt", r.negativePrompt);
    takeIf(json, "width", r.width);
    takeIf(json, "height", r.height);
    takeIf(json, "steps", r.steps);
    takeIf(json, "cfgScale", r.cfgScale);
    takeIf(json, "seed", r.seed);
    takeIf(json, "batchCount", r.batchCount);
    takeIf(json, "sampler", r.sampler);
    takeIf(json, "scheduler", r.scheduler);
    takeIf(json, "clipSkip", r.clipSkip);
    takeIf(json, "vaeTiling", r.vaeTiling);
    takeIf(json, "format", r.format);
    takeIf(json, "embedMetadata", r.embedMetadata);
    takeIf(json, "safetyFilter", r.safetyFilter);
    takeIf(json, "strength", r.strength);
    return r;
}

GenerationRequest requestFromJson(const Json& json) {
    return applyRequestJson(GenerationRequest{}, json);
}

Json toJson(const LoadOptions& o) {
    return {{"modelPath", o.modelPath},
            {"vaePath", o.vaePath},
            {"device", o.device},
            {"threads", o.threads},
            {"flashAttention", o.flashAttention},
            {"offloadParamsToCpu", o.offloadParamsToCpu},
            {"clipOnCpu", o.clipOnCpu},
            {"vaeOnCpu", o.vaeOnCpu},
            {"ttlMinutes", o.ttlMinutes}};
}

LoadOptions applyLoadOptionsJson(LoadOptions o, const Json& json) {
    if (!json.is_object()) {
        throw std::runtime_error("load options must be a JSON object");
    }
    takeIf(json, "modelPath", o.modelPath);
    takeIf(json, "vaePath", o.vaePath);
    takeIf(json, "device", o.device);
    takeIf(json, "threads", o.threads);
    takeIf(json, "flashAttention", o.flashAttention);
    takeIf(json, "offloadParamsToCpu", o.offloadParamsToCpu);
    takeIf(json, "clipOnCpu", o.clipOnCpu);
    takeIf(json, "vaeOnCpu", o.vaeOnCpu);
    takeIf(json, "ttlMinutes", o.ttlMinutes);
    return o;
}

LoadOptions loadOptionsFromJson(const Json& json) {
    return applyLoadOptionsJson(LoadOptions{}, json);
}

}  // namespace gh::engine
