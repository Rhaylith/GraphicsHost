#pragma once

// Turns the JSON bodies of the generation endpoints into a GenerateCall. The OpenAI names (size, n, guidance_scale ...)
// and the native names (width, batchCount, cfgScale ...) are both accepted, so a client can use either.

#include "host/GenerationService.h"

#include <nlohmann/json.hpp>

#include <regex>
#include <stdexcept>
#include <string>

namespace gh::server {

// Throws std::invalid_argument with a message for the client when a value has the wrong shape.
inline host::GenerateCall callFromBody(const nlohmann::json& body) {
    using Json = nlohmann::json;
    host::GenerateCall call;
    call.model = body.value("model", std::string());
    call.preset = body.value("preset", std::string());

    Json request = Json::object();
    auto copy = [&](const char* from, const char* to) {
        if (body.contains(from) && !body[from].is_null()) request[to] = body[from];
    };
    for (const char* key : {"prompt", "width", "height", "steps", "seed", "sampler", "scheduler", "format"}) copy(key, key);
    copy("negative_prompt", "negativePrompt");
    copy("negativePrompt", "negativePrompt");
    copy("cfg_scale", "cfgScale");
    copy("guidance_scale", "cfgScale");
    copy("cfgScale", "cfgScale");
    copy("num_inference_steps", "steps");
    copy("n", "batchCount");
    copy("batchCount", "batchCount");
    copy("clip_skip", "clipSkip");
    copy("clipSkip", "clipSkip");
    copy("strength", "strength");
    copy("vae_tiling", "vaeTiling");
    copy("vaeTiling", "vaeTiling");
    copy("output_format", "format");
    copy("embed_metadata", "embedMetadata");
    copy("embedMetadata", "embedMetadata");
    copy("safety_filter", "safetyFilter");
    copy("safetyFilter", "safetyFilter");
    if (body.contains("size") && body["size"].is_string()) {
        static const std::regex pattern(R"((\d+)\s*[xX]\s*(\d+))");
        std::smatch m;
        const std::string size = body["size"].get<std::string>();
        if (!std::regex_match(size, m, pattern)) throw std::invalid_argument("size must look like 512x512");
        request["width"] = std::stoi(m[1]);
        request["height"] = std::stoi(m[2]);
    } else if (body.contains("size") && !body["size"].is_null()) {
        throw std::invalid_argument("size must be a string like 512x512");
    }
    call.request = request;
    if (body.contains("load") && body["load"].is_object()) call.loadOptions = body["load"];
    return call;
}

}  // namespace gh::server
