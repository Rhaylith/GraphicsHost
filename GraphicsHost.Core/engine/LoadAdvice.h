#pragma once

#include "engine/GenerationRequest.h"
#include "engine/ImageEngine.h"

#include <string>
#include <vector>

namespace gh::engine {

// What to try after a model failed to load. The application only OFFERS these (the CLI prints them, the UI shows
// buttons): it never retries on the CPU or with other options by itself, because that changes speed and memory use in
// ways the user should decide.
struct LoadSuggestion {
    std::string description;  // for people: "keep the weights in system RAM"
    std::string cliFlags;     // the equivalent command-line flags: "--offload"
    LoadOptions options;      // the options to retry with
};

// runtimeBackend: "cpu" or "vulkan", the backend of the loaded runtime. A failure on the CPU has no GPU-related advice.
std::vector<LoadSuggestion> suggestLoadFallbacks(const LoadOptions& failed, EngineErrorCode code,
                                                 const std::string& runtimeBackend);

}  // namespace gh::engine
