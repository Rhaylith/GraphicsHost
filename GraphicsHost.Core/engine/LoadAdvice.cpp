#include "engine/LoadAdvice.h"

namespace gh::engine {

std::vector<LoadSuggestion> suggestLoadFallbacks(const LoadOptions& failed, EngineErrorCode code,
                                                 const std::string& runtimeBackend) {
    std::vector<LoadSuggestion> out;
    // Only failures that can depend on the device are worth a retry; a missing file or bad options will fail again.
    if (code != EngineErrorCode::LoadFailed && code != EngineErrorCode::OutOfMemory) {
        return out;
    }
    const bool onGpu = runtimeBackend == "vulkan" && failed.device != "cpu";
    if (!onGpu) {
        return out;
    }
    if (!failed.offloadParamsToCpu) {
        LoadSuggestion offload;
        offload.description = "keep the model weights in system RAM and move only what is needed to the GPU (uses less GPU memory)";
        offload.cliFlags = "--offload";
        offload.options = failed;
        offload.options.offloadParamsToCpu = true;
        out.push_back(std::move(offload));
    }
    LoadSuggestion cpu;
    cpu.description = "run on the CPU instead (works with any amount of GPU memory, but takes minutes per image)";
    cpu.cliFlags = "--device cpu";
    cpu.options = failed;
    cpu.options.device = "cpu";
    cpu.options.offloadParamsToCpu = false;
    out.push_back(std::move(cpu));
    return out;
}

}  // namespace gh::engine
