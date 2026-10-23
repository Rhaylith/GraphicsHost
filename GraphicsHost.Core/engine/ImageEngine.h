#pragma once

#include "engine/GenerationRequest.h"
#include "engine/RuntimeLoader.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

namespace gh::engine {

struct GenerationResult {
    GenerationRequest request;           // as generated: unset fields resolved, seed replaced by the real first seed
    std::vector<Image> images;
    std::vector<int64_t> seeds;          // the seed actually used for each image
    std::vector<std::string> warnings;   // ignored or adjusted parameters
    double seconds = 0;
};

struct StepProgress {
    int step = 0;
    int steps = 0;
    float secondsPerStep = 0;
};
using ProgressSink = std::function<void(const StepProgress&)>;

enum class EngineErrorCode {
    InvalidRequest,  // parameters did not validate
    NotFound,        // model file missing
    LoadFailed,
    OutOfMemory,
    Cancelled,
    Failed,
};

class EngineError : public std::runtime_error {
public:
    EngineError(EngineErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}
    EngineErrorCode code() const { return code_; }

private:
    EngineErrorCode code_;
};

struct ModelInfo {
    std::string modelPath;
    std::string family;       // as the runtime reports it ("SD 1.x", "SDXL", ...)
    std::string runtimeId;    // "cpu-avx2", "vulkan", ...
    std::string device;       // what was requested: auto, cpu, vulkan0
    FamilyDefaults defaults;
};

// A model in memory, ready to generate. Not thread safe: use one generation at a time per process (the JobQueue does).
class ILoadedModel {
public:
    virtual ~ILoadedModel() = default;
    virtual const ModelInfo& info() const = 0;
    // Validates and resolves the request, then generates. Requesting a stop (stop_token) cancels at the next sampling
    // step and throws EngineError(Cancelled). progress may be empty.
    virtual GenerationResult generate(const GenerationRequest& request, const ProgressSink& progress,
                                      std::stop_token stop) = 0;
};

class IImageEngine {
public:
    virtual ~IImageEngine() = default;
    virtual std::string runtimeId() const = 0;
    // Throws EngineError (NotFound, InvalidRequest, LoadFailed, OutOfMemory).
    virtual std::unique_ptr<ILoadedModel> load(const LoadOptions& options) = 0;
};

// The engine over a loaded runtime DLL (sd-*.dll). Models keep the runtime alive.
std::unique_ptr<IImageEngine> makeRuntimeEngine(std::shared_ptr<Runtime> runtime);

}  // namespace gh::engine
