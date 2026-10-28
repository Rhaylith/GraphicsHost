#pragma once

#include "graph/NodeServices.h"
#include "graph/Value.h"
#include "sampling/Guidance.h"
#include "sampling/ModelSampling.h"
#include "sampling/Samplers.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gh::graph {

// What the loader nodes put on their links (workflow plan, section 6.3). A loader does not read weights: it describes what
// to load and how, so a model that is never used costs nothing and the component cache and memory manager decide when
// the weights are really opened. Handles are immutable; a node that changes one (a LoRA) makes a copy.

// How a component is to be opened. "auto" lets the memory manager decide.
struct LoadSettings {
    std::string device = "auto";            // auto | vulkan0 | vulkan1 ...
    std::string flashAttention = "auto";    // auto | on | off
    std::string offloadToRam = "auto";      // auto | on | off
    int idleUnloadMinutes = 10;             // 0 = never
    std::string weightDtype = "default";    // default | fp8 e4m3 | fp8 e4m3 fast | fp8 e5m2 ...
    bool operator==(const LoadSettings&) const = default;
};

// A low-rank (or full) weight change to be applied to a component when it is next used.
struct PatchRef {
    ModelFileRef lora;
    double strengthModel = 1.0;
    double strengthText = 1.0;
};

struct ModelHandle {
    ModelFileRef file;             // a checkpoint (its diffusion part) or a diffusion model file
    LoadSettings load;
    std::vector<PatchRef> patches;
    std::optional<sampling::SamplingParams> sampling;   // set by the model sampling nodes; otherwise the family default
    sampling::GuidanceTweaks guidance;                    // set by Rescale CFG, CFG-Zero* and APG
};

struct TextEncoderHandle {
    std::vector<ModelFileRef> files;   // the checkpoint itself, or one file per encoder
    std::string family = "auto";       // which family the encoders are for (separate encoder files only)
    LoadSettings load;
    std::vector<PatchRef> patches;
    bool fromCheckpoint = false;
    int clipSkip = 1;                  // 1 = the last layer, 2 = the one before ...
};

struct VaeHandle {
    ModelFileRef file;                 // empty path when pixelSpace
    bool fromCheckpoint = false;       // the VAE inside the checkpoint
    bool pixelSpace = false;           // the model works on pixels: encoding and decoding are the identity
    bool fallbackFp32 = true;          // decode again in fp32 when the result has invalid values (setting precision.vaeFallbackFp32)
    LoadSettings load;
};

// A NOISE link: where the starting noise of a sampler comes from.
struct NoiseHandle {
    bool none = false;          // no noise at all
    uint64_t seed = 0;
    std::string rng = "cpu-compatible";   // resolved: cpu-compatible | philox
};

// A GUIDER link: how the model is called for a step and how its predictions are combined.
struct GuiderHandle {
    enum class Kind { Basic, Cfg, DualCfg, PerpNeg };
    Kind kind = Kind::Cfg;
    Value model;
    std::vector<Value> conditioning;   // basic: [cond]; cfg: [positive, negative]; dual: [A, B, negative]; perp-neg: [positive, negative, empty]
    double cfg = 8.0;
    double cfgA = 8.0, cfgB = 8.0;
    bool nested = false;
    double negativeScale = 1.0;
};

// A SAMPLER link: which sampler to run and with what settings.
struct SamplerHandle {
    std::string name;
    sampling::SamplerOptions options;
    std::string generator = "setting default";   // for the SDE noise: setting default | cpu-compatible | philox
};

// A SIGMAS link: the noise levels of a run, high to low, as a 1-D tensor.
inline Value sigmasValue(const std::vector<float>& sigmas) {
    return Value::makeTensor(ValueType::Sigmas, tensor::Tensor({static_cast<int64_t>(sigmas.size())}, sigmas));
}
inline std::vector<float> sigmasOf(const Value& v) { return v.asTensor().values(); }

template <typename Handle>
Value handleValue(ValueType type, Handle handle) {
    return Value::makeOpaque(type, std::make_shared<const Handle>(std::move(handle)));
}

}  // namespace gh::graph
