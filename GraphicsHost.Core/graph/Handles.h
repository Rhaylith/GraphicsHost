#pragma once

#include "graph/NodeServices.h"
#include "graph/Value.h"

#include <memory>
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
};

struct TextEncoderHandle {
    std::vector<ModelFileRef> files;   // the checkpoint itself, or one file per encoder
    std::string family = "auto";       // which family the encoders are for (separate encoder files only)
    LoadSettings load;
    std::vector<PatchRef> patches;
    bool fromCheckpoint = false;
};

struct VaeHandle {
    ModelFileRef file;                 // empty path when pixelSpace
    bool fromCheckpoint = false;       // the VAE inside the checkpoint
    bool pixelSpace = false;           // the model works on pixels: encoding and decoding are the identity
    LoadSettings load;
};

template <typename Handle>
Value handleValue(ValueType type, Handle handle) {
    return Value::makeOpaque(type, std::make_shared<const Handle>(std::move(handle)));
}

}  // namespace gh::graph
