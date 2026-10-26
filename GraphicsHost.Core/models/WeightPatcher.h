#pragma once

#include "tensor/Tensor.h"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gh::models {

// Adapters that change the weights of a model without retraining it (workflow plan, section 6.3): LoRA (and LoCon, which adds
// convolution layers), LoHa, LoKr, DoRA, and full weight differences. The patcher turns the tensors of an adapter file into a
// change of each weight it targets, adds that change to the weights of an opened component, and takes it away again: the
// originals are kept, so a different list of adapters can be applied without reopening the component.

// The weights of one opened component as the patcher sees them. The runtime-backed component implements it; tests use a map.
class WeightTarget {
public:
    virtual ~WeightTarget() = default;
    virtual std::vector<std::string> keys() const = 0;
    virtual bool read(const std::string& key, tensor::Tensor& out) const = 0;
    virtual void write(const std::string& key, const tensor::Tensor& value) = 0;
};

enum class TargetKind { Diffusion, TextEncoder };

// The tensors that belong to one layer of an adapter file.
struct AdapterModule {
    std::string name;                                  // "lora_unet_down_blocks_0_attentions_0_proj_in" (no tensor role)
    std::map<std::string, tensor::Tensor> tensors;     // by role: down, up, mid, hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b, hada_t1, hada_t2,
                                                       // lokr_w1, lokr_w1_a, lokr_w1_b, lokr_w2, lokr_w2_a, lokr_w2_b, lokr_t2, dora_scale, diff, diff_b
    std::optional<double> alpha;
};

class AdapterFile {
public:
    // Reads a .safetensors adapter file. nullptr and a reason in `error` when it cannot be read or holds no adapter tensors.
    static std::shared_ptr<const AdapterFile> load(const std::filesystem::path& file, std::string* error = nullptr);
    // From tensors by name, as they would be in a file (tests, and adapters made in memory).
    static std::shared_ptr<const AdapterFile> fromTensors(const std::map<std::string, tensor::Tensor>& tensors, std::string identity = "memory");

    std::vector<AdapterModule> modules;
    std::string identity;   // path, size and time of the file: two applications of the same file are recognised

    // Which component the module's name says it is for. Unknown names are tried on both.
    static std::optional<TargetKind> targetOf(const std::string& moduleName);
};

struct PatchSpec {
    std::shared_ptr<const AdapterFile> file;
    double strengthModel = 1.0;   // for modules of the diffusion model
    double strengthText = 1.0;    // for modules of the text encoders
};

// The change an adapter module makes to a weight of the given shape, at strength 1 (alpha and rank already taken into account;
// DoRA is not part of it, see WeightPatcher). Throws std::runtime_error when the module's tensors do not fit the shape.
tensor::Tensor adapterDelta(const AdapterModule& module, const std::vector<int64_t>& weightShape);

// "input_blocks.1.1.transformer_blocks.0.attn1.to_q" -> "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q": the
// diffusers name of a layer of the original (ldm) UNet layout. Unknown paths come back unchanged.
std::string diffusersUnetPath(const std::string& ldmPath);

// Finds the weight of a model that an adapter module is for, whatever naming the adapter file and the model use (kohya
// "lora_unet_...", "lora_te_/te1_/te2_...", dotted diffusers/PEFT names, and the weight keys of "diff" files).
class KeyResolver {
public:
    explicit KeyResolver(const std::vector<std::string>& modelKeys);
    struct Hit {
        std::string key;    // the weight in the model
        int part = 0;       // the weight packs several layers by rows (OpenCLIP q, k, v): this is layer `part` of `parts`
        int parts = 1;
    };
    std::optional<Hit> resolve(const std::string& moduleName) const;

private:
    std::map<std::string, Hit> aliases_;
    std::vector<std::string> keys_;
};

class WeightPatcher {
public:
    struct Report {
        int modules = 0;                       // modules of the adapters meant for this kind of component
        int patched = 0;                       // of those, weights changed
        std::vector<std::string> unmatched;    // modules that name no weight of the model
    };

    // Adds the adapters to the weights. Whatever was applied before is taken away first.
    Report apply(WeightTarget& target, TargetKind kind, const std::vector<PatchSpec>& patches);
    // Puts the original weights back.
    void undo(WeightTarget& target);
    // Applies only when the list differs from what is applied now (same files, same strengths): the lazy path nodes take
    // before every call.
    Report ensure(WeightTarget& target, TargetKind kind, const std::vector<PatchSpec>& patches);

    bool applied() const { return !signature_.empty(); }
    const std::string& signature() const { return signature_; }
    static std::string signatureOf(TargetKind kind, const std::vector<PatchSpec>& patches);

private:
    std::map<std::string, tensor::Tensor> originals_;
    std::string signature_;
    Report last_;
};

}  // namespace gh::models
