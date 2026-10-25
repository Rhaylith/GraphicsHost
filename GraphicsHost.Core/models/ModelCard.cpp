#include "models/ModelCard.h"

#include "engine/FitEstimator.h"

namespace gh::models {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {



bool shapeIs(const CheckpointHeader& h, const std::string& tensor, size_t dim, int64_t value) {
    const TensorInfo* t = h.find(tensor);
    return t && t->shape.size() > dim && t->shape[dim] == value;
}

}  // namespace

const char* toString(Family family) {
    switch (family) {
        case Family::Sd1: return "SD 1.x";
        case Family::Sd2: return "SD 2.x";
        case Family::SdXl: return "SDXL";
        case Family::Sd3: return "SD3";
        case Family::Flux: return "FLUX";
        case Family::Unknown: return "unknown";
    }
    return "unknown";
}

const char* toString(Kind kind) {
    switch (kind) {
        case Kind::Checkpoint: return "checkpoint";
        case Kind::Vae: return "vae";
        case Kind::Lora: return "lora";
        case Kind::Embedding: return "embedding";
        case Kind::Other: return "other";
    }
    return "other";
}

FamilyInfo detectFamily(const CheckpointHeader& h) {
    FamilyInfo info;
    // LoRA files: pairs of low-rank matrices
    if (h.hasSubstring("lora_down") || h.hasSubstring("lora_up") || h.hasSubstring("lora_A") || h.hasSubstring("lora_unet_")) {
        info.kind = Kind::Lora;
        info.supported = true;
        info.family = h.hasSubstring("lora_te2_") || h.hasSubstring("text_encoder_2") || h.hasSubstring("1280") ? Family::SdXl : Family::Unknown;
        if (h.hasSubstring("lora_te2_") || h.hasSubstring("text_encoder_2")) info.family = Family::SdXl;
        else if (h.hasSubstring("lora_te_") || h.hasSubstring("lora_unet_")) info.family = Family::Sd1;
        return info;
    }
    // textual inversion embeddings
    if (h.tensors.size() <= 3 && (h.find("emb_params") || h.find("clip_l") || h.find("string_to_param.*"))) {
        info.kind = Kind::Embedding;
        info.supported = true;
        return info;
    }
    // newer families, detected so they can be reported as not supported
    if (h.hasSubstring("double_blocks.") && h.hasSubstring("single_blocks.")) {
        info = {Kind::Checkpoint, Family::Flux, false, false, "FLUX models are not supported in this version (SD 1.x and SDXL only)"};
        return info;
    }
    if (h.hasSubstring("joint_blocks.")) {
        info = {Kind::Checkpoint, Family::Sd3, false, false, "SD3 models are not supported in this version (SD 1.x and SDXL only)"};
        return info;
    }
    const bool hasUnet = h.hasSubstring("diffusion_model.input_blocks.") || h.hasSubstring("diffusion_model.output_blocks.") ||
                         h.hasSubstring("down_blocks.");
    if (hasUnet) {
        info.kind = Kind::Checkpoint;
        info.inpainting = shapeIs(h, "model.diffusion_model.input_blocks.0.0.weight", 1, 9);
        const bool xl = h.hasPrefix("conditioner.embedders.") || h.find("model.diffusion_model.label_emb.0.0.weight") != nullptr ||
                        h.hasSubstring("add_embedding.");
        const bool v2 = h.hasSubstring("cond_stage_model.model.transformer.resblocks") || h.hasSubstring("text_model.encoder.layers.23");
        if (xl) {
            info.family = Family::SdXl;
            info.supported = true;
        } else if (v2) {
            info.family = Family::Sd2;
            info.supported = true;
            info.note = "SD 2.x is accepted by the engine but has not been tested in this version";
        } else {
            info.family = Family::Sd1;
            info.supported = true;
        }
        return info;
    }
    // a standalone VAE: encoder and decoder, no UNet
    if (h.hasPrefix("decoder.") && h.hasPrefix("encoder.")) {
        info.kind = Kind::Vae;
        info.supported = true;
        return info;
    }
    if (h.hasPrefix("first_stage_model.decoder.") && !hasUnet) {
        info.kind = Kind::Vae;
        info.supported = true;
        return info;
    }
    info.note = "not a recognised Stable Diffusion file";
    return info;
}

// What the engine really holds in memory is not "everything as fp16": measured on the two reference checkpoints
// (docs/vram-measurements.md), an fp32 file loads to 0.653 of its size and an fp16 file to 0.955 of its size (some
// tensors stay in higher precision, some are unused). Per tensor type, with those two ratios:
uint64_t estimateWeightsBytes(const CheckpointHeader& h) {
    if (h.format == CheckpointHeader::Format::Gguf) {
        return h.fileSize;
    }
    double total = 0;
    for (const auto& t : h.tensors) {
        if (t.dtype == "F32" || t.dtype == "F64") total += static_cast<double>(t.bytes) * 0.653;
        else if (t.dtype == "F16" || t.dtype == "BF16") total += static_cast<double>(t.bytes) * 0.955;
        else total += static_cast<double>(t.bytes);
    }
    return static_cast<uint64_t>(total);
}

ModelCard describeFile(const fs::path& file) {
    ModelCard card;
    card.path = file;
    card.file = file.filename().string();
    card.name = file.stem().string();
    std::error_code ec;
    card.sizeBytes = fs::file_size(file, ec);

    std::string error;
    if (const auto header = readCheckpointHeader(file, &error)) {
        const FamilyInfo info = detectFamily(*header);
        card.headerRead = true;
        card.kind = info.kind;
        card.family = info.family;
        card.supported = info.supported;
        card.inpainting = info.inpainting;
        card.note = info.note;
        card.weightsBytes = estimateWeightsBytes(*header);
    } else {
        // .ckpt and unreadable files: assume a checkpoint and guess the family from the size
        card.problem = error;
        card.kind = Kind::Checkpoint;
        card.family = card.sizeBytes > 5500ull * 1024 * 1024 ? Family::SdXl : Family::Sd1;
        card.supported = file.extension() == ".ckpt";
        card.note = file.extension() == ".ckpt" ? "family guessed from the file size (.ckpt files are not inspected)" : error;
        card.weightsBytes = engine::estimateWeightsBytes(card.sizeBytes, toString(card.family));
    }
    card.familyName = toString(card.family);
    card.defaults = engine::familyDefaults(card.familyName);
    return card;
}

Json toJson(const ModelCard& c) {
    return {{"name", c.name},
            {"file", c.file},
            {"path", c.path.string()},
            {"sizeBytes", c.sizeBytes},
            {"kind", toString(c.kind)},
            {"family", c.familyName},
            {"supported", c.supported},
            {"inpainting", c.inpainting},
            {"note", c.note},
            {"weightsBytes", c.weightsBytes},
            {"headerRead", c.headerRead},
            {"vae", c.vae},
            {"source", c.source},
            {"companions", c.companions},
            {"defaults", {{"width", c.defaults.width}, {"height", c.defaults.height}, {"steps", c.defaults.steps}, {"cfgScale", c.defaults.cfgScale}}}};
}

}  // namespace gh::models
