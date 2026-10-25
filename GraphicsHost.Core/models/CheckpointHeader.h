#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gh::models {

struct TensorInfo {
    std::string name;
    std::string dtype;              // safetensors names (F32, F16, BF16, ...) or the GGUF type as "GGML_<n>"
    std::vector<int64_t> shape;
    uint64_t bytes = 0;             // size of the tensor data in the file

    int64_t elements() const {
        int64_t n = 1;
        for (const int64_t d : shape) n *= d;
        return shape.empty() ? 0 : n;
    }
};

// What can be learned from a checkpoint without reading its weights: the tensor names, shapes and types, and the
// metadata block. Only the header of the file is read (a few MB at most), also for 7 GB files.
struct CheckpointHeader {
    enum class Format { Safetensors, Gguf };
    Format format = Format::Safetensors;
    uint64_t fileSize = 0;
    std::vector<TensorInfo> tensors;
    nlohmann::json metadata = nlohmann::json::object();  // __metadata__ (safetensors) or the key/value pairs (GGUF)

    const TensorInfo* find(const std::string& name) const;
    bool hasPrefix(const std::string& prefix) const;
    bool hasSubstring(const std::string& part) const;
};

// Reads the header of a .safetensors file. Returns nullopt and fills error when the file is not a valid safetensors
// file (too short, header larger than 256 MB, invalid JSON, offsets outside the file).
std::optional<CheckpointHeader> readSafetensorsHeader(const std::filesystem::path& file, std::string* error = nullptr);

// Reads the header of a .gguf file (versions 2 and 3).
std::optional<CheckpointHeader> readGgufHeader(const std::filesystem::path& file, std::string* error = nullptr);

// By extension: .safetensors / .sft or .gguf; nullopt for anything else (.ckpt is a pickle and is not inspected).
std::optional<CheckpointHeader> readCheckpointHeader(const std::filesystem::path& file, std::string* error = nullptr);

}  // namespace gh::models
