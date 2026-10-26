#pragma once

#include "models/CheckpointHeader.h"
#include "tensor/Tensor.h"

#include <filesystem>
#include <map>
#include <optional>
#include <string>

namespace gh::models {

// Reads one tensor of a safetensors file as float32 (F32, F16, BF16 and F64 are converted). A scalar is returned with the
// shape {1}. nullopt (and a reason in `error`) for other types, tensors with more than 5 dimensions, or a damaged file.
std::optional<tensor::Tensor> readSafetensor(const std::filesystem::path& file, const TensorInfo& info, std::string* error = nullptr);

// All tensors of a file, by name. Meant for small files (LoRAs); a checkpoint would not fit in memory this way.
std::map<std::string, tensor::Tensor> readAllSafetensors(const std::filesystem::path& file, std::string* error = nullptr);

}  // namespace gh::models
