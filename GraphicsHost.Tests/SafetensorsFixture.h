#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Builds small but valid safetensors files for the model library tests.
namespace gh::test::st {

namespace fs = std::filesystem;
using nlohmann::json;

struct FakeTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype = "F16";
};

inline uint64_t dtypeSize(const std::string& d) { return d == "F32" ? 4 : d == "F16" || d == "BF16" ? 2 : 1; }

// A real, valid safetensors file with zero-filled tensors.
inline void writeSafetensors(const fs::path& file, const std::vector<FakeTensor>& tensors) {
    json header = json::object();
    header["__metadata__"] = {{"format", "pt"}};
    uint64_t offset = 0;
    for (const auto& t : tensors) {
        uint64_t n = 1;
        for (auto d : t.shape) n *= static_cast<uint64_t>(d);
        const uint64_t bytes = n * dtypeSize(t.dtype);
        header[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    const std::string text = header.dump();
    fs::create_directories(file.parent_path());
    std::ofstream out(file, std::ios::binary);
    const uint64_t length = text.size();
    out.write(reinterpret_cast<const char*>(&length), 8);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    const std::string zeros(4096, '\0');
    for (uint64_t written = 0; written < offset; written += zeros.size()) {
        out.write(zeros.data(), static_cast<std::streamsize>(std::min<uint64_t>(zeros.size(), offset - written)));
    }
}

inline std::vector<FakeTensor> sd1Tensors() {
    return {{"model.diffusion_model.input_blocks.0.0.weight", {320, 4, 3, 3}},
            {"model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight", {320, 768}},
            {"cond_stage_model.transformer.text_model.embeddings.token_embedding.weight", {128, 768}},
            {"first_stage_model.decoder.conv_in.weight", {512, 4, 3, 3}}};
}


}  // namespace gh::test::st
