#include "models/SafetensorsReader.h"

#include <cstring>
#include <fstream>

namespace gh::models {

namespace {

float halfToFloat(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {   // subnormal: normalise
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

float bfloatToFloat(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

void fail(std::string* error, std::string message) {
    if (error) *error = std::move(message);
}

}  // namespace

std::optional<tensor::Tensor> readSafetensor(const std::filesystem::path& file, const TensorInfo& info, std::string* error) {
    size_t width = 0;
    if (info.dtype == "F32") width = 4;
    else if (info.dtype == "F16" || info.dtype == "BF16") width = 2;
    else if (info.dtype == "F64") width = 8;
    else {
        fail(error, "tensor '" + info.name + "' has the type " + info.dtype + ", which is not read as numbers");
        return std::nullopt;
    }
    if (info.shape.size() > tensor::Tensor::kMaxDims) {
        fail(error, "tensor '" + info.name + "' has more than 5 dimensions");
        return std::nullopt;
    }
    std::vector<int64_t> shape = info.shape;
    if (shape.empty()) shape = {1};
    int64_t count = 1;
    for (const int64_t d : shape) count *= d;
    if (static_cast<uint64_t>(count) * width != info.bytes) {
        fail(error, "tensor '" + info.name + "' has a size that does not match its shape");
        return std::nullopt;
    }
    std::ifstream in(file, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(info.offset));
    std::vector<char> raw(static_cast<size_t>(info.bytes));
    in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    if (static_cast<uint64_t>(in.gcount()) != info.bytes) {
        fail(error, "tensor '" + info.name + "' is cut off");
        return std::nullopt;
    }
    std::vector<float> values(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        const char* p = raw.data() + static_cast<size_t>(i) * width;
        if (info.dtype == "F32") {
            std::memcpy(&values[static_cast<size_t>(i)], p, 4);
        } else if (info.dtype == "F64") {
            double d;
            std::memcpy(&d, p, 8);
            values[static_cast<size_t>(i)] = static_cast<float>(d);
        } else {
            uint16_t h;
            std::memcpy(&h, p, 2);
            values[static_cast<size_t>(i)] = info.dtype == "F16" ? halfToFloat(h) : bfloatToFloat(h);
        }
    }
    return tensor::Tensor(std::move(shape), std::move(values));
}

std::map<std::string, tensor::Tensor> readAllSafetensors(const std::filesystem::path& file, std::string* error) {
    std::map<std::string, tensor::Tensor> out;
    const auto header = readSafetensorsHeader(file, error);
    if (!header) return out;
    for (const TensorInfo& info : header->tensors) {
        std::string problem;
        if (auto t = readSafetensor(file, info, &problem)) out.emplace(info.name, std::move(*t));
        else if (error && error->empty()) *error = problem;   // keep the first reason; unreadable types are skipped
    }
    return out;
}

}  // namespace gh::models
