#pragma once

#include "models/CheckpointHeader.h"
#include "tensor/Tensor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gh::models {

// The number format a component's weights are held in (settings precision.*, the weightDtype input of the loader nodes).
// fp8 is a storage format: the weights are packed to one byte each and converted back to a wider type one layer at a time
// when they are used.
enum class Precision { Auto, Fp32, Fp16, Bf16, Fp8E4m3, Fp8E5m2 };

const char* toString(Precision p);                              // "auto", "fp32", "fp16", "bf16", "fp8 e4m3", "fp8 e5m2"
std::optional<Precision> parsePrecision(const std::string& s);  // also accepts "fp8_e4m3", "default" (= auto) and "fp8 e4m3 fast" (= fp8 e4m3)
size_t bytesPerElement(Precision p);                            // Auto counts as 2 (the usual file type)
bool isFp8(Precision p);

// A weight dtype as the loader nodes and settings write it: "default" means "follow the setting"; "auto" means "keep the
// file's own format"; "fp8 e4m3 fast" asks for fp8 arithmetic where the GPU has it (reported through `fast`).
struct DtypeChoice {
    Precision precision = Precision::Auto;
    bool fast = false;
};
DtypeChoice resolveDtype(const std::string& nodeValue, const std::string& settingValue);

// ---- the number formats ---------------------------------------------------------------------------------------------------
// Rounding is to nearest, ties to even. Values beyond the largest finite number saturate to it (a weight that is too large
// becomes the largest weight, not an infinity); NaN stays NaN.
uint8_t encodeFp8E4m3(float value);    // 1 sign, 4 exponent (bias 7), 3 mantissa bits; no infinities; largest finite 448
uint8_t encodeFp8E5m2(float value);    // 1 sign, 5 exponent (bias 15), 2 mantissa bits; largest finite 57344
float decodeFp8E4m3(uint8_t bits);
float decodeFp8E5m2(uint8_t bits);
uint16_t encodeFp16(float value);
uint16_t encodeBf16(float value);
float decodeFp16(uint16_t bits);
float decodeBf16(uint16_t bits);

// A tensor in its storage format.
struct PackedTensor {
    Precision precision = Precision::Fp32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
    size_t sizeBytes() const { return bytes.size(); }
};
PackedTensor pack(const tensor::Tensor& t, Precision precision);   // Auto packs as fp16
tensor::Tensor unpack(const PackedTensor& packed);

// What the weights of a file occupy when held in `precision`: floating-point tensors take the new width, other types keep their
// size. Auto keeps everything as it is.
uint64_t weightsBytesAt(const CheckpointHeader& header, Precision precision);

// True when a result contains NaN or infinite values (the sign of a decode that overflowed in fp16: retry in fp32).
bool hasInvalidValues(const tensor::Tensor& t);

}  // namespace gh::models
