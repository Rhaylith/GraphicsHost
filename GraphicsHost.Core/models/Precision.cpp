#include "models/Precision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace gh::models {

using tensor::Tensor;

namespace {

struct Fp8Format {
    int expBits, manBits, bias;
    float maxFinite;
    bool hasInf;
    uint8_t nanBits;
};
constexpr Fp8Format kE4m3{4, 3, 7, 448.0f, false, 0x7F};
constexpr Fp8Format kE5m2{5, 2, 15, 57344.0f, true, 0x7E};

uint8_t maxCode(const Fp8Format& f) { return f.hasInf ? 0x7B : 0x7E; }   // the largest finite magnitude, as bits

float decodeBits(const Fp8Format& f, uint8_t bits) {
    const int sign = bits >> 7;
    const int exponent = (bits >> f.manBits) & ((1 << f.expBits) - 1);
    const int mantissa = bits & ((1 << f.manBits) - 1);
    float value;
    if (f.hasInf && exponent == (1 << f.expBits) - 1) {
        value = mantissa == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else if (!f.hasInf && exponent == (1 << f.expBits) - 1 && mantissa == (1 << f.manBits) - 1) {
        value = std::numeric_limits<float>::quiet_NaN();
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), 1 - f.bias - f.manBits);   // subnormal
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / static_cast<float>(1 << f.manBits), exponent - f.bias);
    }
    return sign ? -value : value;
}

uint8_t encodeBits(const Fp8Format& f, float value) {
    if (std::isnan(value)) return f.nanBits;
    const uint8_t sign = std::signbit(value) ? 0x80 : 0;
    float a = std::fabs(value);
    if (a >= f.maxFinite) return static_cast<uint8_t>(sign | maxCode(f));   // saturate (also for infinity)
    if (a == 0.0f) return sign;
    int e = 0;
    std::frexp(a, &e);                  // a = m * 2^e with m in [0.5, 1)
    int exponent = e - 1;               // a = 1.xxx * 2^exponent
    uint32_t code;
    if (exponent < 1 - f.bias) {
        // subnormal: a multiple of the smallest step
        const float step = std::ldexp(1.0f, 1 - f.bias - f.manBits);
        code = static_cast<uint32_t>(std::nearbyint(a / step));   // may reach 1 << manBits: that is the smallest normal number, as bits
    } else {
        const float scaled = std::ldexp(a, -exponent);   // [1, 2)
        uint32_t mantissa = static_cast<uint32_t>(std::nearbyint((scaled - 1.0f) * static_cast<float>(1 << f.manBits)));
        if (mantissa == (1u << f.manBits)) {
            mantissa = 0;
            ++exponent;
        }
        code = (static_cast<uint32_t>(exponent + f.bias) << f.manBits) | mantissa;
    }
    if (code > maxCode(f)) code = maxCode(f);
    return static_cast<uint8_t>(sign | code);
}

const std::array<float, 256>& table(const Fp8Format& f, std::array<float, 256>& storage, bool& ready) {
    if (!ready) {
        for (int i = 0; i < 256; ++i) storage[static_cast<size_t>(i)] = decodeBits(f, static_cast<uint8_t>(i));
        ready = true;
    }
    return storage;
}

}  // namespace

const char* toString(Precision p) {
    switch (p) {
        case Precision::Auto: return "auto";
        case Precision::Fp32: return "fp32";
        case Precision::Fp16: return "fp16";
        case Precision::Bf16: return "bf16";
        case Precision::Fp8E4m3: return "fp8 e4m3";
        case Precision::Fp8E5m2: return "fp8 e5m2";
    }
    return "auto";
}

std::optional<Precision> parsePrecision(const std::string& text) {
    std::string s = text;
    std::replace(s.begin(), s.end(), '_', ' ');
    if (s == "fp8 e4m3 fast") s = "fp8 e4m3";
    if (s == "default") s = "auto";
    for (const Precision p : {Precision::Auto, Precision::Fp32, Precision::Fp16, Precision::Bf16, Precision::Fp8E4m3, Precision::Fp8E5m2})
        if (s == toString(p)) return p;
    return std::nullopt;
}

size_t bytesPerElement(Precision p) {
    switch (p) {
        case Precision::Fp32: return 4;
        case Precision::Fp8E4m3:
        case Precision::Fp8E5m2: return 1;
        default: return 2;
    }
}

bool isFp8(Precision p) { return p == Precision::Fp8E4m3 || p == Precision::Fp8E5m2; }

DtypeChoice resolveDtype(const std::string& nodeValue, const std::string& settingValue) {
    DtypeChoice choice;
    choice.fast = nodeValue.find("fast") != std::string::npos;
    // "default" on a node follows the setting; anything else the node says wins
    const std::string chosen = nodeValue == "default" || nodeValue.empty() ? settingValue : nodeValue;
    choice.precision = parsePrecision(chosen).value_or(Precision::Auto);
    return choice;
}

uint8_t encodeFp8E4m3(float v) { return encodeBits(kE4m3, v); }
uint8_t encodeFp8E5m2(float v) { return encodeBits(kE5m2, v); }

float decodeFp8E4m3(uint8_t bits) {
    static std::array<float, 256> t;
    static bool ready = false;
    return table(kE4m3, t, ready)[bits];
}

float decodeFp8E5m2(uint8_t bits) {
    static std::array<float, 256> t;
    static bool ready = false;
    return table(kE5m2, t, ready)[bits];
}

uint16_t encodeBf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    if (std::isnan(value)) return static_cast<uint16_t>((bits >> 16) | 0x40);
    const uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);   // nearest, ties to even
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

float decodeBf16(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &wide, 4);
    return f;
}

uint16_t encodeFp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    if (std::isnan(value)) return static_cast<uint16_t>(sign | 0x7E00u);
    const float a = std::fabs(value);
    if (a >= 65520.0f) return static_cast<uint16_t>(sign | 0x7C00u);   // rounds to infinity
    if (a < 5.9604645e-8f * 0.5f) return static_cast<uint16_t>(sign);
    int e = 0;
    std::frexp(a, &e);
    int exponent = e - 1;
    uint32_t code;
    if (exponent < -14) {
        code = static_cast<uint32_t>(std::nearbyint(std::ldexp(a, 24)));   // subnormal: multiples of 2^-24
    } else {
        uint32_t mantissa = static_cast<uint32_t>(std::nearbyint((std::ldexp(a, -exponent) - 1.0f) * 1024.0f));
        if (mantissa == 1024u) {
            mantissa = 0;
            ++exponent;
        }
        code = (static_cast<uint32_t>(exponent + 15) << 10) | mantissa;
    }
    return static_cast<uint16_t>(sign | code);
}

float decodeFp16(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu, mantissa = h & 0x3FFu;
    float value;
    if (exponent == 0) value = std::ldexp(static_cast<float>(mantissa), -24);
    else if (exponent == 31) value = mantissa == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    else value = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, static_cast<int>(exponent) - 15);
    return sign ? -value : value;
}

PackedTensor pack(const Tensor& t, Precision precision) {
    if (precision == Precision::Auto) precision = Precision::Fp16;
    PackedTensor out;
    out.precision = precision;
    out.shape = t.shape();
    const size_t width = bytesPerElement(precision);
    out.bytes.resize(static_cast<size_t>(t.numel()) * width);
    for (int64_t i = 0; i < t.numel(); ++i) {
        const float v = t.data()[i];
        uint8_t* p = out.bytes.data() + static_cast<size_t>(i) * width;
        switch (precision) {
            case Precision::Fp32: std::memcpy(p, &v, 4); break;
            case Precision::Fp16: { const uint16_t h = encodeFp16(v); std::memcpy(p, &h, 2); break; }
            case Precision::Bf16: { const uint16_t h = encodeBf16(v); std::memcpy(p, &h, 2); break; }
            case Precision::Fp8E4m3: p[0] = encodeFp8E4m3(v); break;
            case Precision::Fp8E5m2: p[0] = encodeFp8E5m2(v); break;
            default: break;
        }
    }
    return out;
}

Tensor unpack(const PackedTensor& packed) {
    int64_t count = 1;
    for (const int64_t d : packed.shape) count *= d;
    std::vector<float> values(static_cast<size_t>(count));
    const size_t width = bytesPerElement(packed.precision);
    for (int64_t i = 0; i < count; ++i) {
        const uint8_t* p = packed.bytes.data() + static_cast<size_t>(i) * width;
        float v = 0;
        switch (packed.precision) {
            case Precision::Fp32: std::memcpy(&v, p, 4); break;
            case Precision::Fp16: { uint16_t h; std::memcpy(&h, p, 2); v = decodeFp16(h); break; }
            case Precision::Bf16: { uint16_t h; std::memcpy(&h, p, 2); v = decodeBf16(h); break; }
            case Precision::Fp8E4m3: v = decodeFp8E4m3(p[0]); break;
            case Precision::Fp8E5m2: v = decodeFp8E5m2(p[0]); break;
            default: break;
        }
        values[static_cast<size_t>(i)] = v;
    }
    return Tensor(packed.shape, std::move(values));
}

uint64_t weightsBytesAt(const CheckpointHeader& header, Precision precision) {
    if (precision == Precision::Auto) return header.fileSize;
    uint64_t total = 0;
    for (const TensorInfo& t : header.tensors) {
        const bool floating = t.dtype == "F32" || t.dtype == "F16" || t.dtype == "BF16" || t.dtype == "F64";
        total += floating ? static_cast<uint64_t>(t.elements()) * bytesPerElement(precision) : t.bytes;
    }
    return total;
}

bool hasInvalidValues(const Tensor& t) {
    for (int64_t i = 0; i < t.numel(); ++i)
        if (!std::isfinite(t.data()[i])) return true;
    return false;
}

}  // namespace gh::models
