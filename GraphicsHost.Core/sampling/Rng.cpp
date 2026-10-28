#include "sampling/Rng.h"

#include <cmath>

namespace gh::sampling {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u, kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
}  // namespace

const char* toString(RngKind kind) { return kind == RngKind::Philox ? "philox" : "cpu-compatible"; }

std::optional<RngKind> parseRng(const std::string& text) {
    if (text == "cpu-compatible") return RngKind::CpuCompatible;
    if (text == "philox") return RngKind::Philox;
    return std::nullopt;
}

void Philox4x32::block(const uint32_t counter[4], const uint32_t keyIn[2], uint32_t out[4]) {
    uint32_t c[4] = {counter[0], counter[1], counter[2], counter[3]};
    uint32_t k0 = keyIn[0], k1 = keyIn[1];
    for (int round = 0; round < 10; ++round) {
        const uint64_t p0 = static_cast<uint64_t>(kM0) * c[0];
        const uint64_t p1 = static_cast<uint64_t>(kM1) * c[2];
        const uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ c[1] ^ k0;
        const uint32_t n1 = static_cast<uint32_t>(p1);
        const uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ c[3] ^ k1;
        const uint32_t n3 = static_cast<uint32_t>(p0);
        c[0] = n0;
        c[1] = n1;
        c[2] = n2;
        c[3] = n3;
        k0 += kW0;
        k1 += kW1;
    }
    for (int i = 0; i < 4; ++i) out[i] = c[i];
}

NormalGenerator::NormalGenerator(RngKind kind, uint64_t seed) : kind_(kind), mt_(static_cast<uint32_t>(seed)), key_(seed) {}

float NormalGenerator::uniform24() {
    return static_cast<float>(mt_() & ((1u << 24) - 1)) * (1.0f / 16777216.0f);
}

void NormalGenerator::fillCpu(float* data, size_t size) {
    if (size < 16) {
        for (size_t i = 0; i < size; ++i) {
            if (cachedNormal_) {
                data[i] = static_cast<float>(*cachedNormal_);
                cachedNormal_.reset();
                continue;
            }
            const auto uniform64 = [&] {
                const uint64_t hi = mt_(), lo = mt_();
                return static_cast<double>(((hi << 32) | lo) & ((1ull << 53) - 1)) * std::ldexp(1.0, -53);
            };
            const double u1 = uniform64(), u2 = uniform64();
            const double r = std::sqrt(-2.0 * std::log1p(-u2));
            const double theta = 2.0 * kPi * u1;
            cachedNormal_ = r * std::sin(theta);
            data[i] = static_cast<float>(r * std::cos(theta));
        }
        return;
    }
    const auto fill16 = [](float* d) {
        for (int j = 0; j < 8; ++j) {
            const float u1 = 1.0f - d[j], u2 = d[j + 8];
            const float radius = std::sqrt(-2.0f * std::log(u1));
            const float theta = 2.0f * static_cast<float>(kPi) * u2;
            d[j] = radius * std::cos(theta);
            d[j + 8] = radius * std::sin(theta);
        }
    };
    for (size_t i = 0; i < size; ++i) data[i] = uniform24();
    for (size_t i = 0; i + 16 <= size; i += 16) fill16(data + i);
    if (size % 16 != 0) {   // the last 16 values are drawn again so that every block is whole
        float* tail = data + size - 16;
        for (int i = 0; i < 16; ++i) tail[i] = uniform24();
        fill16(tail);
    }
}

void NormalGenerator::fillPhilox(float* out, size_t count) {
    const uint32_t key[2] = {static_cast<uint32_t>(key_), static_cast<uint32_t>(key_ >> 32)};
    size_t produced = 0;
    while (produced < count) {
        const uint32_t counter[4] = {static_cast<uint32_t>(block_), static_cast<uint32_t>(block_ >> 32), 0, 0};
        uint32_t r[4];
        Philox4x32::block(counter, key, r);
        ++block_;
        // four words = two uniform pairs = four normals
        for (int pair = 0; pair < 2 && produced < count; ++pair) {
            const float u1 = (static_cast<float>(r[2 * pair] >> 8) + 0.5f) * (1.0f / 16777216.0f);   // in (0, 1): log never sees 0
            const float u2 = static_cast<float>(r[2 * pair + 1] >> 8) * (1.0f / 16777216.0f);
            const float radius = std::sqrt(-2.0f * std::log(u1));
            const float theta = 2.0f * static_cast<float>(kPi) * u2;
            out[produced++] = radius * std::cos(theta);
            if (produced < count) out[produced++] = radius * std::sin(theta);
        }
    }
}

void NormalGenerator::fill(float* out, size_t count) {
    if (kind_ == RngKind::Philox) fillPhilox(out, count);
    else fillCpu(out, count);
}

std::vector<float> NormalGenerator::normal(size_t count) {
    std::vector<float> v(count);
    fill(v.data(), count);
    return v;
}

}  // namespace gh::sampling
