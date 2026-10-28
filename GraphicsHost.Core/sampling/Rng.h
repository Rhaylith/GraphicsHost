#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace gh::sampling {

// The two random generators of the sampling code. Noise is always made on the host, so a seed gives the same noise on every
// GPU.
//   cpu-compatible  mt19937 seeded with the (32-bit) seed and turned into normal numbers the way the CPU generator of the common
//                   tensor libraries does it: blocks of 16 uniform 24-bit numbers through Box-Muller, the last block overlapping
//                   when the count is not a multiple of 16, and for fewer than 16 numbers the scalar path with a cached second
//                   value.
//   philox          Philox-4x32-10 (Salmon et al. 2011), counter based: the seed is the key, four 32-bit words per block, two
//                   uniform numbers per Box-Muller pair.
enum class RngKind { CpuCompatible, Philox };
const char* toString(RngKind kind);
std::optional<RngKind> parseRng(const std::string& text);   // "cpu-compatible", "philox"

// The Philox-4x32-10 block function, exposed for its known-answer test.
struct Philox4x32 {
    static void block(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]);
};

// A stream of standard normal numbers. Calls continue where the previous one stopped.
class NormalGenerator {
public:
    NormalGenerator(RngKind kind, uint64_t seed);
    void fill(float* out, size_t count);
    std::vector<float> normal(size_t count);
    RngKind kind() const { return kind_; }

private:
    float uniform24();
    void fillCpu(float* out, size_t count);
    void fillPhilox(float* out, size_t count);

    RngKind kind_;
    std::mt19937 mt_;
    std::optional<double> cachedNormal_;
    uint64_t key_ = 0;
    uint64_t block_ = 0;
};

}  // namespace gh::sampling
