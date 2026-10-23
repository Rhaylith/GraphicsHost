#pragma once

#include <string>

namespace gh::engine {

struct CpuFeatures {
    std::string vendor;  // "AuthenticAMD", "GenuineIntel", ...
    std::string brand;   // marketing name
    int logicalCores = 0;
    bool sse42 = false;
    bool avx = false;    // CPU supports it and the OS saves the registers
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
    bool bmi2 = false;
    bool avx512f = false;

    // What the avx2 runtime needs (ggml's AVX2 build also uses FMA and F16C).
    bool canRunAvx2Runtime() const { return avx2 && fma && f16c; }
};

CpuFeatures detectCpuFeatures();

}  // namespace gh::engine
