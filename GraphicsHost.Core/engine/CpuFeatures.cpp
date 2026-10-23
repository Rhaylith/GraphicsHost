#include "engine/CpuFeatures.h"

#include <windows.h>

#include <intrin.h>

#include <array>
#include <cstring>

namespace gh::engine {

namespace {

std::array<int, 4> cpuid(int leaf, int subleaf = 0) {
    std::array<int, 4> regs{};
    __cpuidex(regs.data(), leaf, subleaf);
    return regs;
}

bool bit(int value, int index) {
    return (value >> index) & 1;
}

}  // namespace

CpuFeatures detectCpuFeatures() {
    CpuFeatures f;
    // GetSystemInfo only counts the current processor group (at most 64 threads); big CPUs have several groups.
    f.logicalCores = static_cast<int>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    if (f.logicalCores <= 0) {
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        f.logicalCores = static_cast<int>(info.dwNumberOfProcessors);
    }

    const auto leaf0 = cpuid(0);
    const int maxLeaf = leaf0[0];
    char vendor[13] = {};
    std::memcpy(vendor + 0, &leaf0[1], 4);
    std::memcpy(vendor + 4, &leaf0[3], 4);
    std::memcpy(vendor + 8, &leaf0[2], 4);
    f.vendor = vendor;

    const auto maxExtended = cpuid(static_cast<int>(0x80000000))[0];
    if (static_cast<unsigned>(maxExtended) >= 0x80000004u) {
        char brand[49] = {};
        for (int i = 0; i < 3; ++i) {
            const auto regs = cpuid(static_cast<int>(0x80000002u + static_cast<unsigned>(i)));
            std::memcpy(brand + i * 16, regs.data(), 16);
        }
        f.brand = brand;
        const auto first = f.brand.find_first_not_of(' ');
        f.brand = first == std::string::npos ? std::string() : f.brand.substr(first);
    }

    if (maxLeaf >= 1) {
        const auto leaf1 = cpuid(1);
        f.sse42 = bit(leaf1[2], 20);
        f.fma = bit(leaf1[2], 12);
        f.f16c = bit(leaf1[2], 29);
        const bool osxsave = bit(leaf1[2], 27);
        const bool avxCpu = bit(leaf1[2], 28);
        // The OS must have enabled saving the YMM (and for AVX-512 the ZMM) state.
        unsigned long long xcr0 = 0;
        if (osxsave) {
            xcr0 = _xgetbv(0);
        }
        const bool ymmEnabled = (xcr0 & 0x6) == 0x6;
        const bool zmmEnabled = (xcr0 & 0xE6) == 0xE6;
        f.avx = avxCpu && ymmEnabled;
        f.fma = f.fma && f.avx;
        f.f16c = f.f16c && f.avx;
        if (maxLeaf >= 7) {
            const auto leaf7 = cpuid(7, 0);
            f.avx2 = bit(leaf7[1], 5) && f.avx;
            f.bmi2 = bit(leaf7[1], 8);
            f.avx512f = bit(leaf7[1], 16) && zmmEnabled;
        }
    }
    return f;
}

}  // namespace gh::engine
