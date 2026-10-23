#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace gh::host {

// One reading of how busy this computer is.
struct HardwareUsage {
    double cpuPercent = 0;                       // all logical cores, since the previous reading
    uint64_t ramUsedBytes = 0;
    uint64_t ramTotalBytes = 0;
    std::optional<double> gpuPercent;            // 3D / compute engines of the busiest GPU; empty when Windows gives no counter
    std::optional<uint64_t> vramUsedBytes;       // dedicated memory in use on the GPU with the most of it
};

// Reads CPU and memory from the system and GPU load and video memory from the Windows performance counters ("GPU Engine"
// and "GPU Adapter Memory"). The first reading after construction has no CPU delta yet and reports 0.
class HardwareMonitor {
public:
    HardwareMonitor();
    ~HardwareMonitor();
    HardwareMonitor(const HardwareMonitor&) = delete;
    HardwareMonitor& operator=(const HardwareMonitor&) = delete;

    HardwareUsage sample();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::mutex mutex_;
};

}  // namespace gh::host
