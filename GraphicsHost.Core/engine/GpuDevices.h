#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gh::engine {

struct GpuDevice {
    int index = 0;                 // Vulkan enumeration order
    std::string name;
    uint32_t vendorId = 0;
    std::string vendor;            // "AMD", "NVIDIA", "Intel" or the hex id
    bool discrete = false;
    uint64_t vramBytes = 0;        // sum of device-local memory heaps (shared memory on integrated GPUs)
    std::string driverVersion;     // as reported by the driver, vendor specific
    bool supported = false;        // v1 supports AMD GPUs only
    std::string note;              // why a device is not supported
};

struct GpuProbe {
    bool vulkanAvailable = false;  // a Vulkan instance could be created
    std::string error;             // why not, when vulkanAvailable is false
    std::vector<GpuDevice> devices;

    bool hasSupportedDevice() const;
};

// Asks the system's Vulkan loader (vulkan-1.dll, loaded at run time, so the app also starts without a Vulkan driver)
// for the physical devices. Never throws.
GpuProbe probeGpus();

}  // namespace gh::engine
