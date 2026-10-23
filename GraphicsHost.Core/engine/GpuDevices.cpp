#include "engine/GpuDevices.h"

#include <windows.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>

namespace gh::engine {

namespace {

constexpr uint32_t kVendorAmd = 0x1002;
constexpr uint32_t kVendorNvidia = 0x10DE;
constexpr uint32_t kVendorIntel = 0x8086;

std::string vendorName(uint32_t id) {
    switch (id) {
        case kVendorAmd: return "AMD";
        case kVendorNvidia: return "NVIDIA";
        case kVendorIntel: return "Intel";
        default: {
            char buffer[16];
            std::snprintf(buffer, sizeof(buffer), "0x%04X", id);
            return buffer;
        }
    }
}

std::string resultName(VkResult r) {
    switch (r) {
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        default: return "VkResult " + std::to_string(static_cast<int>(r));
    }
}

}  // namespace

bool GpuProbe::hasSupportedDevice() const {
    return std::any_of(devices.begin(), devices.end(), [](const GpuDevice& d) { return d.supported; });
}

GpuProbe probeGpus() {
    GpuProbe probe;
    HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
    if (!loader) {
        probe.error = "vulkan-1.dll not found (no Vulkan driver installed)";
        return probe;
    }
    auto getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(loader, "vkGetInstanceProcAddr"));
    if (!getInstanceProcAddr) {
        probe.error = "vulkan-1.dll does not export vkGetInstanceProcAddr";
        FreeLibrary(loader);
        return probe;
    }
    auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(getInstanceProcAddr(nullptr, "vkCreateInstance"));
    if (!createInstance) {
        probe.error = "vkCreateInstance is not available";
        FreeLibrary(loader);
        return probe;
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "GraphicsHost";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult created = createInstance(&create, nullptr, &instance);
    if (created != VK_SUCCESS || instance == VK_NULL_HANDLE) {
        probe.error = "vkCreateInstance failed: " + resultName(created);
        FreeLibrary(loader);
        return probe;
    }

    auto destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(getInstanceProcAddr(instance, "vkDestroyInstance"));
    auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
        getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
    auto getProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        getInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties"));
    auto getMemory = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        getInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties"));

    probe.vulkanAvailable = true;
    if (enumerate && getProperties && getMemory) {
        uint32_t count = 0;
        if (enumerate(instance, &count, nullptr) == VK_SUCCESS && count > 0) {
            std::vector<VkPhysicalDevice> handles(count);
            enumerate(instance, &count, handles.data());
            handles.resize(count);
            for (uint32_t i = 0; i < count; ++i) {
                VkPhysicalDeviceProperties props{};
                VkPhysicalDeviceMemoryProperties memory{};
                getProperties(handles[i], &props);
                getMemory(handles[i], &memory);

                GpuDevice device;
                device.index = static_cast<int>(i);
                device.name = props.deviceName;
                device.vendorId = props.vendorID;
                device.vendor = vendorName(props.vendorID);
                device.discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                for (uint32_t h = 0; h < memory.memoryHeapCount; ++h) {
                    if (memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                        device.vramBytes += memory.memoryHeaps[h].size;
                    }
                }
                device.driverVersion = std::to_string(VK_API_VERSION_MAJOR(props.driverVersion)) + "." +
                                       std::to_string(VK_API_VERSION_MINOR(props.driverVersion)) + "." +
                                       std::to_string(VK_API_VERSION_PATCH(props.driverVersion));
                const bool software = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
                device.supported = props.vendorID == kVendorAmd && !software;
                if (software) {
                    device.note = "software renderer";
                } else if (props.vendorID != kVendorAmd) {
                    device.note = "unsupported in this version (AMD GPUs only)";
                }
                probe.devices.push_back(std::move(device));
            }
        }
    }
    if (destroyInstance) {
        destroyInstance(instance, nullptr);
    }
    FreeLibrary(loader);
    if (probe.devices.empty()) {
        probe.error = "Vulkan is available but reports no physical devices";
    }
    return probe;
}

}  // namespace gh::engine
