#include "host/HardwareMonitor.h"

#include <windows.h>

#include <pdh.h>
#include <pdhmsg.h>

#include <algorithm>
#include <string>
#include <vector>

namespace gh::host {

namespace {

uint64_t ticks(const FILETIME& f) {
    return (static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
}

// Formatted values of a wildcard counter: (instance name, value) pairs. Empty when the counter has nothing.
std::vector<std::pair<std::wstring, double>> readArray(PDH_HCOUNTER counter) {
    std::vector<std::pair<std::wstring, double>> out;
    if (!counter) return out;
    DWORD bytes = 0, count = 0;
    PDH_STATUS status = PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &bytes, &count, nullptr);
    if (status != PDH_MORE_DATA || bytes == 0) return out;
    std::vector<uint8_t> buffer(bytes);
    auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
    status = PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &bytes, &count, items);
    if (status != ERROR_SUCCESS) return out;
    for (DWORD i = 0; i < count; ++i) {
        if (items[i].FmtValue.CStatus == PDH_CSTATUS_VALID_DATA || items[i].FmtValue.CStatus == PDH_CSTATUS_NEW_DATA) {
            out.emplace_back(items[i].szName, items[i].FmtValue.doubleValue);
        }
    }
    return out;
}

}  // namespace

struct HardwareMonitor::Impl {
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER engine = nullptr;
    PDH_HCOUNTER memory = nullptr;
    bool primed = false;
    uint64_t lastIdle = 0, lastTotal = 0;
    bool haveCpu = false;

    Impl() {
        if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) {
            query = nullptr;
            return;
        }
        if (PdhAddEnglishCounterW(query, L"\\GPU Engine(*)\\Utilization Percentage", 0, &engine) != ERROR_SUCCESS) engine = nullptr;
        if (PdhAddEnglishCounterW(query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &memory) != ERROR_SUCCESS) memory = nullptr;
        if (engine || memory) PdhCollectQueryData(query);   // rate counters need a first reading to compare with
    }
    ~Impl() {
        if (query) PdhCloseQuery(query);
    }
};

HardwareMonitor::HardwareMonitor() : impl_(std::make_unique<Impl>()) {}
HardwareMonitor::~HardwareMonitor() = default;

HardwareUsage HardwareMonitor::sample() {
    std::lock_guard<std::mutex> lock(mutex_);
    HardwareUsage usage;

    FILETIME idle, kernel, user;
    if (GetSystemTimes(&idle, &kernel, &user)) {
        const uint64_t i = ticks(idle), total = ticks(kernel) + ticks(user);   // kernel time includes idle time
        if (impl_->haveCpu && total > impl_->lastTotal) {
            const double busy = static_cast<double>((total - impl_->lastTotal) - (i - impl_->lastIdle));
            usage.cpuPercent = std::clamp(100.0 * busy / static_cast<double>(total - impl_->lastTotal), 0.0, 100.0);
        }
        impl_->lastIdle = i;
        impl_->lastTotal = total;
        impl_->haveCpu = true;
    }

    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) {
        usage.ramTotalBytes = memory.ullTotalPhys;
        usage.ramUsedBytes = memory.ullTotalPhys - memory.ullAvailPhys;
    }

    if (impl_->query && (impl_->engine || impl_->memory) && PdhCollectQueryData(impl_->query) == ERROR_SUCCESS) {
        // GPU load: per adapter (the "luid_..." part of the instance name) add up the 3D and compute engines; the busiest
        // adapter counts
        std::vector<std::pair<std::wstring, double>> perAdapter;
        for (const auto& [name, value] : readArray(impl_->engine)) {
            if (name.find(L"engtype_3D") == std::wstring::npos && name.find(L"engtype_Compute") == std::wstring::npos) continue;
            const auto engPos = name.find(L"_eng_");
            const std::wstring adapter = engPos == std::wstring::npos ? name : name.substr(0, engPos);
            auto it = std::find_if(perAdapter.begin(), perAdapter.end(), [&](const auto& p) { return p.first == adapter; });
            if (it == perAdapter.end()) perAdapter.emplace_back(adapter, value);
            else it->second += value;
        }
        if (!perAdapter.empty()) {
            double busiest = 0;
            for (const auto& p : perAdapter) busiest = std::max(busiest, p.second);
            usage.gpuPercent = std::clamp(busiest, 0.0, 100.0);
        }
        double vram = -1;
        for (const auto& [name, value] : readArray(impl_->memory)) vram = std::max(vram, value);
        if (vram >= 0) usage.vramUsedBytes = static_cast<uint64_t>(vram);
    }
    return usage;
}

}  // namespace gh::host
