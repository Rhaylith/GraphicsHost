#include "host/HostJson.h"

namespace gh::host {

using Json = nlohmann::json;

Json cpuJson(const engine::CpuFeatures& c) {
    return {{"vendor", c.vendor}, {"brand", c.brand}, {"logicalCores", c.logicalCores}, {"sse42", c.sse42},
            {"avx", c.avx},       {"avx2", c.avx2},   {"fma", c.fma},                   {"f16c", c.f16c},
            {"bmi2", c.bmi2},     {"avx512f", c.avx512f}};
}

Json gpuJson(const engine::GpuProbe& g) {
    Json devices = Json::array();
    for (const auto& d : g.devices) {
        devices.push_back({{"index", d.index},
                           {"name", d.name},
                           {"vendor", d.vendor},
                           {"discrete", d.discrete},
                           {"vramBytes", d.vramBytes},
                           {"driverVersion", d.driverVersion},
                           {"supported", d.supported},
                           {"note", d.note}});
    }
    return {{"available", g.vulkanAvailable}, {"error", g.error}, {"devices", devices}};
}

Json runtimesJson(const std::vector<engine::RuntimeCandidate>& list) {
    Json out = Json::array();
    for (const auto& r : list) {
        out.push_back({{"id", r.id},
                       {"backend", r.backend},
                       {"path", r.path.string()},
                       {"bundled", r.bundled},
                       {"compatible", r.compatible},
                       {"reason", r.reason}});
    }
    return out;
}

}  // namespace gh::host
