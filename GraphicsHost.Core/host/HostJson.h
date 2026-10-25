#pragma once

#include "engine/CpuFeatures.h"
#include "engine/GpuDevices.h"
#include "engine/RuntimeLoader.h"

#include <nlohmann/json.hpp>

#include <vector>

namespace gh::host {

// The JSON the CLI (hardware --json) and the HTTP API (/api/v1/hardware) both give for the machine.
nlohmann::json cpuJson(const engine::CpuFeatures& cpu);
nlohmann::json gpuJson(const engine::GpuProbe& gpu);
nlohmann::json runtimesJson(const std::vector<engine::RuntimeCandidate>& runtimes);

}  // namespace gh::host
