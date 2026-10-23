#include "cli/Cli.h"

#include "core/Version.h"
#include "host/AppHost.h"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace gh::cli {

namespace {

using Json = nlohmann::json;

std::string gib(uint64_t bytes) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB";
    return s.str();
}

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

int cmdVersion(bool json, std::ostream& out) {
    if (json) {
        out << Json{{"name", kAppName}, {"version", kAppVersion}, {"runtimeAbi", GH_ABI_VERSION}}.dump(2) << "\n";
    } else {
        out << kAppName << " " << kAppVersion << " (runtime ABI " << GH_ABI_VERSION << ")\n";
    }
    return kOk;
}

int cmdHardware(const host::AppHost& host, bool json, std::ostream& out) {
    const auto* selected = host.selectedRuntime();
    if (json) {
        out << Json{{"cpu", cpuJson(host.cpu())},
                    {"vulkan", gpuJson(host.gpu())},
                    {"runtimes", runtimesJson(host.runtimes())},
                    {"backendSetting", host.settings().backend},
                    {"selectedRuntime", selected ? Json(selected->id) : Json(nullptr)}}
                   .dump(2)
            << "\n";
        return kOk;
    }
    const auto& c = host.cpu();
    out << "CPU      " << c.brand << " (" << c.logicalCores << " logical cores)\n";
    out << "         SSE4.2 " << c.sse42 << "  AVX " << c.avx << "  AVX2 " << c.avx2 << "  FMA " << c.fma << "  F16C "
        << c.f16c << "  AVX-512F " << c.avx512f << "\n";
    const auto& g = host.gpu();
    if (!g.vulkanAvailable) {
        out << "Vulkan   not available: " << g.error << "\n";
    } else if (g.devices.empty()) {
        out << "Vulkan   available, but no devices: " << g.error << "\n";
    } else {
        for (const auto& d : g.devices) {
            out << "GPU " << d.index << "    " << d.name << "  [" << d.vendor << "]  " << gib(d.vramBytes) << "  "
                << (d.supported ? "supported" : d.note) << "\n";
        }
    }
    out << "Runtimes\n";
    if (host.runtimes().empty()) {
        out << "         none found next to the executable or in " << host.paths().runtimesDir().string() << "\n";
    }
    for (const auto& r : host.runtimes()) {
        out << "         " << std::left << std::setw(10) << r.id << (r.compatible ? "ok" : "not usable: " + r.reason)
            << (r.bundled ? "" : "  (user installed)") << "\n";
    }
    out << "Backend  setting '" << host.settings().backend << "' -> "
        << (selected ? selected->id : std::string("no runtime available")) << "\n";
    return kOk;
}

int cmdRuntimeInfo(const host::AppHost& host, bool json, std::ostream& out, std::ostream& err) {
    std::string error;
    auto runtime = host.loadSelectedRuntime(error);
    if (!runtime) {
        err << "error: " << error << "\n";
        return host.selectedRuntime() ? kFailure : kNotFound;
    }
    if (json) {
        out << Json{{"id", runtime->id()},
                    {"backend", runtime->backend()},
                    {"engineVersion", runtime->engineVersion()},
                    {"systemInfo", runtime->systemInfo()},
                    {"devices", runtime->listDevices()}}
                   .dump(2)
            << "\n";
    } else {
        out << "Runtime  " << runtime->id() << " (" << runtime->backend() << ")\n";
        out << "Engine   " << runtime->engineVersion() << "\n";
        out << "System   " << runtime->systemInfo() << "\n";
        out << "Devices\n" << runtime->listDevices() << "\n";
    }
    return kOk;
}

// "server.port" -> "/server/port"
Json::json_pointer pointerFor(const std::string& dotted) {
    std::string p = "/";
    for (char ch : dotted) {
        p += ch == '.' ? '/' : ch;
    }
    return Json::json_pointer(p);
}

bool coerce(const Json& current, const std::string& text, Json& out, std::string& problem) {
    try {
        if (current.is_boolean()) {
            if (text == "true" || text == "1" || text == "on" || text == "yes") out = true;
            else if (text == "false" || text == "0" || text == "off" || text == "no") out = false;
            else {
                problem = "expected true or false";
                return false;
            }
        } else if (current.is_number_integer()) {
            size_t used = 0;
            const int value = std::stoi(text, &used);
            if (used != text.size()) {
                problem = "expected a whole number";
                return false;
            }
            out = value;
        } else {
            out = text;
        }
        return true;
    } catch (const std::exception&) {
        problem = "expected a whole number";
        return false;
    }
}

int cmdConfig(host::AppHost& host, const std::string& action, const std::vector<std::string>& args, bool json,
              std::ostream& out, std::ostream& err) {
    if (action == "path") {
        out << host.paths().settingsFile().string() << "\n";
        return kOk;
    }
    Json document = core::toJson(host.settings());
    if (action == "show") {
        if (json) {
            document.erase("unknown");
            out << document.dump(2) << "\n";
        } else {
            for (const auto& [key, value] : document.items()) {
                if (value.is_object()) {
                    for (const auto& [subKey, subValue] : value.items()) {
                        out << key << "." << subKey << " = " << subValue.dump() << "\n";
                    }
                } else {
                    out << key << " = " << value.dump() << "\n";
                }
            }
        }
        return kOk;
    }
    if (action == "set") {
        if (args.size() != 2) {
            err << "usage: graphicshost config set <key> <value>\n";
            return kUsage;
        }
        const auto pointer = pointerFor(args[0]);
        if (args[0] == "schemaVersion" || !document.contains(pointer)) {
            err << "error: unknown or read-only setting '" << args[0] << "'\n";
            return kNotFound;
        }
        Json value;
        std::string problem;
        if (!coerce(document[pointer], args[1], value, problem)) {
            err << "error: " << args[0] << ": " << problem << "\n";
            return kUsage;
        }
        document[pointer] = value;
        core::Settings updated;
        try {
            updated = core::settingsFromJson(document);
        } catch (const std::exception& e) {
            err << "error: " << e.what() << "\n";
            return kUsage;
        }
        const auto errors = core::validate(updated);
        if (!errors.empty()) {
            for (const auto& e : errors) {
                err << "error: " << e.field << " " << e.message << "\n";
            }
            return kUsage;
        }
        host.updateSettings(std::move(updated));
        out << args[0] << " = " << value.dump() << "\n";
        return kOk;
    }
    err << "usage: graphicshost config show|path|set <key> <value>\n";
    return kUsage;
}

}  // namespace

int exitCodeFor(engine::EngineErrorCode code) {
    switch (code) {
        case engine::EngineErrorCode::InvalidRequest: return kUsage;
        case engine::EngineErrorCode::NotFound: return kNotFound;
        case engine::EngineErrorCode::OutOfMemory: return kOutOfMemory;
        case engine::EngineErrorCode::Cancelled: return kCancelled;
        case engine::EngineErrorCode::LoadFailed:
        case engine::EngineErrorCode::Failed: return kFailure;
    }
    return kFailure;
}

bool wantsGui(int argc, char** argv) {
    if (argc <= 1) {
        return true;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg != "--smoke-test" && arg != "gui") {
            return false;
        }
    }
    return true;
}

bool wantsCli(int argc, char** argv) {
    return !wantsGui(argc, argv);
}

int run(int argc, char** argv, std::ostream& out, std::ostream& err, std::optional<std::filesystem::path> home) {
    CLI::App app{"GraphicsHost - local image generation host", "graphicshost"};
    app.require_subcommand(1);
    app.set_version_flag("--version", std::string(kAppName) + " " + kAppVersion);

    bool json = false;
    std::string homeOverride;
    app.add_flag("--json", json, "print results as JSON");
    app.add_option("--home", homeOverride, "data folder (default: %USERPROFILE%\\.graphicshost)");

    auto* version = app.add_subcommand("version", "show the version");
    auto* hardware = app.add_subcommand("hardware", "show CPU, GPUs, runtimes and which runtime is selected");
    auto* runtime = app.add_subcommand("runtime", "work with the inference runtime");
    runtime->require_subcommand(1);
    auto* runtimeInfo = runtime->add_subcommand("info", "load the selected runtime and show what it reports");
    auto* config = app.add_subcommand("config", "show or change settings");
    std::string configAction = "show";
    std::vector<std::string> configArgs;
    config->add_option("action", configAction, "show | path | set")->capture_default_str();
    config->add_option("args", configArgs, "for set: <key> <value>");
    app.add_subcommand("gui", "open the desktop window");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::ostringstream text;
        const int code = app.exit(e, text, text);
        (code == 0 ? out : err) << text.str();
        return code == 0 ? kOk : kUsage;
    }

    std::optional<std::filesystem::path> effectiveHome = home;
    if (!homeOverride.empty()) {
        effectiveHome = std::filesystem::path(homeOverride);
    }

    try {
        if (version->parsed()) {
            return cmdVersion(json, out);
        }
        host::AppHost host(effectiveHome);
        if (host.settingsWarning()) {
            err << "warning: " << *host.settingsWarning() << "\n";
        }
        if (hardware->parsed()) {
            return cmdHardware(host, json, out);
        }
        if (runtimeInfo->parsed()) {
            return cmdRuntimeInfo(host, json, out, err);
        }
        if (config->parsed()) {
            return cmdConfig(host, configAction, configArgs, json, out, err);
        }
    } catch (const std::exception& e) {
        err << "error: " << e.what() << "\n";
        return kFailure;
    }
    err << "error: no command given\n";
    return kUsage;
}

}  // namespace gh::cli
