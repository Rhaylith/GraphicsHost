#include "cli/Cli.h"

#include "cli/Commands.h"
#include "core/Version.h"
#include "host/AppHost.h"
#include "host/HostJson.h"

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
using host::cpuJson;
using host::gpuJson;
using host::runtimesJson;

std::string gib(uint64_t bytes) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB";
    return s.str();
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
        if (!coerceJsonValue(document[pointer], args[1], value, problem)) {
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
        case engine::EngineErrorCode::Blocked: return kUsage;   // the request was refused
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
        if (arg == "--screenshot-dir" || arg == "--generate-prompt") {
            ++i;   // its value is the folder
        } else if (arg != "--smoke-test" && arg != "gui") {
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

    bool quiet = false;
    app.add_flag("-q,--quiet", quiet, "no progress or notes on stderr");

    // gen: only the options the user actually gives become part of the request, so presets and model defaults can fill the rest
    auto* gen = app.add_subcommand("gen", "generate images from a prompt");
    GenInvocation inv;
    std::string size, sampler, scheduler, negative, format, device, vae;
    int steps = 0, batch = 1, clipSkip = -1, threads = 0;
    int64_t seed = -1;
    double cfg = 0;
    bool vaeTiling = false, noMetadata = false, offload = false, flashAttn = false, clipOnCpu = false, vaeOnCpu = false;
    gen->add_option("prompt", inv.prompt, "what to draw")->required();
    gen->add_option("-m,--model", inv.model, "checkpoint file, or a name in the models folder (see: ls)");
    gen->add_option("--preset", inv.preset, "named parameter set (see: preset list)");
    auto* oSize = gen->add_option("--size", size, "WIDTHxHEIGHT, multiples of 8");
    auto* oSteps = gen->add_option("-s,--steps", steps, "sampling steps");
    auto* oCfg = gen->add_option("--cfg", cfg, "guidance scale");
    auto* oSeed = gen->add_option("--seed", seed, "-1 = random (the real seed is always reported)");
    auto* oNeg = gen->add_option("-n,--negative", negative, "negative prompt");
    auto* oBatch = gen->add_option("-b,--batch", batch, "number of images");
    auto* oSampler = gen->add_option("--sampler", sampler, "euler_a, dpm++2m, ...");
    auto* oScheduler = gen->add_option("--scheduler", scheduler, "karras, discrete, ...");
    auto* oClip = gen->add_option("--clip-skip", clipSkip, "CLIP layers to skip");
    auto* oFormat = gen->add_option("--format", format, "png | jpeg | webp");
    auto* fTiling = gen->add_flag("--vae-tiling", vaeTiling, "decode in tiles: slower, uses much less memory");
    auto* fNoMeta = gen->add_flag("--no-metadata", noMetadata, "do not write the parameters into the image file");
    auto* oDevice = gen->add_option("--device", device, "auto | cpu | vulkan0");
    auto* oThreads = gen->add_option("--threads", threads, "CPU threads, 0 = automatic");
    auto* oVae = gen->add_option("--vae", vae, "separate VAE file");
    auto* fOffload = gen->add_flag("--offload", offload, "keep the weights in RAM (low GPU memory)");
    auto* fFlash = gen->add_flag("--flash-attn", flashAttn, "flash attention (much faster for SD 1.x above 512)");
    auto* fClipCpu = gen->add_flag("--clip-on-cpu", clipOnCpu, "run the text encoder on the CPU");
    auto* fVaeCpu = gen->add_flag("--vae-on-cpu", vaeOnCpu, "run the VAE on the CPU");
    gen->add_option("-o,--out", inv.outPath, "write the image(s) here: a file (one image) or a folder");
    gen->add_option("--quality", inv.quality, "JPEG / lossy WebP quality 1-100")->capture_default_str();
    gen->add_flag("--dry-run", inv.dryRun, "show the resolved parameters and the memory estimate, do not generate");
    gen->add_flag("--no-gallery", inv.noGallery, "do not add the images to the gallery");
    gen->add_flag("--no-fit-check", inv.noFitCheck, "skip the memory estimate and its automatic adjustments");

    auto* ls = app.add_subcommand("ls", "list the checkpoints in the models folder");

    auto* preset = app.add_subcommand("preset", "named parameter sets");
    std::string presetAction = "list";
    std::vector<std::string> presetArgs;
    preset->add_option("action", presetAction, "list | show | set | delete | import | export")->capture_default_str();
    preset->add_option("args", presetArgs, "arguments of the action");

    auto* defaults = app.add_subcommand("defaults", "default parameters per checkpoint file");
    std::string defaultsAction = "list";
    std::vector<std::string> defaultsArgs;
    defaults->add_option("action", defaultsAction, "list | show | set | clear")->capture_default_str();
    defaults->add_option("args", defaultsArgs, "arguments of the action");

    auto* galleryCmd = app.add_subcommand("gallery", "the images generated so far");
    std::string galleryAction = "list";
    std::vector<std::string> galleryArgs;
    int galleryLimit = 20;
    galleryCmd->add_option("action", galleryAction, "list | show | rm | rescan")->capture_default_str();
    galleryCmd->add_option("args", galleryArgs, "list: text to search for; show, rm: the image id");
    galleryCmd->add_option("--limit", galleryLimit, "list: how many")->capture_default_str();

    // model discovery and downloads
    auto* catalogCmd = app.add_subcommand("catalog", "the curated list of models");
    std::string catalogText, catalogFamily, catalogKind;
    catalogCmd->add_option("text", catalogText, "search text (name, family, tag)");
    catalogCmd->add_option("--family", catalogFamily, "SD 1.x or SDXL");
    catalogCmd->add_option("--kind", catalogKind, "checkpoint | vae | lora");
    auto* searchCmd = app.add_subcommand("search", "search Hugging Face for text-to-image models");
    std::string searchText;
    int searchLimit = 20;
    searchCmd->add_option("text", searchText, "what to look for")->required();
    searchCmd->add_option("--limit", searchLimit, "how many results")->capture_default_str();
    auto* infoCmd = app.add_subcommand("info", "the usable files in a Hugging Face repository");
    std::string infoRepo;
    infoCmd->add_option("repository", infoRepo, "publisher/name")->required();
    auto* getCmd = app.add_subcommand("get", "download a model: a catalog key, or a repository and file");
    GetInvocation getInv;
    getCmd->add_option("target", getInv.target, "catalog key, or publisher/name")->required();
    getCmd->add_option("file", getInv.file, "for a repository: the file to download");
    getCmd->add_flag("--no-companions", getInv.noCompanions, "do not download the VAE that belongs to a catalog model");
    getCmd->add_flag("--dry-run", getInv.dryRun, "show what would be downloaded");
    auto* downloadsCmd = app.add_subcommand("downloads", "interrupted downloads");
    std::string downloadsAction = "list";
    downloadsCmd->add_option("action", downloadsAction, "list | clear")->capture_default_str();
    auto* updatesCmd = app.add_subcommand("updates", "check the downloaded models for newer versions");
    auto* loginCmd = app.add_subcommand("login", "store a Hugging Face access token (needed for gated models)");
    std::string loginToken;
    loginCmd->add_option("token", loginToken, "the token; asked for when left out");
    auto* logoutCmd = app.add_subcommand("logout", "remove the stored Hugging Face token");
    auto* whoamiCmd = app.add_subcommand("whoami", "the Hugging Face user of the stored token");

    auto* serveCmd = app.add_subcommand("serve", "run the HTTP server (API and browser UI) in this console, without a window");
    int servePort = -1;
    std::string serveBind;
    serveCmd->add_option("--port", servePort, "port (default: the server.port setting; 0 = any free port)");
    serveCmd->add_option("--bind", serveBind, "address to listen on (default 127.0.0.1, or 0.0.0.0 with the lan setting)");
    bool serveAsService = false;
    serveCmd->add_flag("--service", serveAsService, "run as a Windows service (started by Windows, see: graphicshost service install)");
    auto* serviceCmd = app.add_subcommand("service", "install or remove the Windows service that runs the server at start-up");
    std::string serviceAction = "status";
    serviceCmd->add_option("action", serviceAction, "status | install | uninstall")->capture_default_str();
    auto* tokenCmd = app.add_subcommand("token", "API access tokens for the HTTP server");
    std::string tokenAction = "list", tokenPerms;
    std::vector<std::string> tokenArgs;
    tokenCmd->add_option("action", tokenAction, "create | list | revoke")->capture_default_str();
    tokenCmd->add_option("args", tokenArgs, "create: the name; revoke: the id or name");
    tokenCmd->add_option("--perm", tokenPerms, "create: read,generate,manage,admin (default read,generate)");

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
        if (logoutCmd->parsed()) return runLogout(json, out, err);
        if (serveCmd->parsed()) return runServe(host, servePort, serveBind, serveAsService, json, out, err);
        if (serviceCmd->parsed()) return runService(host, serviceAction, json, out, err);
        if (tokenCmd->parsed()) return runToken(host, tokenAction, tokenArgs, tokenPerms, json, out, err);
        if (catalogCmd->parsed()) return runCatalog(host, catalogText, catalogFamily, catalogKind, json, out, err);
        if (searchCmd->parsed()) return runSearch(host, searchText, searchLimit, json, out, err);
        if (infoCmd->parsed()) return runInfo(host, infoRepo, json, out, err);
        if (getCmd->parsed()) return runGet(host, getInv, json, quiet, out, err);
        if (downloadsCmd->parsed()) return runDownloads(host, downloadsAction, json, out, err);
        if (updatesCmd->parsed()) return runUpdates(host, json, out, err);
        if (loginCmd->parsed()) return runLogin(host, loginToken, json, out, err);
        if (whoamiCmd->parsed()) return runWhoami(host, json, out, err);
        if (ls->parsed()) {
            return runLs(host, json, out, err);
        }
        if (preset->parsed()) {
            return runPreset(host, presetAction, presetArgs, json, out, err);
        }
        if (defaults->parsed()) {
            return runDefaults(host, defaultsAction, defaultsArgs, json, out, err);
        }
        if (galleryCmd->parsed()) {
            return runGallery(host, galleryAction, galleryArgs, galleryLimit, json, out, err);
        }
        if (gen->parsed()) {
            auto given = [](CLI::Option* o) { return o->count() > 0; };
            if (given(oSize)) {
                int w = 0, h = 0;
                char x = 0;
                std::istringstream parts(size);
                if (!(parts >> w >> x >> h) || (x != 'x' && x != 'X')) {
                    err << "error: --size must look like 512x768\n";
                    return kUsage;
                }
                inv.request["width"] = w;
                inv.request["height"] = h;
            }
            if (given(oSteps)) inv.request["steps"] = steps;
            if (given(oCfg)) inv.request["cfgScale"] = cfg;
            if (given(oSeed)) inv.request["seed"] = seed;
            if (given(oNeg)) inv.request["negativePrompt"] = negative;
            if (given(oBatch)) inv.request["batchCount"] = batch;
            if (given(oSampler)) inv.request["sampler"] = sampler;
            if (given(oScheduler)) inv.request["scheduler"] = scheduler;
            if (given(oClip)) inv.request["clipSkip"] = clipSkip;
            if (given(oFormat)) inv.request["format"] = format;
            if (given(fTiling)) inv.request["vaeTiling"] = true;
            if (given(fNoMeta)) inv.request["embedMetadata"] = false;
            if (given(oDevice)) inv.load["device"] = device;
            if (given(oThreads)) inv.load["threads"] = threads;
            if (given(oVae)) inv.load["vaePath"] = vae;
            if (given(fOffload)) inv.load["offloadParamsToCpu"] = true;
            if (given(fFlash)) inv.load["flashAttention"] = true;
            if (given(fClipCpu)) inv.load["clipOnCpu"] = true;
            if (given(fVaeCpu)) inv.load["vaeOnCpu"] = true;
            return runGen(host, inv, json, quiet, out, err);
        }
    } catch (const std::exception& e) {
        err << "error: " << e.what() << "\n";
        return kFailure;
    }
    err << "error: no command given\n";
    return kUsage;
}

}  // namespace gh::cli
