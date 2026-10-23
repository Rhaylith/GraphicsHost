#include "cli/Commands.h"

#include "cli/Cli.h"
#include "core/AtomicFile.h"
#include "core/Log.h"
#include "core/Version.h"
#include "engine/FitEstimator.h"
#include "engine/ImageCodec.h"
#include "engine/LoadAdvice.h"
#include "models/ModelLibrary.h"
#include "presets/PresetStore.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace gh::cli {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

std::string humanSize(uint64_t bytes) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB";
    return s.str();
}

// A path, or a name (with or without extension, or a unique prefix) in the models folder.
bool resolveModel(host::AppHost& host, const std::string& arg, models::ModelCard& card, std::string& problem, int& code) {
    if (arg.empty()) {
        problem = "give the model with --model <file or name> (see: graphicshost ls)";
        code = kUsage;
        return false;
    }
    const auto r = host.library().resolve(arg);
    if (r.status == models::ModelLibrary::Resolution::Status::Found) {
        card = *r.card;
        return true;
    }
    problem = r.problem;
    code = r.status == models::ModelLibrary::Resolution::Status::Ambiguous ? kUsage : kNotFound;
    return false;
}

std::string errorList(const std::vector<core::FieldError>& errors) {
    std::string text;
    for (const auto& e : errors) {
        text += "  " + e.field + " " + e.message + "\n";
    }
    return text;
}

// ---- Ctrl+C cancels the running generation ------------------------------------------------------------------------
std::atomic<engine::Job*> g_runningJob{nullptr};

BOOL WINAPI onConsoleControl(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
        if (engine::Job* job = g_runningJob.load()) {
            job->cancel();
            return TRUE;  // handled: the generation stops at its next step and the command ends normally
        }
    }
    return FALSE;
}

Json fitJson(const engine::FitReport& r) {
    return {{"status", engine::toString(r.status)}, {"device", r.device},        {"requiredBytes", r.requiredBytes},
            {"availableBytes", r.availableBytes},   {"needsOffload", r.needsOffload}, {"summary", r.summary},
            {"suggestions", r.suggestions}};
}

}  // namespace

bool coerceJsonValue(const Json& sample, const std::string& text, Json& out, std::string& problem) {
    try {
        if (sample.is_boolean()) {
            if (text == "true" || text == "1" || text == "on" || text == "yes") out = true;
            else if (text == "false" || text == "0" || text == "off" || text == "no") out = false;
            else {
                problem = "expected true or false";
                return false;
            }
        } else if (sample.is_number_integer()) {
            size_t used = 0;
            const long long value = std::stoll(text, &used);
            if (used != text.size()) {
                problem = "expected a whole number";
                return false;
            }
            out = value;
        } else if (sample.is_number_float()) {
            size_t used = 0;
            const double value = std::stod(text, &used);
            if (used != text.size()) {
                problem = "expected a number";
                return false;
            }
            out = value;
        } else {
            out = text;
        }
        return true;
    } catch (const std::exception&) {
        problem = sample.is_number_float() ? "expected a number" : "expected a whole number";
        return false;
    }
}

// ---- ls --------------------------------------------------------------------------------------------------------------

int runLs(host::AppHost& host, bool json, std::ostream& out, std::ostream&) {
    const fs::path dir = host.modelsDir();
    const auto cards = host.library().scan();
    Json models = Json::array();
    for (const auto& c : cards) {
        Json entry = models::toJson(c);
        entry["hasDefaults"] = host.modelDefaults().get(c.file).has_value();
        models.push_back(std::move(entry));
    }
    if (json) {
        out << Json{{"modelsDir", dir.string()}, {"models", models}}.dump(2) << "\n";
        return kOk;
    }
    out << "Models in " << dir.string() << "\n";
    if (models.empty()) {
        out << "  none (put .safetensors, .ckpt or .gguf files there, download some with: graphicshost get, or set modelsDir with: graphicshost config set modelsDir <folder>)\n";
    }
    for (const auto& m : models) {
        std::string family = m["kind"].get<std::string>() == "checkpoint" ? m["family"].get<std::string>() : m["kind"].get<std::string>();
        if (!m["supported"].get<bool>()) family += " (not supported)";
        out << "  " << std::left << std::setw(40) << m["file"].get<std::string>() << std::right << std::setw(12)
            << humanSize(m["sizeBytes"].get<uint64_t>()) << "  " << family
            << (m["hasDefaults"].get<bool>() ? "  (has defaults)" : "")
            << (!m["vae"].get<std::string>().empty() ? "  + VAE " + m["vae"].get<std::string>() : std::string()) << "\n";
    }
    return kOk;
}

// ---- gen -------------------------------------------------------------------------------------------------------------

int runGen(host::AppHost& host, const GenInvocation& in, bool json, bool quiet, std::ostream& out, std::ostream& err) {
    // 1. the model
    models::ModelCard card;
    std::string problem;
    int code = kOk;
    if (!resolveModel(host, in.model, card, problem, code)) {
        err << "error: " << problem << "\n";
        return code;
    }
    if (card.kind != models::Kind::Checkpoint) {
        err << "error: " << card.file << " is a " << models::toString(card.kind) << ", not a checkpoint that can generate images\n";
        return kUsage;
    }
    if (!card.supported) {
        err << "error: " << card.file << ": " << (card.note.empty() ? "not supported in this version" : card.note) << "\n";
        return kUsage;
    }
    const fs::path modelPath = card.path;
    const std::string modelFile = card.file;

    // 2. the layers: model defaults < preset < what was given on the command line
    std::vector<Json> requestLayers, loadLayers;
    if (const auto defaults = host.modelDefaults().get(modelFile)) {
        requestLayers.push_back(defaults->request);
        loadLayers.push_back(defaults->load);
    }
    if (!in.preset.empty()) {
        const auto preset = host.presets().get(in.preset);
        if (!preset) {
            err << "error: no preset named '" << in.preset << "' (see: graphicshost preset list)\n";
            return kNotFound;
        }
        requestLayers.push_back(preset->request);
        loadLayers.push_back(preset->load);
    }
    requestLayers.push_back(in.request);
    loadLayers.push_back(in.load);

    engine::GenerationRequest request;
    engine::LoadOptions load;
    try {
        if (!in.prompt.empty()) {
            requestLayers.push_back({{"prompt", in.prompt}});
        }
        request = presets::layerRequest(requestLayers);
        load = presets::layerLoadOptions(loadLayers);
    } catch (const std::exception& e) {
        err << "error: " << e.what() << "\n";
        return kUsage;
    }
    if (const auto verdict = host.checkSafety(request); !verdict.allowed) {
        err << "error: " << verdict.reason << "\n";
        return kUsage;
    }
    load.modelPath = modelPath.string();
    if (load.vaePath.empty() && !card.vae.empty()) {
        load.vaePath = (modelPath.parent_path() / card.vae).string();  // the VAE the library linked to this checkpoint
    }

    if (const auto errors = engine::validate(request); !errors.empty()) {
        err << "error: invalid parameters:\n" << errorList(errors);
        return kUsage;
    }
    if (const auto errors = engine::validate(load); !errors.empty()) {
        err << "error: invalid load options:\n" << errorList(errors);
        return kUsage;
    }

    // 3. the runtime and the memory estimate
    const engine::RuntimeCandidate* chosen = host.selectedRuntime();
    if (!chosen) {
        err << "error: no inference runtime fits the backend setting '" << host.settings().backend
            << "' (see: graphicshost hardware)\n";
        return kNotFound;
    }
    std::error_code ec;
    const uint64_t fileSize = fs::file_size(modelPath, ec);
    const std::string guessedFamily = card.familyName;  // from the file header, not from the size
    const engine::FamilyDefaults sizes = card.defaults;
    const engine::GenerationRequest sized = engine::resolve(request, sizes);

    engine::FitReport fit;
    if (!in.noFitCheck) {
        engine::FitInput fi;
        fi.weightsBytes = card.weightsBytes > 0 ? card.weightsBytes : engine::estimateWeightsBytes(fileSize, guessedFamily);
        fi.family = guessedFamily;
        fi.width = sized.width;
        fi.height = sized.height;
        fi.device = load.device;
        fi.flashAttention = load.flashAttention;
        fi.offloadParamsToCpu = load.offloadParamsToCpu;
        const auto mem = engine::systemMemory();
        fi.systemRamFreeBytes = mem.freeBytes;
        if (chosen->backend == "vulkan") {
            for (const auto& d : host.gpu().devices) {
                if (d.supported) {
                    fi.gpuMemoryBytes = d.vramBytes;
                    fi.gpuIsIntegrated = !d.discrete;
                    break;
                }
            }
        }
        fit = engine::estimateFit(fi);
        if (fit.status == engine::FitStatus::WontFit) {
            err << "error: " << fit.summary << "\n";
            for (const auto& s : fit.suggestions) err << "  - " << s << "\n";
            err << "  (use --no-fit-check to try anyway)\n";
            return kOutOfMemory;
        }
        if (fit.needsOffload && !load.offloadParamsToCpu) {
            load.offloadParamsToCpu = true;
            if (!quiet) err << "note: " << fit.summary << "; weights offloaded to RAM\n";
        }
        if (fit.status == engine::FitStatus::CpuOnly && load.device == "auto") {
            load.device = "cpu";
            if (!quiet) err << "note: " << fit.summary << "\n";
        }
        if (fit.suggestFlashAttention && !quiet) {
            err << "tip: " << fit.suggestions.front() << " (--flash-attn)\n";
        }
    }

    if (in.dryRun) {
        const Json plan = {{"model", modelPath.string()},
                           {"runtime", chosen->id},
                           {"load", engine::toJson(load)},
                           {"request", engine::toJson(sized)},
                           {"fit", fitJson(fit)}};
        if (json) {
            out << plan.dump(2) << "\n";
        } else {
            out << "Model     " << modelPath.string() << " (" << humanSize(fileSize) << ")\n";
            out << "Runtime   " << chosen->id << ", device " << load.device << (load.offloadParamsToCpu ? ", weights offloaded" : "") << "\n";
            out << "Request   " << engine::toJson(sized).dump() << "\n";
            out << "Memory    " << (fit.summary.empty() ? "not checked" : fit.summary) << "\n";
        }
        return kOk;
    }

    // 4. load
    engine::ModelManager& models = host.models();
    try {
        if (!quiet) err << "loading " << modelFile << " ...\n";
        models.load(load);
    } catch (const engine::EngineError& e) {
        err << "error: " << e.what() << "\n";
        const auto offers = engine::suggestLoadFallbacks(load, e.code(), chosen->backend);
        if (!offers.empty()) {
            err << "You could try:\n";
            for (const auto& o : offers) err << "  " << o.cliFlags << "   " << o.description << "\n";
        }
        return exitCodeFor(e.code());
    }

    // 5. generate, with progress on stderr and Ctrl+C support
    std::shared_ptr<engine::Job> job;
    try {
        job = models.submit(request);
    } catch (const engine::EngineError& e) {
        err << "error: " << e.what() << "\n";
        return exitCodeFor(e.code());
    }
    g_runningJob = job.get();
    SetConsoleCtrlHandler(onConsoleControl, TRUE);
    int lastShown = -1;
    while (!job->waitFor(std::chrono::milliseconds(150))) {
        const auto p = job->progress();
        if (!quiet && !json && p.step != lastShown && p.steps > 0) {
            err << "\rstep " << p.step << "/" << p.steps << "   " << std::flush;
            lastShown = p.step;
        }
    }
    if (lastShown >= 0) err << "\n";
    SetConsoleCtrlHandler(onConsoleControl, FALSE);
    g_runningJob = nullptr;

    if (job->state() != engine::JobState::Succeeded) {
        err << "error: " << (job->state() == engine::JobState::Cancelled ? "cancelled" : job->error()) << "\n";
        return exitCodeFor(job->errorCode());
    }

    // 6. store and report
    const engine::GenerationResult result = job->result();
    const auto info = models.loaded()->info;
    std::vector<std::string> files;
    Json images = Json::array();
    std::vector<gallery::GalleryEntry> entries;
    if (!in.noGallery) {
        try {
            entries = host.gallery().addResult(result, info, in.quality);
        } catch (const std::exception& e) {
            err << "warning: could not add to the gallery: " << e.what() << "\n";
        }
    }
    const auto format = engine::parseImageFormat(result.request.format).value_or(engine::ImageFormat::Png);
    if (!in.outPath.empty()) {
        fs::path target(in.outPath);
        const bool singleFile = result.images.size() == 1 && target.has_extension();
        if (!singleFile) {
            fs::create_directories(target);
        } else if (target.has_parent_path()) {
            fs::create_directories(target.parent_path());
        }
        for (size_t i = 0; i < result.images.size(); ++i) {
            fs::path file = singleFile ? target
                                       : target / (std::to_string(result.seeds[i]) + "-" + std::to_string(i + 1) + engine::extensionFor(format));
            std::optional<std::string> meta;
            if (result.request.embedMetadata) {
                engine::GenerationRequest r = result.request;
                r.seed = result.seeds[i];
                meta = Json{{"app", kAppName}, {"version", kAppVersion}, {"model", modelFile}, {"seed", result.seeds[i]},
                            {"request", engine::toJson(r)}}.dump();
            }
            const auto bytes = engine::encodeImage(result.images[i], format, meta, in.quality);
            core::writeFileAtomic(file, std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
            files.push_back(file.string());
        }
    }
    for (size_t i = 0; i < result.images.size(); ++i) {
        Json image = {{"seed", result.seeds[i]}, {"width", result.images[i].width}, {"height", result.images[i].height}};
        if (i < entries.size()) {
            image["galleryId"] = entries[i].id;
            image["galleryPath"] = host.gallery().absolutePath(entries[i]).string();
        }
        if (i < files.size()) image["file"] = files[i];
        images.push_back(std::move(image));
    }
    if (json) {
        out << Json{{"images", images},
                    {"seconds", result.seconds},
                    {"family", info.family},
                    {"runtime", info.runtimeId},
                    {"device", info.device},
                    {"warnings", result.warnings},
                    {"request", engine::toJson(result.request)}}
                   .dump(2)
            << "\n";
    } else {
        out << "Generated " << result.images.size() << " image" << (result.images.size() == 1 ? "" : "s") << " in "
            << std::fixed << std::setprecision(1) << result.seconds << " s (" << info.family << ", runtime " << info.runtimeId << ")\n";
        for (const auto& image : images) {
            out << "  seed " << image["seed"] << "  " << image["width"] << "x" << image["height"];
            if (image.contains("file")) out << "  " << image["file"].get<std::string>();
            else if (image.contains("galleryPath")) out << "  " << image["galleryPath"].get<std::string>();
            out << "\n";
        }
        for (const auto& w : result.warnings) out << "warning: " << w << "\n";
    }
    return kOk;
}

// ---- presets, model defaults, gallery --------------------------------------------------------------------------------

namespace {

// key value pairs from "set": keys starting with "load." go to the load part
bool collectPairs(const std::vector<std::string>& args, size_t from, Json& request, Json& load, std::ostream& err) {
    if (args.size() <= from || (args.size() - from) % 2 != 0) {
        err << "error: give key value pairs, for example: steps 30 sampler dpm++2m load.flashAttention true\n";
        return false;
    }
    const Json requestSample = engine::toJson(engine::GenerationRequest{});
    const Json loadSample = engine::toJson(engine::LoadOptions{});
    for (size_t i = from; i < args.size(); i += 2) {
        std::string key = args[i];
        const bool isLoad = key.rfind("load.", 0) == 0;
        if (isLoad) key = key.substr(5);
        const Json& sample = isLoad ? loadSample : requestSample;
        if (!sample.contains(key)) {
            err << "error: unknown " << (isLoad ? "load option" : "parameter") << " '" << key << "'\n";
            return false;
        }
        Json value;
        std::string problem;
        if (!coerceJsonValue(sample[key], args[i + 1], value, problem)) {
            err << "error: " << args[i] << ": " << problem << "\n";
            return false;
        }
        (isLoad ? load : request)[key] = value;
    }
    return true;
}

}  // namespace

int runPreset(host::AppHost& host, const std::string& action, const std::vector<std::string>& args, bool json,
              std::ostream& out, std::ostream& err) {
    auto& store = host.presets();
    try {
        if (action == "list") {
            const auto names = store.list();
            if (json) {
                out << Json(names).dump(2) << "\n";
            } else if (names.empty()) {
                out << "no presets in " << host.paths().presetsDir().string() << "\n";
            } else {
                for (const auto& n : names) {
                    const auto p = store.get(n);
                    out << std::left << std::setw(24) << n << (p ? p->description : std::string("(unreadable)")) << "\n";
                }
            }
            return kOk;
        }
        if (action == "show" && args.size() == 1) {
            const auto p = store.get(args[0]);
            if (!p) {
                err << "error: no preset named '" << args[0] << "'\n";
                return kNotFound;
            }
            out << Json{{"name", p->name}, {"description", p->description}, {"request", p->request}, {"load", p->load}}.dump(2) << "\n";
            return kOk;
        }
        if (action == "set" && args.size() >= 1) {
            Json request = Json::object(), load = Json::object();
            if (!collectPairs(args, 1, request, load, err)) return kUsage;
            presets::Preset preset;
            if (const auto existing = store.get(args[0])) preset = *existing;
            preset.name = args[0];
            for (const auto& [k, v] : request.items()) preset.request[k] = v;
            for (const auto& [k, v] : load.items()) preset.load[k] = v;
            store.save(preset);
            out << "saved preset " << preset.name << "\n";
            return kOk;
        }
        if (action == "delete" && args.size() == 1) {
            if (!store.remove(args[0])) {
                err << "error: no preset named '" << args[0] << "'\n";
                return kNotFound;
            }
            out << "deleted preset " << args[0] << "\n";
            return kOk;
        }
        if (action == "import" && (args.size() == 1 || args.size() == 2)) {
            out << "imported preset " << store.importFile(args[0], args.size() == 2 ? args[1] : std::string()) << "\n";
            return kOk;
        }
        if (action == "export" && args.size() == 2) {
            store.exportFile(args[0], args[1]);
            out << "exported to " << args[1] << "\n";
            return kOk;
        }
    } catch (const std::invalid_argument& e) {
        err << "error: " << e.what() << "\n";
        return kUsage;
    } catch (const std::exception& e) {
        err << "error: " << e.what() << "\n";
        return kFailure;
    }
    err << "usage: graphicshost preset list | show NAME | set NAME key value... | delete NAME | import FILE [NAME] | export NAME FILE\n";
    return kUsage;
}

int runDefaults(host::AppHost& host, const std::string& action, const std::vector<std::string>& args, bool json,
                std::ostream& out, std::ostream& err) {
    auto& store = host.modelDefaults();
    try {
        if (action == "list") {
            const auto names = store.models();
            if (json) out << Json(names).dump(2) << "\n";
            else if (names.empty()) out << "no model defaults set\n";
            else for (const auto& n : names) out << n << "\n";
            return kOk;
        }
        if (action == "show" && args.size() == 1) {
            const auto d = store.get(args[0]);
            if (!d) {
                err << "error: no defaults for '" << args[0] << "'\n";
                return kNotFound;
            }
            out << Json{{"model", args[0]}, {"request", d->request}, {"load", d->load}}.dump(2) << "\n";
            return kOk;
        }
        if (action == "set" && args.size() >= 1) {
            Json request = Json::object(), load = Json::object();
            if (!collectPairs(args, 1, request, load, err)) return kUsage;
            presets::ModelDefaultsStore::Defaults d;
            if (const auto existing = store.get(args[0])) d = *existing;
            for (const auto& [k, v] : request.items()) d.request[k] = v;
            for (const auto& [k, v] : load.items()) d.load[k] = v;
            store.set(args[0], d);
            out << "saved defaults for " << args[0] << "\n";
            return kOk;
        }
        if (action == "clear" && args.size() == 1) {
            if (!store.remove(args[0])) {
                err << "error: no defaults for '" << args[0] << "'\n";
                return kNotFound;
            }
            out << "cleared defaults for " << args[0] << "\n";
            return kOk;
        }
    } catch (const std::invalid_argument& e) {
        err << "error: " << e.what() << "\n";
        return kUsage;
    } catch (const std::exception& e) {
        err << "error: " << e.what() << "\n";
        return kFailure;
    }
    err << "usage: graphicshost defaults list | show MODELFILE | set MODELFILE key value... | clear MODELFILE\n";
    return kUsage;
}

int runGallery(host::AppHost& host, const std::string& action, const std::vector<std::string>& args, int limit,
               bool json, std::ostream& out, std::ostream& err) {
    try {
        auto& gallery = host.gallery();
        if (action == "list") {
            gallery::GalleryQuery query;
            query.text = args.empty() ? std::string() : args[0];
            query.limit = limit;
            const auto entries = gallery.list(query);
            if (json) {
                Json list = Json::array();
                for (const auto& e : entries) {
                    list.push_back({{"id", e.id}, {"path", gallery.absolutePath(e).string()}, {"created", e.created},
                                    {"prompt", e.prompt}, {"model", e.model}, {"seed", e.seed},
                                    {"width", e.width}, {"height", e.height}, {"format", e.format}});
                }
                out << Json{{"total", gallery.count(query.text)}, {"images", list}}.dump(2) << "\n";
            } else {
                for (const auto& e : entries) {
                    out << std::right << std::setw(5) << e.id << "  " << e.created.substr(0, 19) << "  " << e.width << "x" << e.height
                        << "  seed " << e.seed << "  " << e.prompt.substr(0, 60) << "\n";
                }
                out << gallery.count(query.text) << " image(s) in total" << (query.text.empty() ? "" : " matching") << "\n";
            }
            return kOk;
        }
        if ((action == "show" || action == "rm") && args.size() == 1) {
            int64_t id = 0;
            try { id = std::stoll(args[0]); } catch (const std::exception&) { err << "error: the id must be a number\n"; return kUsage; }
            const auto entry = gallery.get(id);
            if (!entry) {
                err << "error: no image with id " << id << "\n";
                return kNotFound;
            }
            if (action == "rm") {
                gallery.remove(id);
                out << "removed image " << id << "\n";
            } else {
                Json record = entry->parameters;
                record["id"] = entry->id;
                record["file"] = gallery.absolutePath(*entry).string();
                out << record.dump(2) << "\n";
            }
            return kOk;
        }
        if (action == "rescan" && args.empty()) {
            out << "added " << gallery.rescan() << " image(s) to the index\n";
            return kOk;
        }
    } catch (const std::exception& e) {
        err << "error: " << e.what() << "\n";
        return kFailure;
    }
    err << "usage: graphicshost gallery list [text] [--limit N] | show ID | rm ID | rescan\n";
    return kUsage;
}

}  // namespace gh::cli
