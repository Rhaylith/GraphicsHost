#include "host/GenerationService.h"

#include "core/Log.h"
#include "engine/FitEstimator.h"
#include "engine/ImageCodec.h"
#include "host/AppHost.h"
#include "presets/PresetStore.h"

#include <algorithm>
#include <cmath>
#include <regex>

namespace gh::host {

using Json = nlohmann::json;
using engine::EngineError;
using engine::EngineErrorCode;

namespace {

std::string errorList(const char* what, const std::vector<core::FieldError>& errors) {
    std::string message = what;
    for (const auto& e : errors) message += " " + e.field + " " + e.message + ";";
    return message;
}


// "<lora:name:0.8>" tags in a prompt: removed from the text, returned as references (a missing weight is 1).
std::vector<engine::LoraRef> takeLoraTags(std::string& prompt) {
    static const std::regex tag(R"(<\s*(?:lora|lyco)\s*:\s*([^:>]+?)\s*(?::\s*(-?[0-9]*\.?[0-9]+))?\s*>)", std::regex::icase);
    std::vector<engine::LoraRef> found;
    std::string rest;
    auto begin = std::sregex_iterator(prompt.begin(), prompt.end(), tag);
    size_t last = 0;
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        engine::LoraRef ref;
        ref.name = (*it)[1].str();
        if ((*it)[2].matched) ref.weight = std::stof((*it)[2].str());
        found.push_back(std::move(ref));
        rest += prompt.substr(last, static_cast<size_t>(it->position()) - last);
        last = static_cast<size_t>(it->position() + it->length());
    }
    if (!found.empty()) {
        rest += prompt.substr(last);
        // tidy the gaps the tags leave behind
        rest = std::regex_replace(rest, std::regex(R"(\s{2,})"), " ");
        const auto first = rest.find_first_not_of(" ,");
        const auto end = rest.find_last_not_of(" ,");
        prompt = first == std::string::npos ? std::string() : rest.substr(first, end - first + 1);
    }
    return found;
}
}  // namespace

GenerationService::GenerationService(AppHost& host) : host_(host) {
    host_.models().queue().setObserver([this](const engine::Job& job) { onJob(job); });
}

GenerationService::~GenerationService() {
    // the queue belongs to AppHost and may outlive this object: stop calling into us
    host_.models().queue().setObserver({});
}

std::vector<gallery::GalleryEntry> GenerationService::entries(uint64_t jobId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = stored_.find(jobId);
    return it == stored_.end() ? std::vector<gallery::GalleryEntry>{} : it->second;
}

GenerationCounters GenerationService::counters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return counters_;
}

GenerationService::Prepared GenerationService::prepare(const GenerateCall& call) {
    Prepared prep;
    Submission& result = prep.submission;
    engine::ModelManager& manager = host_.models();

    // 1. which model
    models::ModelCard card;
    {
        std::string arg = call.model;
        if (arg.empty()) {
            const auto loaded = manager.loaded();
            if (!loaded) {
                throw EngineError(EngineErrorCode::NotFound, "no model is loaded: name one in 'model' (see GET /api/v1/models)");
            }
            arg = loaded->options.modelPath;
        }
        const auto resolved = host_.library().resolve(arg);
        if (resolved.status != models::ModelLibrary::Resolution::Status::Found) {
            throw EngineError(resolved.status == models::ModelLibrary::Resolution::Status::Ambiguous ? EngineErrorCode::InvalidRequest : EngineErrorCode::NotFound,
                              resolved.problem);
        }
        card = *resolved.card;
    }
    if (card.kind != models::Kind::Checkpoint) {
        throw EngineError(EngineErrorCode::InvalidRequest, card.file + " is a " + models::toString(card.kind) + ", not a checkpoint that can generate images");
    }
    if (!card.supported) {
        throw EngineError(EngineErrorCode::InvalidRequest, card.file + ": " + (card.note.empty() ? "not supported in this version" : card.note));
    }
    result.model = card.file;

    // 2. layers: model defaults < preset < the caller
    std::vector<Json> requestLayers, loadLayers;
    if (const auto defaults = host_.modelDefaults().get(card.file)) {
        requestLayers.push_back(defaults->request);
        loadLayers.push_back(defaults->load);
    }
    if (!call.preset.empty()) {
        const auto preset = host_.presets().get(call.preset);
        if (!preset) throw EngineError(EngineErrorCode::NotFound, "no preset named '" + call.preset + "'");
        requestLayers.push_back(preset->request);
        loadLayers.push_back(preset->load);
    }
    requestLayers.push_back(call.request);
    loadLayers.push_back(call.loadOptions);

    engine::GenerationRequest request;
    engine::LoadOptions load;
    try {
        request = presets::layerRequest(requestLayers);
        load = presets::layerLoadOptions(loadLayers);
    } catch (const std::exception& e) {
        throw EngineError(EngineErrorCode::InvalidRequest, e.what());
    }
    // LoRAs: the request's list plus the tags in the prompt (a tag wins over a list entry of the same name), each
    // resolved to a file of the library and checked against the model's family
    {
        for (auto& tagged : takeLoraTags(request.prompt)) {
            auto same = std::find_if(request.loras.begin(), request.loras.end(), [&](const engine::LoraRef& l) { return l.name == tagged.name; });
            if (same != request.loras.end()) *same = tagged;
            else request.loras.push_back(tagged);
        }
        if (!request.loras.empty()) {
            const auto available = host_.library().ofKind(models::Kind::Lora);
            for (auto& ref : request.loras) {
                std::string lower = ref.name;
                std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                const models::ModelCard* match = nullptr;
                std::vector<const models::ModelCard*> prefixed;
                for (const auto& c : available) {
                    auto low = [](std::string s) { std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }); return s; };
                    if (low(c.file) == lower || low(c.name) == lower) { match = &c; break; }
                    if (low(c.name).rfind(lower, 0) == 0) prefixed.push_back(&c);
                }
                if (!match && prefixed.size() == 1) match = prefixed.front();
                if (!match && prefixed.size() > 1) throw EngineError(EngineErrorCode::InvalidRequest, "the LoRA name '" + ref.name + "' matches several files: use a longer name");
                if (!match) throw EngineError(EngineErrorCode::NotFound, "no LoRA named '" + ref.name + "' in the models folder (see GET /api/v1/models)");
                if (!match->familyName.empty() && !card.familyName.empty() && match->familyName != card.familyName) {
                    throw EngineError(EngineErrorCode::InvalidRequest, "the LoRA " + match->file + " is for " + match->familyName + ", this model is " + card.familyName);
                }
                ref.name = match->name;
                ref.path = match->path.string();
            }
        }
    }
    if (call.initImage) {
        // the result has the size of the picture unless the request says otherwise; the picture is scaled to the result
        const engine::Image& in = *call.initImage;
        if (in.width == 0 || in.height == 0) throw EngineError(EngineErrorCode::InvalidRequest, "the source image is empty");
        auto round8 = [](double v) { return std::max(64, static_cast<int>(std::lround(v / 8.0)) * 8); };
        const double aspect = static_cast<double>(in.width) / in.height;
        if (request.width <= 0 && request.height <= 0) {
            const int limit = 2 * std::max(card.defaults.width, card.defaults.height);   // keep a photo from becoming a huge job
            const double scale = std::min(1.0, static_cast<double>(limit) / std::max(in.width, in.height));
            request.width = round8(in.width * scale);
            request.height = round8(in.height * scale);
        } else if (request.height <= 0) {
            request.height = round8(request.width / aspect);
        } else if (request.width <= 0) {
            request.width = round8(request.height * aspect);
        }
        try {
            request.initImage = std::make_shared<const engine::Image>(
                engine::resizeImage(in, static_cast<uint32_t>(request.width), static_cast<uint32_t>(request.height)));
        } catch (const std::exception& e) {
            throw EngineError(EngineErrorCode::InvalidRequest, std::string("the source image cannot be used: ") + e.what());
        }
        if (call.maskImage) {
            try {
                request.maskImage = std::make_shared<const engine::Image>(
                    engine::resizeImage(*call.maskImage, static_cast<uint32_t>(request.width), static_cast<uint32_t>(request.height)));
            } catch (const std::exception& e) {
                throw EngineError(EngineErrorCode::InvalidRequest, std::string("the mask cannot be used: ") + e.what());
            }
        }
    } else if (call.maskImage) {
        throw EngineError(EngineErrorCode::InvalidRequest, "a mask needs a source image (init_image)");
    }
    if (const auto verdict = host_.checkSafety(request); !verdict.allowed) {
        throw EngineError(EngineErrorCode::Blocked, verdict.reason);
    }
    load.modelPath = card.path.string();
    if (load.vaePath.empty() && !card.vae.empty()) load.vaePath = (card.path.parent_path() / card.vae).string();
    if (const auto errors = engine::validate(request); !errors.empty()) {
        throw EngineError(EngineErrorCode::InvalidRequest, errorList("invalid parameters:", errors));
    }
    if (const auto errors = engine::validate(load); !errors.empty()) {
        throw EngineError(EngineErrorCode::InvalidRequest, errorList("invalid load options:", errors));
    }

    // 3. does it fit? (only when the model is not already loaded the way it is asked for)
    const engine::RuntimeCandidate* runtime = host_.selectedRuntime();
    const auto already = manager.loaded();
    if (runtime && !(already && already->options == load)) {
        const engine::GenerationRequest sized = engine::resolve(request, card.defaults);
        engine::FitInput fi;
        fi.family = card.familyName;
        fi.weightsBytes = card.weightsBytes;
        fi.width = sized.width;
        fi.height = sized.height;
        fi.device = load.device;
        fi.flashAttention = load.flashAttention;
        fi.offloadParamsToCpu = load.offloadParamsToCpu;
        fi.systemRamFreeBytes = engine::systemMemory().freeBytes;
        if (runtime->backend == "vulkan") {
            for (const auto& d : host_.gpu().devices) {
                if (d.supported) {
                    fi.gpuMemoryBytes = d.vramBytes;
                    fi.gpuIsIntegrated = !d.discrete;
                    break;
                }
            }
        }
        const engine::FitReport fit = engine::estimateFit(fi);
        if (fit.status == engine::FitStatus::WontFit) {
            throw EngineError(EngineErrorCode::OutOfMemory, fit.summary);
        }
        if (fit.needsOffload && !load.offloadParamsToCpu) {
            load.offloadParamsToCpu = true;
            result.notes.push_back(fit.summary + "; weights offloaded to RAM");
        }
        if (fit.status == engine::FitStatus::CpuOnly && load.device == "auto") {
            load.device = "cpu";
            result.notes.push_back(fit.summary);
        }
    }
    prep.request = request;
    prep.load = load;
    return prep;
}

Submission GenerationService::loadModel(const GenerateCall& call) {
    GenerateCall withPrompt = call;   // only the load options matter; the request must still validate
    if (!withPrompt.request.contains("prompt")) withPrompt.request["prompt"] = "load";
    Prepared prep = prepare(withPrompt);
    host_.models().load(prep.load);
    return prep.submission;
}

Submission GenerationService::submit(const GenerateCall& call) {
    Prepared prep = prepare(call);
    Submission& result = prep.submission;
    engine::ModelManager& manager = host_.models();
    // load just in time, then queue
    manager.load(prep.load);
    result.job = manager.submit(prep.request);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        wantsGallery_[result.job->id()] = call.addToGallery;
        if (const auto info = manager.loaded()) modelOf_[result.job->id()] = info->info;
        ++counters_.submitted;
        trim();
    }
    if (result.job->finished()) onJob(*result.job);  // it finished before it was registered above
    return result;
}

void GenerationService::onJob(const engine::Job& job) {
    if (!job.finished()) return;
    bool toGallery;
    engine::ModelInfo model;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!wantsGallery_.count(job.id()) || counted_[job.id()]) return;  // not registered yet, or already handled
        counted_[job.id()] = true;
        toGallery = wantsGallery_[job.id()];
        model = modelOf_[job.id()];
    }
    const auto state = job.state();
    std::vector<gallery::GalleryEntry> entries;
    uint64_t images = 0;
    double seconds = 0;
    if (state == engine::JobState::Succeeded) {
        try {
            const engine::GenerationResult generated = job.result();
            images = generated.images.size();
            seconds = generated.seconds;
            if (toGallery) entries = host_.gallery().addResult(generated, model);
        } catch (const std::exception& e) {
            log::warn(std::string("could not store the images of job ") + std::to_string(job.id()) + ": " + e.what());
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!entries.empty()) stored_[job.id()] = std::move(entries);
        if (state == engine::JobState::Succeeded) {
            ++counters_.succeeded;
            counters_.images += images;
            counters_.seconds += seconds;
        } else if (state == engine::JobState::Cancelled) {
            ++counters_.cancelled;
        } else {
            ++counters_.failed;
        }
        handled_[job.id()] = true;
    }
    handledCv_.notify_all();
}

void GenerationService::waitHandled(uint64_t jobId, std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(mutex_);
    handledCv_.wait_for(lock, timeout, [&] { return !wantsGallery_.count(jobId) || handled_.count(jobId); });
}

void GenerationService::trim() {
    // only the recent jobs matter to clients (the queue itself keeps 100)
    while (wantsGallery_.size() > 500) {
        const uint64_t oldest = wantsGallery_.begin()->first;
        wantsGallery_.erase(oldest);
        stored_.erase(oldest);
        modelOf_.erase(oldest);
        counted_.erase(oldest);
        handled_.erase(oldest);
    }
}

}  // namespace gh::host
