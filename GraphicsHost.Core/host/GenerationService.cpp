#include "host/GenerationService.h"

#include "core/Log.h"
#include "engine/FitEstimator.h"
#include "host/AppHost.h"
#include "presets/PresetStore.h"

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
