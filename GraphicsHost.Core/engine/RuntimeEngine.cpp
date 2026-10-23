#include "engine/ImageEngine.h"

#include "core/Log.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <mutex>

namespace gh::engine {

namespace {

// Image to image runs only the last part of the schedule: steps * strength steps (one fewer when that would be
// all of them) plus one, which is how stable-diffusion.cpp counts and the progress callback reports it.
int sampledSteps(int steps, float strength) {
    int n = static_cast<int>(static_cast<float>(steps) * strength);
    if (n >= steps) n = steps - 1;
    return n + 1;   // the schedule keeps its starting point
}

bool looksLikeOutOfMemory(const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    for (const char* needle : {"out of memory", "out_of_device_memory", "out_of_host_memory", "failed to allocate",
                               "alloc failed", "not enough memory", "bad_alloc"}) {
        if (lower.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string joinNativeErrors(const std::vector<std::string>& lines) {
    std::string out;
    for (const auto& line : lines) {
        out += (out.empty() ? "" : "; ") + line;
    }
    return out;
}

class RuntimeModel final : public ILoadedModel {
public:
    RuntimeModel(std::shared_ptr<Runtime> runtime, gh_model* handle, ModelInfo info)
        : runtime_(std::move(runtime)), handle_(handle), info_(std::move(info)) {}

    ~RuntimeModel() override { runtime_->api().free(handle_); }

    const ModelInfo& info() const override { return info_; }

    GenerationResult generate(const GenerationRequest& requested, const ProgressSink& progress,
                              std::stop_token stop) override {
        if (const auto errors = validate(requested); !errors.empty()) {
            std::string message = "invalid parameters:";
            for (const auto& e : errors) {
                message += " " + e.field + " " + e.message + ";";
            }
            throw EngineError(EngineErrorCode::InvalidRequest, message);
        }
        if (stop.stop_requested()) {
            throw EngineError(EngineErrorCode::Cancelled, "cancelled before it started");
        }
        GenerationRequest request = resolve(requested, info_.defaults);

        std::vector<std::string> warnings;
        if (request.cfgScale <= 1.0f && !request.negativePrompt.empty()) {
            warnings.push_back("the negative prompt has no effect when cfgScale is 1 or lower");
        }

        if (request.initImage && (runtime_->features() & GH_FEATURE_IMG2IMG) == 0) {
            throw EngineError(EngineErrorCode::InvalidRequest, "this runtime build does not support image to image: update the runtime DLLs");
        }
        if (request.initImage && (static_cast<int>(request.initImage->width) != request.width || static_cast<int>(request.initImage->height) != request.height)) {
            throw EngineError(EngineErrorCode::InvalidRequest, "the source image must have the size of the result");
        }

        gh_generate_params params{};
        params.struct_size = sizeof(params);
        params.prompt = request.prompt.c_str();
        params.negative_prompt = request.negativePrompt.c_str();
        params.width = request.width;
        params.height = request.height;
        params.steps = request.steps;
        params.cfg_scale = request.cfgScale;
        params.seed = request.seed;
        params.batch_count = request.batchCount;
        params.sampler = request.sampler.empty() ? nullptr : request.sampler.c_str();
        params.scheduler = request.scheduler.empty() ? nullptr : request.scheduler.c_str();
        params.clip_skip = request.clipSkip;
        params.vae_tiling = request.vaeTiling ? 1 : 0;
        if (request.maskImage && (runtime_->features() & GH_FEATURE_INPAINT) == 0) {
            throw EngineError(EngineErrorCode::InvalidRequest, "this runtime build does not support inpainting: update the runtime DLLs");
        }
        if (request.maskImage && (static_cast<int>(request.maskImage->width) != request.width || static_cast<int>(request.maskImage->height) != request.height)) {
            throw EngineError(EngineErrorCode::InvalidRequest, "the mask must have the size of the result");
        }
        gh_image maskView{};
        gh_image initView{};
        if (request.initImage) {
            initView.width = request.initImage->width;
            initView.height = request.initImage->height;
            initView.channels = request.initImage->channels;
            initView.data = const_cast<uint8_t*>(request.initImage->data.data());   // read only for the runtime
            params.init_image = &initView;
            params.strength = request.strength;
            if (request.maskImage) {
                maskView.width = request.maskImage->width;
                maskView.height = request.maskImage->height;
                maskView.channels = request.maskImage->channels;
                maskView.data = const_cast<uint8_t*>(request.maskImage->data.data());
                params.mask_image = &maskView;
            }
        }

        // The engine reports progress for weight loading and VAE decoding as well (totals such as 140 tensors).
        // Only the sampling loop has exactly as many steps as were asked for, so everything else is dropped.
        struct Trampoline {
            const ProgressSink* sink;
            int expectedSteps;
        } trampoline{&progress, request.initImage ? sampledSteps(request.steps, request.strength) : request.steps};
        gh_progress_fn onProgress = [](int32_t step, int32_t steps, float seconds, void* user) {
            const auto* t = static_cast<Trampoline*>(user);
            if (t->sink && *t->sink && steps == t->expectedSteps && step >= 1) {  // reported as 1..steps
                (*t->sink)(StepProgress{step, steps, seconds});
            }
        };

        // A stop request becomes gh_cancel, which takes effect at the next sampling step.
        gh_model* handle = handle_;
        const auto cancel = runtime_->api().cancel;
        std::stop_callback onStop(stop, [handle, cancel]() { cancel(handle); });

        const uint64_t mark = log::nativeMark();
        gh_result result{};
        const int32_t status = runtime_->api().generate(handle_, &params, onProgress, &trampoline, &result);
        if (status != GH_OK) {
            const std::string detail = runtime_->api().lastError();
            const auto nativeErrors = log::nativeErrorsSince(mark);
            if (stop.stop_requested()) {
                throw EngineError(EngineErrorCode::Cancelled, "generation cancelled");
            }
            const std::string all = detail + " " + joinNativeErrors(nativeErrors);
            throw EngineError(looksLikeOutOfMemory(all) ? EngineErrorCode::OutOfMemory : EngineErrorCode::Failed,
                              "generation failed: " + detail + (nativeErrors.empty() ? "" : " (" + joinNativeErrors(nativeErrors) + ")"));
        }

        GenerationResult out;
        out.seconds = result.generate_seconds;
        out.warnings = std::move(warnings);
        for (int i = 0; i < result.image_count; ++i) {
            const gh_image& src = result.images[i];
            Image image;
            image.width = src.width;
            image.height = src.height;
            image.channels = src.channels;
            image.data.assign(src.data, src.data + static_cast<size_t>(src.width) * src.height * src.channels);
            out.images.push_back(std::move(image));
            out.seeds.push_back(result.seeds[i]);
        }
        runtime_->api().freeResult(&result);
        request.seed = out.seeds.empty() ? request.seed : out.seeds.front();
        out.request = std::move(request);
        return out;
    }

private:
    std::shared_ptr<Runtime> runtime_;
    gh_model* handle_;
    ModelInfo info_;
};

class RuntimeEngine final : public IImageEngine {
public:
    explicit RuntimeEngine(std::shared_ptr<Runtime> runtime) : runtime_(std::move(runtime)) {
        // the engine's warnings and errors explain failed loads; this also puts them into the application log
        runtime_->api().setLogCallback([](int32_t level, const char* text, void*) { log::nativeLog(level, text); },
                                       nullptr);
    }

    std::string runtimeId() const override { return runtime_->id(); }

    std::unique_ptr<ILoadedModel> load(const LoadOptions& options) override {
        if (const auto errors = validate(options); !errors.empty()) {
            std::string message = "invalid load options:";
            for (const auto& e : errors) {
                message += " " + e.field + " " + e.message + ";";
            }
            throw EngineError(EngineErrorCode::InvalidRequest, message);
        }
        if (!std::filesystem::exists(options.modelPath)) {
            throw EngineError(EngineErrorCode::NotFound, "model file not found: " + options.modelPath);
        }
        if (options.device.rfind("vulkan", 0) == 0 && runtime_->backend() != "vulkan") {
            throw EngineError(EngineErrorCode::LoadFailed, "device " + options.device + " needs the Vulkan runtime, but the " +
                                                              runtime_->id() + " runtime is loaded");
        }

        gh_load_params params{};
        params.struct_size = sizeof(params);
        params.model_path = options.modelPath.c_str();
        params.vae_path = options.vaePath.empty() ? nullptr : options.vaePath.c_str();
        params.backend = options.device == "auto" ? nullptr : options.device.c_str();
        params.threads = options.threads;
        params.flash_attention = options.flashAttention ? 1 : 0;
        params.offload_params_to_cpu = options.offloadParamsToCpu ? 1 : 0;
        params.clip_on_cpu = options.clipOnCpu ? 1 : 0;
        params.vae_on_cpu = options.vaeOnCpu ? 1 : 0;

        const uint64_t mark = log::nativeMark();
        gh_model* handle = runtime_->api().load(&params);
        if (!handle) {
            const std::string detail = runtime_->api().lastError();
            const auto nativeErrors = log::nativeErrorsSince(mark);
            const std::string all = detail + " " + joinNativeErrors(nativeErrors);
            throw EngineError(looksLikeOutOfMemory(all) ? EngineErrorCode::OutOfMemory : EngineErrorCode::LoadFailed,
                              "cannot load " + options.modelPath + ": " + detail +
                                  (nativeErrors.empty() ? "" : " (" + joinNativeErrors(nativeErrors) + ")"));
        }
        ModelInfo info;
        info.modelPath = options.modelPath;
        const char* family = runtime_->api().modelFamily(handle);
        info.family = family ? family : "";
        info.runtimeId = runtime_->id();
        info.device = options.device;
        info.defaults = familyDefaults(info.family);
        log::info("model loaded: " + options.modelPath + " (" + info.family + ", runtime " + info.runtimeId + ")");
        return std::make_unique<RuntimeModel>(runtime_, handle, std::move(info));
    }

private:
    std::shared_ptr<Runtime> runtime_;
};

}  // namespace

std::unique_ptr<IImageEngine> makeRuntimeEngine(std::shared_ptr<Runtime> runtime) {
    return std::make_unique<RuntimeEngine>(std::move(runtime));
}

}  // namespace gh::engine
