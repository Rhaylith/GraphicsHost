// Implementation of the GraphicsHost runtime ABI (gh_runtime.h) on top of stable-diffusion.cpp.
// This is the only first-party code that includes the engine headers.
#define GH_RUNTIME_BUILD
#include "gh_runtime.h"

#include <stable-diffusion.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#ifndef GH_RUNTIME_ID
#  error "define GH_RUNTIME_ID (e.g. \"cpu-avx2\") and GH_RUNTIME_BACKEND (\"cpu\" or \"vulkan\")"
#endif

struct gh_model {
    sd_ctx_t* ctx = nullptr;
    std::string family;
};

namespace {

thread_local std::string g_last_error;

// Progress and log hooks are process-wide in the engine, so they are plain globals guarded by a mutex.
std::mutex g_hook_mutex;
gh_progress_fn g_progress = nullptr;
void* g_progress_user = nullptr;
gh_log_fn g_log = nullptr;
void* g_log_user = nullptr;

int32_t fail(gh_status status, std::string message) {
    g_last_error = std::move(message);
    return status;
}

void on_progress(int step, int steps, float seconds, void*) {
    gh_progress_fn fn = nullptr;
    void* user = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hook_mutex);
        fn = g_progress;
        user = g_progress_user;
    }
    if (fn) {
        fn(step, steps, seconds, user);
    }
}

// The engine prints its own progress bar to stderr when no callback is set; this keeps it quiet between calls.
void ignore_progress(int, int, float, void*) {}

void on_log(sd_log_level_t level, const char* text, void*) {
    gh_log_fn fn = nullptr;
    void* user = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hook_mutex);
        fn = g_log;
        user = g_log_user;
    }
    if (fn) {
        fn(static_cast<int32_t>(level), text, user);
    }
}

bool has_field(uint32_t struct_size, size_t offset, size_t size) {
    return struct_size >= offset + size;
}

std::string module_backend_spec(const gh_load_params& p, const std::string& base) {
    std::string spec = "all=" + base;
    if (p.clip_on_cpu) {
        spec += ",te=cpu";
    }
    if (p.vae_on_cpu) {
        spec += ",vae=cpu";
    }
    return spec;
}

}  // namespace

extern "C" {

GH_API int32_t gh_abi_version(void) {
    return GH_ABI_VERSION;
}

GH_API int32_t gh_runtime_get_info(gh_runtime_info* out) {
    if (!out || out->struct_size < sizeof(uint32_t)) {
        return fail(GH_ERR_INVALID_ARGUMENT, "gh_runtime_get_info: null or uninitialised struct");
    }
    const uint32_t size = out->struct_size;
    gh_runtime_info info{};
    info.struct_size = size;
    info.abi_version = GH_ABI_VERSION;
    info.id = GH_RUNTIME_ID;
    info.backend = GH_RUNTIME_BACKEND;
    info.engine_version = sd_version();
    info.engine_commit = sd_commit();
    info.system_info = sd_get_system_info();
    info.features = GH_FEATURE_IMG2IMG;
    std::memcpy(out, &info, size < sizeof(info) ? size : sizeof(info));
    return GH_OK;
}

GH_API const char* gh_last_error(void) {
    return g_last_error.c_str();
}

GH_API size_t gh_list_devices(char* buffer, size_t buffer_size) {
    return sd_list_devices(buffer, buffer_size);
}

GH_API void gh_set_log_callback(gh_log_fn fn, void* user) {
    {
        std::lock_guard<std::mutex> lock(g_hook_mutex);
        g_log = fn;
        g_log_user = user;
    }
    sd_set_log_callback(fn ? on_log : nullptr, nullptr);
}

GH_API gh_model* gh_load(const gh_load_params* params) {
    if (!params || !has_field(params->struct_size, offsetof(gh_load_params, model_path), sizeof(params->model_path)) ||
        !params->model_path || !*params->model_path) {
        fail(GH_ERR_INVALID_ARGUMENT, "gh_load: model_path is required");
        return nullptr;
    }
    try {
        // Every string given to the engine must outlive new_sd_ctx.
        const std::string default_backend = std::strcmp(GH_RUNTIME_BACKEND, "vulkan") == 0 ? "vulkan0" : "cpu";
        const std::string base = (params->backend && *params->backend) ? params->backend : default_backend;
        const bool split = params->clip_on_cpu || params->vae_on_cpu;
        const std::string backend = split ? module_backend_spec(*params, base) : base;
        const std::string params_backend = params->offload_params_to_cpu ? "cpu" : "";
        const std::string vae = params->vae_path ? params->vae_path : "";

        sd_ctx_params_t cp;
        sd_ctx_params_init(&cp);
        cp.model_path = params->model_path;
        if (!vae.empty()) {
            cp.vae_path = vae.c_str();
        }
        if ((params->backend && *params->backend) || split || std::strcmp(GH_RUNTIME_BACKEND, "vulkan") == 0) {
            cp.backend = backend.c_str();
        }
        if (!params_backend.empty()) {
            cp.params_backend = params_backend.c_str();
        }
        if (params->threads > 0) {
            cp.n_threads = params->threads;
        }
        cp.flash_attn = params->flash_attention != 0;

        sd_set_progress_callback(ignore_progress, nullptr);
        sd_ctx_t* ctx = new_sd_ctx(&cp);
        if (!ctx) {
            fail(GH_ERR_LOAD_FAILED, std::string("failed to load model: ") + params->model_path);
            return nullptr;
        }
        auto model = std::make_unique<gh_model>();
        model->ctx = ctx;
        const char* family = sd_get_model_version_name(ctx);
        model->family = family ? family : "";
        return model.release();
    } catch (const std::exception& e) {
        fail(GH_ERR_INTERNAL, e.what());
        return nullptr;
    } catch (...) {
        fail(GH_ERR_INTERNAL, "unknown exception in gh_load");
        return nullptr;
    }
}

GH_API void gh_free(gh_model* model) {
    if (!model) {
        return;
    }
    if (model->ctx) {
        free_sd_ctx(model->ctx);
    }
    delete model;
}

GH_API const char* gh_model_family(const gh_model* model) {
    return model ? model->family.c_str() : nullptr;
}

GH_API int32_t gh_generate(gh_model* model, const gh_generate_params* params, gh_progress_fn progress, void* user,
                           gh_result* out) {
    if (!model || !model->ctx || !params || !out || !params->prompt) {
        return fail(GH_ERR_INVALID_ARGUMENT, "gh_generate: model, params, prompt and out are required");
    }
    if (params->width <= 0 || params->height <= 0 || params->width % 8 != 0 || params->height % 8 != 0) {
        return fail(GH_ERR_INVALID_ARGUMENT, "gh_generate: width and height must be positive multiples of 8");
    }
    try {
        sd_img_gen_params_t gp;
        sd_img_gen_params_init(&gp);
        gp.prompt = params->prompt;
        gp.negative_prompt = params->negative_prompt ? params->negative_prompt : "";
        gp.width = params->width;
        gp.height = params->height;
        gp.batch_count = params->batch_count > 0 ? params->batch_count : 1;
        gp.clip_skip = params->clip_skip;
        gp.vae_tiling_params.enabled = params->vae_tiling != 0;
        gp.sample_params.sample_steps = params->steps > 0 ? params->steps : gp.sample_params.sample_steps;
        gp.sample_params.guidance.txt_cfg = params->cfg_scale;

        // image to image (ABI fields appended in 5.2: only read when the caller's struct has them)
        std::vector<uint8_t> initPixels;
        if (has_field(params->struct_size, offsetof(gh_generate_params, strength), sizeof(params->strength)) && params->init_image &&
            params->init_image->data && params->init_image->width > 0 && params->init_image->height > 0) {
            const gh_image& in = *params->init_image;
            if (in.channels != 3 && in.channels != 4) {
                return fail(GH_ERR_INVALID_ARGUMENT, "gh_generate: init_image must have 3 or 4 channels");
            }
            if (static_cast<int32_t>(in.width) != params->width || static_cast<int32_t>(in.height) != params->height) {
                return fail(GH_ERR_INVALID_ARGUMENT, "gh_generate: init_image must have the size of the result (width x height)");
            }
            // the engine wants RGB: drop an alpha channel
            const size_t pixels = static_cast<size_t>(in.width) * in.height;
            initPixels.resize(pixels * 3);
            for (size_t i = 0; i < pixels; ++i) {
                std::memcpy(&initPixels[i * 3], &in.data[i * in.channels], 3);
            }
            gp.init_image = sd_image_t{in.width, in.height, 3, initPixels.data()};
            gp.strength = params->strength;
        }

        sample_method_t method = sd_get_default_sample_method(model->ctx);
        if (params->sampler && *params->sampler) {
            method = str_to_sample_method(params->sampler);
            if (method == SAMPLE_METHOD_COUNT) {
                return fail(GH_ERR_INVALID_ARGUMENT, std::string("unknown sampler: ") + params->sampler);
            }
        }
        scheduler_t scheduler = sd_get_default_scheduler(model->ctx, method);
        if (params->scheduler && *params->scheduler) {
            scheduler = str_to_scheduler(params->scheduler);
            if (scheduler == SCHEDULER_COUNT) {
                return fail(GH_ERR_INVALID_ARGUMENT, std::string("unknown scheduler: ") + params->scheduler);
            }
        }
        gp.sample_params.sample_method = method;
        gp.sample_params.scheduler = scheduler;

        int64_t seed = params->seed;
        if (seed < 0) {
            std::random_device rd;
            seed = static_cast<int64_t>((static_cast<uint64_t>(rd()) << 31) ^ rd()) & 0x7fffffffffffffffLL;
        }
        gp.seed = seed;

        {
            std::lock_guard<std::mutex> lock(g_hook_mutex);
            g_progress = progress;
            g_progress_user = user;
        }
        sd_set_progress_callback(progress ? on_progress : ignore_progress, nullptr);
        sd_cancel_generation(model->ctx, SD_CANCEL_RESET);

        const auto t0 = std::chrono::steady_clock::now();
        sd_image_t* images = nullptr;
        int count = 0;
        const bool ok = generate_image(model->ctx, &gp, &images, &count);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        {
            std::lock_guard<std::mutex> lock(g_hook_mutex);
            g_progress = nullptr;
            g_progress_user = nullptr;
        }

        if (!ok || !images || count <= 0) {
            free_sd_images(images, count);
            return fail(GH_ERR_GENERATION_FAILED, "generation failed or was cancelled");
        }

        // Copy into memory owned by this DLL so gh_free_result releases it with the matching allocator.
        std::memset(out, 0, sizeof(*out));
        out->struct_size = sizeof(*out);
        out->image_count = count;
        out->generate_seconds = seconds;
        out->images = new gh_image[count]();
        out->seeds = new int64_t[count];
        for (int i = 0; i < count; ++i) {
            const sd_image_t& src = images[i];
            const size_t bytes = static_cast<size_t>(src.width) * src.height * src.channel;
            gh_image& dst = out->images[i];
            dst.width = src.width;
            dst.height = src.height;
            dst.channels = src.channel;
            dst.data = new uint8_t[bytes];
            std::memcpy(dst.data, src.data, bytes);
            out->seeds[i] = seed + i;
        }
        free_sd_images(images, count);
        return GH_OK;
    } catch (const std::exception& e) {
        return fail(GH_ERR_INTERNAL, e.what());
    } catch (...) {
        return fail(GH_ERR_INTERNAL, "unknown exception in gh_generate");
    }
}

GH_API void gh_cancel(gh_model* model) {
    if (model && model->ctx) {
        sd_cancel_generation(model->ctx, SD_CANCEL_ALL);
    }
}

GH_API void gh_free_result(gh_result* result) {
    if (!result) {
        return;
    }
    if (result->images) {
        for (int i = 0; i < result->image_count; ++i) {
            delete[] result->images[i].data;
        }
        delete[] result->images;
    }
    delete[] result->seeds;
    std::memset(result, 0, sizeof(*result));
}

}  // extern "C"
