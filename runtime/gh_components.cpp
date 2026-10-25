// The component functions of the runtime ABI (gh_runtime.h, GH_FEATURE_COMPONENTS): the pipeline in pieces. This is the
// second first-party file that looks inside the engine, so it is compiled with the engine's own include paths and
// language standard (see Runtime.props). The spike of workflow plan step W0 lives here.
#define GH_RUNTIME_BUILD
#include "gh_runtime.h"
#include "gh_internal.h"

#include "pipeline/diffusion_engine.h"
#include "pipeline/request.h"
#include "conditioning/conditioner.hpp"
#include "model/diffusion/model.hpp"
#include "runtime/denoiser.hpp"

#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

// stable-diffusion.cpp defines this struct privately; the layout is a single pointer.
struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

namespace {

thread_local std::string g_error;

int32_t fail(gh_status status, std::string message) {
    g_error = std::move(message);
    return status;
}

StableDiffusionGGML* engine_of(gh_model* model) {
    return model && model->ctx ? model->ctx->sd : nullptr;
}

// The engine runs everything inside an "execution scope" (it refreshes the model sources and releases the runners at the
// end). A session keeps one scope open across many calls; without one every call opens its own.
using Scope = StableDiffusionGGML::ExecutionScope;

struct ScopeHolder {
    std::unique_ptr<Scope> own;
    bool ready = false;
    ScopeHolder(gh_model* model, StableDiffusionGGML* sd) {
        if (model->session) {
            ready = true;
        } else {
            own = std::make_unique<Scope>(*sd);
            ready = own->ready;
        }
    }
};

sd::Tensor<float> from_gh(const gh_tensor& t) {
    std::vector<int64_t> shape(t.shape, t.shape + t.ndim);
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    return sd::Tensor<float>(std::move(shape), std::vector<float>(t.data, t.data + n));
}

void to_gh(const sd::Tensor<float>& t, gh_tensor* out) {
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(*out);
    out->ndim = static_cast<int32_t>(t.dim());
    for (int i = 0; i < out->ndim && i < 5; ++i) out->shape[i] = t.shape()[i];
    const size_t n = static_cast<size_t>(t.numel());
    out->data = new float[n ? n : 1];
    if (n) std::memcpy(out->data, t.data(), n * sizeof(float));
}

}  // namespace

extern "C" {

GH_API int32_t gh_session_begin(gh_model* model) {
    StableDiffusionGGML* sd = engine_of(model);
    if (!sd) return fail(GH_ERR_INVALID_ARGUMENT, "gh_session_begin: no model");
    if (model->session) return GH_OK;
    auto* scope = new Scope(*sd);
    if (!scope->ready) {
        delete scope;
        return fail(GH_ERR_GENERATION_FAILED, "the engine could not start a session (another call is running?)");
    }
    sd->reset_cancel_flag();
    model->session = scope;
    return GH_OK;
}

GH_API void gh_session_end(gh_model* model) {
    if (model && model->session) {
        delete static_cast<Scope*>(model->session);
        model->session = nullptr;
    }
}

GH_API int32_t gh_text_encode(gh_model* model, const char* text, int32_t clip_skip, int32_t width, int32_t height, gh_tensor* hidden,
                              gh_tensor* pooled) {
    StableDiffusionGGML* sd = engine_of(model);
    if (!sd || !hidden) return fail(GH_ERR_INVALID_ARGUMENT, "gh_text_encode: model and hidden are required");
    ScopeHolder scope(model, sd);
    if (!scope.ready) return fail(GH_ERR_GENERATION_FAILED, "the engine is busy");
    ConditionerParams params;
    params.text = text ? text : "";
    params.clip_skip = clip_skip;
    params.width = width;
    params.height = height;
    SDCondition cond = sd->get_learned_condition(params);
    if (cond.c_crossattn.empty()) return fail(GH_ERR_GENERATION_FAILED, "the text encoder produced nothing");
    to_gh(cond.c_crossattn, hidden);
    if (pooled) to_gh(cond.c_vector, pooled);
    return GH_OK;
}

GH_API int32_t gh_sigma_schedule(gh_model* model, const gh_generate_params* p, float** sigmas, int32_t* count) {
    StableDiffusionGGML* sd = engine_of(model);
    if (!sd || !p || !sigmas || !count) return fail(GH_ERR_INVALID_ARGUMENT, "gh_sigma_schedule: bad arguments");
    ScopeHolder scope(model, sd);
    if (!scope.ready) return fail(GH_ERR_GENERATION_FAILED, "the engine is busy");

    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    gp.width = p->width;
    gp.height = p->height;
    gp.sample_params.sample_steps = p->steps > 0 ? p->steps : gp.sample_params.sample_steps;
    sample_method_t method = sd_get_default_sample_method(model->ctx);
    if (p->sampler && *p->sampler) {
        method = str_to_sample_method(p->sampler);
        if (method == SAMPLE_METHOD_COUNT) return fail(GH_ERR_INVALID_ARGUMENT, std::string("unknown sampler: ") + p->sampler);
    }
    scheduler_t scheduler = sd_get_default_scheduler(model->ctx, method);
    if (p->scheduler && *p->scheduler) {
        scheduler = str_to_scheduler(p->scheduler);
        if (scheduler == SCHEDULER_COUNT) return fail(GH_ERR_INVALID_ARGUMENT, std::string("unknown scheduler: ") + p->scheduler);
    }
    gp.sample_params.sample_method = method;
    gp.sample_params.scheduler = scheduler;
    gp.seed = p->seed;

    sd::pipeline::GenerationRequest request(sd, &gp);
    sd::pipeline::SamplePlan plan(sd, &gp, request);
    *count = static_cast<int32_t>(plan.sigmas.size());
    *sigmas = new float[plan.sigmas.size()];
    std::memcpy(*sigmas, plan.sigmas.data(), plan.sigmas.size() * sizeof(float));
    return GH_OK;
}

GH_API int32_t gh_initial_latent(gh_model* model, int64_t seed, int32_t width, int32_t height, float sigma0, gh_tensor* latent) {
    StableDiffusionGGML* sd = engine_of(model);
    if (!sd || !latent) return fail(GH_ERR_INVALID_ARGUMENT, "gh_initial_latent: model and latent are required");
    ScopeHolder scope(model, sd);
    if (!scope.ready) return fail(GH_ERR_GENERATION_FAILED, "the engine is busy");
    sd->rng->manual_seed(seed);
    sd->sampler_rng->manual_seed(seed);
    sd::Tensor<float> init = sd->generate_init_latent(width, height);
    sd::Tensor<float> processed = sd->denoiser->process_latent_in(init);
    const sd::Tensor<float>& base = processed.empty() ? init : processed;
    sd::Tensor<float> noise = sd::randn_like<float>(init, sd->rng);
    sd::Tensor<float> x_t = sd->denoiser->noise_scaling(sigma0, noise, base);
    to_gh(x_t, latent);
    return GH_OK;
}

GH_API int32_t gh_denoise(gh_model* model, const gh_denoise_args* a, gh_tensor* out) {
    StableDiffusionGGML* sd = engine_of(model);
    if (!sd || !a || !a->x || !a->cond_hidden || !out) return fail(GH_ERR_INVALID_ARGUMENT, "gh_denoise: model, x, cond_hidden and out are required");
    ScopeHolder scope(model, sd);
    if (!scope.ready) return fail(GH_ERR_GENERATION_FAILED, "the engine is busy");
    if (sd->get_cancel_flag() == SD_CANCEL_ALL) return fail(GH_ERR_CANCELLED, "cancelled");

    const sd::Tensor<float> x = from_gh(*a->x);
    const std::vector<float> scaling = sd->denoiser->get_scalings(a->sigma);
    const float c_skip = scaling[0], c_out = scaling[1], c_in = scaling[2];
    const std::vector<float> ts = sd->prepare_sample_timesteps(a->sigma, 0);
    const sd::Tensor<float> noised = x * c_in;
    const sd::Tensor<float> timesteps({static_cast<int64_t>(ts.size())}, ts);

    auto evaluate = [&](const gh_tensor* hidden, const gh_tensor* pooled) -> sd::Tensor<float> {
        const sd::Tensor<float> context = from_gh(*hidden);
        sd::Tensor<float> y;
        if (pooled && pooled->ndim > 0) y = from_gh(*pooled);
        std::vector<sd::Tensor<float>> controls;
        DiffusionParams params;
        params.x = &noised;
        params.timesteps = &timesteps;
        params.context = &context;
        params.y = y.empty() ? nullptr : &y;
        params.extra = UNetDiffusionExtra{-1, &controls, 0.f};
        return sd->diffusion_model->compute(sd->n_threads, params);
    };

    sd::Tensor<float> guided = evaluate(a->cond_hidden, a->cond_pooled);
    if (guided.empty()) return fail(GH_ERR_GENERATION_FAILED, "the diffusion model failed");
    if (a->uncond_hidden) {
        const sd::Tensor<float> uncond = evaluate(a->uncond_hidden, a->uncond_pooled);
        if (uncond.empty()) return fail(GH_ERR_GENERATION_FAILED, "the diffusion model failed");
        guided = uncond + a->cfg_scale * (guided - uncond);
    }
    to_gh(guided * c_out + x * c_skip, out);
    return GH_OK;
}

GH_API int32_t gh_vae_decode(gh_model* model, const gh_tensor* latent, gh_tensor* image) {
    StableDiffusionGGML* sd = engine_of(model);
    if (!sd || !latent || !image) return fail(GH_ERR_INVALID_ARGUMENT, "gh_vae_decode: model, latent and image are required");
    ScopeHolder scope(model, sd);
    if (!scope.ready) return fail(GH_ERR_GENERATION_FAILED, "the engine is busy");
    sd::Tensor<float> x = from_gh(*latent);
    x = sd->denoiser->process_latent_out(std::move(x));
    sd::Tensor<float> decoded = sd->decode_first_stage(x);
    if (decoded.empty()) return fail(GH_ERR_GENERATION_FAILED, "the VAE failed");
    to_gh(decoded, image);
    return GH_OK;
}

GH_API void gh_tensor_free(gh_tensor* tensor) {
    if (tensor && tensor->data) {
        delete[] tensor->data;
        std::memset(tensor, 0, sizeof(*tensor));
    }
}

GH_API void gh_float_array_free(float* values) {
    delete[] values;
}

}  // extern "C"
