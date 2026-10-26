/*
 * GraphicsHost runtime ABI.
 *
 * Every sd-*.dll (sd-cpu-avx2, sd-vulkan) exports exactly this plain C interface. The host loads one of
 * them with LoadLibrary and never includes ggml or stable-diffusion headers, so the runtime can be replaced by a newer
 * build without recompiling the host.
 *
 * Rules:
 *  - Structs start with struct_size so fields can be appended without breaking older callers (zero-initialise, set
 *    struct_size = sizeof(...), fill what you need).
 *  - Strings are UTF-8, owned by the caller for the duration of the call.
 *  - Functions return 0 on success and a negative gh_status on failure; gh_last_error() then gives a message for the
 *    calling thread.
 *  - gh_generate must not be called concurrently from several threads (progress and log callbacks are process-wide
 *    in the engine); the host's job queue serialises generation. gh_cancel may be called from any thread.
 */
#ifndef GH_RUNTIME_H
#define GH_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef GH_RUNTIME_BUILD
#    define GH_API __declspec(dllexport)
#  else
#    define GH_API __declspec(dllimport)
#  endif
#else
#  define GH_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GH_ABI_VERSION 1

typedef enum gh_status {
    GH_OK = 0,
    GH_ERR_INVALID_ARGUMENT = -1,
    GH_ERR_LOAD_FAILED = -2,
    GH_ERR_GENERATION_FAILED = -3,
    GH_ERR_CANCELLED = -4,
    GH_ERR_INTERNAL = -5
} gh_status;

/* What a runtime can do beyond plain text-to-image (gh_runtime_info.features). */
#define GH_FEATURE_IMG2IMG 0x1u   /* gh_generate_params.init_image + strength */
#define GH_FEATURE_INPAINT 0x2u   /* gh_generate_params.mask_image (with init_image) */
#define GH_FEATURE_LORA 0x4u      /* gh_generate_params.loras */

typedef struct gh_model gh_model;

/* Describes the runtime itself. Pointers stay valid until the DLL is unloaded. */
typedef struct gh_runtime_info {
    uint32_t struct_size;
    int32_t abi_version;        /* GH_ABI_VERSION the DLL was built with */
    const char* id;             /* "cpu-avx2", "vulkan" */
    const char* backend;        /* "cpu" or "vulkan" */
    const char* engine_version; /* stable-diffusion.cpp version string */
    const char* engine_commit;
    const char* system_info;    /* CPU feature line reported by ggml */
    uint32_t features;          /* GH_FEATURE_* bits; 0 when the DLL predates the field */
} gh_runtime_info;

typedef struct gh_load_params {
    uint32_t struct_size;
    const char* model_path;     /* .safetensors / .ckpt / .gguf checkpoint */
    const char* vae_path;       /* optional separate VAE, NULL for the one inside the checkpoint */
    const char* backend;        /* "cpu", "vulkan0", ...; NULL = the runtime's default device */
    int32_t threads;            /* CPU threads, 0 = number of physical cores */
    int32_t flash_attention;    /* 0/1 */
    int32_t offload_params_to_cpu; /* keep weights in system RAM and stream them to the GPU (low VRAM) */
    int32_t clip_on_cpu;        /* run the text encoder on the CPU */
    int32_t vae_on_cpu;         /* run the VAE on the CPU */
} gh_load_params;

typedef struct gh_lora {
    const char* path;           /* .safetensors LoRA file */
    float multiplier;           /* strength, usually 0.2 to 1.2; negative values subtract the style */
} gh_lora;

typedef struct gh_generate_params {
    uint32_t struct_size;
    const char* prompt;
    const char* negative_prompt; /* may be NULL */
    int32_t width;
    int32_t height;
    int32_t steps;
    float cfg_scale;
    int64_t seed;               /* -1 = random */
    int32_t batch_count;
    const char* sampler;        /* e.g. "euler_a"; NULL = model default */
    const char* scheduler;      /* e.g. "karras"; NULL = model default */
    int32_t clip_skip;          /* -1 = model default */
    int32_t vae_tiling;         /* 0/1 */
    /* appended in step 5.2: needs GH_FEATURE_IMG2IMG */
    const struct gh_image* init_image; /* start from this picture (RGB); NULL = text to image */
    float strength;             /* 0..1: how much of init_image is replaced (1 = ignore it); used with init_image */
    /* appended in step 5.3: needs GH_FEATURE_INPAINT */
    const struct gh_image* mask_image; /* with init_image: white (255) = repaint, black (0) = keep; RGB or RGBA, same size as the result; NULL = no mask */
    /* appended in step 5.4: needs GH_FEATURE_LORA */
    const gh_lora* loras;       /* applied for this generation only; NULL = none */
    int32_t lora_count;
} gh_generate_params;

typedef struct gh_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;          /* 3 = RGB, 4 = RGBA */
    uint8_t* data;              /* width * height * channels bytes, owned by the gh_result */
} gh_image;

typedef struct gh_result {
    uint32_t struct_size;
    int32_t image_count;
    gh_image* images;           /* image_count entries */
    int64_t* seeds;             /* the seed actually used for each image */
    double generate_seconds;
} gh_result;

/* Called after every sampling step. Return value is ignored; use gh_cancel to stop. */
typedef void (*gh_progress_fn)(int32_t step, int32_t steps, float seconds_per_step, void* user);
/* level: 0 debug, 1 verbose, 2 info, 3 warning, 4 error */
typedef void (*gh_log_fn)(int32_t level, const char* text, void* user);

GH_API int32_t gh_abi_version(void);
GH_API int32_t gh_runtime_get_info(gh_runtime_info* out);
GH_API const char* gh_last_error(void);

/* Writes "name<TAB>description" lines for the ggml devices this runtime can see. Returns the bytes needed
 * (without the terminating NUL); pass NULL/0 to query the size. */
GH_API size_t gh_list_devices(char* buffer, size_t buffer_size);

GH_API void gh_set_log_callback(gh_log_fn fn, void* user);

GH_API gh_model* gh_load(const gh_load_params* params);
GH_API void gh_free(gh_model* model);
/* Name of the detected model family ("SD 1.x", "SDXL", ...). NULL if model is NULL. */
GH_API const char* gh_model_family(const gh_model* model);

GH_API int32_t gh_generate(gh_model* model, const gh_generate_params* params, gh_progress_fn progress, void* user,
                           gh_result* out);
GH_API void gh_cancel(gh_model* model);
GH_API void gh_free_result(gh_result* result);

/* ---- component functions (workflow plan, step W0): the pipeline in pieces ----------------------------------------- *
 * Spike version: the pieces of an already loaded model (gh_load). A host that wants them calls gh_session_begin once, then
 * any of the functions below as often as it likes, then gh_session_end; the model stays loaded between sessions.
 * Everything runs where the model runs (on the GPU for sd-vulkan). */
#define GH_FEATURE_COMPONENTS 0x8u

/* A float32 tensor. shape[0] varies fastest (ggml order): an SD 1.x latent is {W/8, H/8, 4, B}, a text encoding
 * {768, tokens, B}, a decoded image {W, H, 3, B} with values in 0..1. data is owned by whoever allocated it: tensors
 * filled in by the runtime are released with gh_tensor_free. */
typedef struct gh_tensor {
    uint32_t struct_size;
    int32_t ndim;
    int64_t shape[5];
    float* data;
} gh_tensor;

typedef struct gh_denoise_args {
    uint32_t struct_size;
    const gh_tensor* x;              /* the noisy latent */
    float sigma;                     /* noise level of x */
    const gh_tensor* cond_hidden;    /* text encoding of the prompt (gh_text_encode) */
    const gh_tensor* cond_pooled;    /* pooled vector (SDXL), NULL for SD 1.x */
    const gh_tensor* uncond_hidden;  /* text encoding of the negative prompt; NULL = no guidance */
    const gh_tensor* uncond_pooled;
    float cfg_scale;                 /* classifier-free guidance scale; used when uncond_hidden is given */
} gh_denoise_args;

GH_API int32_t gh_session_begin(gh_model* model);
GH_API void gh_session_end(gh_model* model);
/* hidden and pooled (may be NULL) are filled in; pooled stays empty (ndim 0) for models without one. zero_out != 0 gives the
 * all-zero encoding SDXL uses for an empty negative prompt (the engine does this when the negative prompt is empty). */
GH_API int32_t gh_text_encode(gh_model* model, const char* text, int32_t clip_skip, int32_t width, int32_t height, int32_t zero_out, gh_tensor* hidden,
                              gh_tensor* pooled);
/* The sigmas the engine would use for these generate params (steps, sampler, scheduler, size): steps + 1 values. Release the
 * array with gh_float_array_free. */
GH_API int32_t gh_sigma_schedule(gh_model* model, const gh_generate_params* params, float** sigmas, int32_t* count);
/* The starting latent for a seed: noise scaled to sigma0, drawn the way gh_generate draws it for the same seed. */
GH_API int32_t gh_initial_latent(gh_model* model, int64_t seed, int32_t width, int32_t height, float sigma0, gh_tensor* latent);
/* One guided model evaluation: the denoised estimate of x at args->sigma. */
GH_API int32_t gh_denoise(gh_model* model, const gh_denoise_args* args, gh_tensor* denoised);
/* Decodes a latent to an image tensor in 0..1 ({W, H, 3, B}). */
GH_API int32_t gh_vae_decode(gh_model* model, const gh_tensor* latent, gh_tensor* image);
GH_API void gh_tensor_free(gh_tensor* tensor);
GH_API void gh_float_array_free(float* values);

#ifdef __cplusplus
}
#endif

#endif /* GH_RUNTIME_H */
