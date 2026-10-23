/*
 * GraphicsHost runtime ABI.
 *
 * Every sd-*.dll (sd-cpu-avx2, sd-cpu-noavx, sd-vulkan) exports exactly this plain C interface. The host loads one of
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

typedef struct gh_model gh_model;

/* Describes the runtime itself. Pointers stay valid until the DLL is unloaded. */
typedef struct gh_runtime_info {
    uint32_t struct_size;
    int32_t abi_version;        /* GH_ABI_VERSION the DLL was built with */
    const char* id;             /* "cpu-avx2", "cpu-noavx", "vulkan" */
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

#ifdef __cplusplus
}
#endif

#endif /* GH_RUNTIME_H */
