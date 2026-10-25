// Phase 0 spike: generate one image with stable-diffusion.cpp on a chosen backend and report timings.
//
//   spike --list-devices
//   spike --model <file> --prompt "<text>" [--backend cpu|vulkan0] [--steps N] [--size WxH]
//         [--cfg F] [--seed N] [--threads N] [--out file.png] [--vae-tiling]
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <stable-diffusion.h>
#include <ggml.h>
#include <ggml-backend.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static void on_log(sd_log_level_t level, const char* text, void*) {
    static const bool verbose = std::getenv("SPIKE_VERBOSE") != nullptr;
    if (verbose || level >= SD_LOG_INFO) {
        std::fputs(text, stderr);
    }
}

static void on_progress(int step, int steps, float time, void*) {
    std::fprintf(stderr, "  step %d/%d  %.2fs/it\n", step, steps, time);
}

int main(int argc, char** argv) {
    std::string model, prompt = "a lighthouse on a cliff at sunset, detailed, oil painting";
    std::string negative, backend = "cpu", out = "spike.png";
    int steps = 20, width = 512, height = 512, threads = 0;
    float cfg = 7.0f;
    int64_t seed = 1;
    bool list = false, vae_tiling = false, flash_attn = false, offload = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--list-devices") list = true;
        else if (a == "--model") model = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--negative") negative = next();
        else if (a == "--backend") backend = next();
        else if (a == "--steps") steps = std::atoi(next());
        else if (a == "--cfg") cfg = (float)std::atof(next());
        else if (a == "--seed") seed = std::atoll(next());
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--out") out = next();
        else if (a == "--vae-tiling") vae_tiling = true;
        else if (a == "--flash-attn") flash_attn = true;
        else if (a == "--offload") offload = true;
        else if (a == "--size") std::sscanf(next(), "%dx%d", &width, &height);
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }

    sd_set_log_callback(on_log, nullptr);
    sd_set_progress_callback(on_progress, nullptr);
    if (std::getenv("SPIKE_VERBOSE")) {
        // backend registration and device probing happen before sd's own ggml log hook is installed
        ggml_log_set([](ggml_log_level, const char* text, void*) { std::fputs(text, stderr); }, nullptr);
    }
    std::fprintf(stderr, "stable-diffusion.cpp %s (%s)\n%s\n", sd_version(), sd_commit(), sd_get_system_info());

    if (list && std::getenv("SPIKE_VERBOSE")) {
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t r = ggml_backend_reg_get(i);
            std::fprintf(stderr, "backend reg %zu: %s (%zu devices)\n", i, ggml_backend_reg_name(r), ggml_backend_reg_dev_count(r));
        }
    }
    if (list) {
        size_t need = sd_list_devices(nullptr, 0);
        std::string buf(need + 1, '\0');
        sd_list_devices(buf.data(), buf.size());
        std::printf("%s\n", buf.c_str());
        return 0;
    }
    if (model.empty()) {
        std::fprintf(stderr, "--model is required\n");
        return 2;
    }

    sd_ctx_params_t cp;
    sd_ctx_params_init(&cp);
    cp.model_path = model.c_str();
    cp.backend = backend.c_str();
    if (threads > 0) cp.n_threads = threads;
    cp.flash_attn = flash_attn;
    if (offload) cp.params_backend = "cpu";

    auto t0 = Clock::now();
    sd_ctx_t* ctx = new_sd_ctx(&cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const double load_s = seconds_since(t0);
    std::fprintf(stderr, "model version: %s\n", sd_get_model_version_name(ctx));

    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    gp.prompt = prompt.c_str();
    gp.negative_prompt = negative.c_str();
    gp.width = width;
    gp.height = height;
    gp.seed = seed;
    gp.batch_count = 1;
    gp.sample_params.sample_steps = steps;
    gp.sample_params.guidance.txt_cfg = cfg;
    gp.vae_tiling_params.enabled = vae_tiling;

    t0 = Clock::now();
    sd_image_t* images = nullptr;
    int count = 0;
    const bool ok = generate_image(ctx, &gp, &images, &count);
    const double gen_s = seconds_since(t0);

    int rc = 1;
    if (ok && images && count > 0) {
        const sd_image_t& im = images[0];
        if (stbi_write_png(out.c_str(), (int)im.width, (int)im.height, (int)im.channel, im.data, (int)(im.width * im.channel))) {
            rc = 0;
        }
        std::printf("RESULT backend=%s size=%ux%u steps=%d load_s=%.2f generate_s=%.2f s_per_step=%.3f out=%s\n",
                    backend.c_str(), im.width, im.height, steps, load_s, gen_s, gen_s / steps, out.c_str());
    } else {
        std::fprintf(stderr, "generation failed\n");
    }
    free_sd_images(images, count);
    free_sd_ctx(ctx);
    return rc;
}
