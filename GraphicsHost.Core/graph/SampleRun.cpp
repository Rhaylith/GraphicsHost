#include "graph/SampleRun.h"

#include "sampling/BrownianTree.h"
#include "sampling/Guidance.h"
#include "sampling/Preview.h"
#include "sampling/Rng.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace gh::graph {

namespace {

sampling::RngKind rngOf(const std::string& wanted, const std::string& fallback) {
    return sampling::parseRng(wanted == "setting default" ? fallback : wanted).value_or(sampling::RngKind::CpuCompatible);
}

}  // namespace

SampleResult runSampling(const SampleRequest& req) {
    if (!req.inference) throw NodeError("sampling needs a model: the job runs without an inference service");
    if (!sampling::samplerAvailable(req.sampler.name)) throw NodeError("the sampler '" + req.sampler.name + "' is not available in this version", "sampler");
    const auto model = req.guider.model.asObject<ModelHandle>();
    const sampling::ModelSampling ms(model->sampling.value_or(sampling::SamplingParams{}));

    const tensor::Tensor& latent = req.latent;
    if (latent.ndim() != 4) throw NodeError("a latent has the shape [batch, channels, height, width]", "latent");
    LatentView view;
    view.batch = latent.dim(0);
    view.channels = latent.dim(1);
    view.height = latent.dim(2);
    view.width = latent.dim(3);
    const size_t n = static_cast<size_t>(latent.numel());
    const size_t perSample = n / static_cast<size_t>(std::max<int64_t>(view.batch, 1));

    // the part of the schedule to run
    const int total = static_cast<int>(req.sigmas.size()) - 1;
    if (total < 1) return {latent, latent};
    const int first = std::clamp(req.firstStep, 0, total);
    const int last = req.lastStep < 0 ? total : std::clamp(req.lastStep, first, total);
    if (first >= last) return {latent, latent};
    std::vector<float> sigmas(req.sigmas.begin() + first, req.sigmas.begin() + last + 1);
    if (last < total && !req.returnWithLeftoverNoise) sigmas.back() = 0.0f;   // an early stop still ends clean

    // the starting point: the latent with the seed's noise added, or the latent as it is
    sampling::Latent x(latent.values());
    if (!req.noise.none) {
        sampling::NormalGenerator gen(rngOf(req.noise.rng, req.defaultRng), req.noise.seed);
        const std::vector<float> noise = gen.normal(n);
        ms.scaleNoise(sigmas.front(), noise.data(), latent.data(), x.data(), n);
    }
    const sampling::Latent original(latent.values());

    // noise for the ancestral and SDE samplers, kept apart from the starting noise
    const uint64_t stepSeed = req.noise.seed ^ 0x9E3779B97F4A7C15ull;
    const sampling::RngKind kind = rngOf(req.sampler.generator, req.defaultRng);
    auto stepGen = std::make_shared<sampling::NormalGenerator>(kind, stepSeed);
    float sigmaMin = 0, sigmaMax = 0;
    for (const float s : sigmas) {
        sigmaMax = std::max(sigmaMax, s);
        if (s > 0) sigmaMin = sigmaMin == 0 ? s : std::min(sigmaMin, s);
    }
    if (sigmaMin == 0) sigmaMin = 1e-4f;
    auto tree = std::make_shared<sampling::BrownianTree>(sigmaMin, std::max(sigmaMax, sigmaMin * 2), n, stepSeed, kind);

    sampling::SamplerEnv env;
    env.noise = [stepGen](float* out, size_t count, int, float, float) { stepGen->fill(out, count); };
    env.brownian = [tree](float* out, size_t count, float a, float b) {
        const std::vector<float> v = tree->noise(a, b);
        std::copy_n(v.begin(), std::min(count, v.size()), out);
    };

    // the guided prediction: the guider decides which conditionings are asked for and how the answers are combined
    sampling::GuidanceState guidanceState;
    sampling::Latent uncond;
    sampling::Latent lastDenoised(n, 0.0f);
    const GuiderHandle& g = req.guider;
    const auto call = [&](const Value& cond, const sampling::Latent& at, float sigma, std::vector<float>& out) {
        LatentView v = view;
        v.data = at.data();
        req.inference->denoise(*model, cond, v, sigma, out);
        if (out.size() != n) throw NodeError("the model returned a latent of the wrong size");
    };
    env.denoise = [&](const sampling::Latent& at, float sigma, int, sampling::Latent& out) {
        if (req.stop.stop_requested()) return false;
        out.assign(n, 0.0f);
        LatentView v = view;
        v.data = at.data();
        switch (g.kind) {
            case GuiderHandle::Kind::Basic: call(g.conditioning.at(0), at, sigma, out); break;
            case GuiderHandle::Kind::Cfg: {
                const bool needUncond = g.cfg != 1.0 || model->guidance.apg || model->guidance.zeroStar || model->guidance.zeroInitSteps > 0;
                const bool plainCfg = !model->guidance.rescale && !model->guidance.apg && !model->guidance.zeroStar && model->guidance.zeroInitSteps == 0 && req.sampler.name.find("cfg++") == std::string::npos;
                if (needUncond && plainCfg) {
                    req.inference->denoiseGuided(*model, g.conditioning.at(0), g.conditioning.at(1), static_cast<float>(g.cfg), v, sigma, out);
                    if (out.size() != n) throw NodeError("the model returned a latent of the wrong size");
                    ++guidanceState.step;
                } else if (!needUncond) {
                    call(g.conditioning.at(0), at, sigma, out);
                    uncond = out;
                    ++guidanceState.step;
                } else {
                    std::vector<float> cond;
                    req.inference->denoisePair(*model, g.conditioning.at(0), g.conditioning.at(1), v, sigma, cond, uncond);
                    if (cond.size() != n || uncond.size() != n) throw NodeError("the model returned a latent of the wrong size");
                    sampling::guide(model->guidance, guidanceState, at.data(), cond.data(), uncond.data(), g.cfg, perSample, static_cast<size_t>(view.batch), out.data());
                }
                break;
            }
            case GuiderHandle::Kind::DualCfg: {
                std::vector<float> a, b, neg;
                call(g.conditioning.at(0), at, sigma, a);
                call(g.conditioning.at(1), at, sigma, b);
                call(g.conditioning.at(2), at, sigma, neg);
                sampling::dualCfg(a.data(), b.data(), neg.data(), g.cfgA, g.cfgB, g.nested, out.data(), n);
                uncond = neg;
                break;
            }
            case GuiderHandle::Kind::PerpNeg: {
                std::vector<float> pos, neg, empty;
                call(g.conditioning.at(0), at, sigma, pos);
                call(g.conditioning.at(1), at, sigma, neg);
                call(g.conditioning.at(2), at, sigma, empty);
                sampling::perpNegCfg(pos.data(), neg.data(), empty.data(), g.cfg, g.negativeScale, perSample, static_cast<size_t>(view.batch), out.data());
                uncond = empty;
                break;
            }
        }
        if (req.noiseMask) {
            // inpainting: outside the mask the estimate is the original picture
            const tensor::Tensor& m = *req.noiseMask;
            const size_t mn = static_cast<size_t>(m.numel());
            for (size_t i = 0; i < n; ++i) {
                const float w = mn == n ? m.data()[i] : m.data()[(i / perSample) * (mn / static_cast<size_t>(view.batch)) + (i % perSample) % (mn / static_cast<size_t>(view.batch))];
                out[i] = out[i] * w + original[i] * (1.0f - w);
            }
        }
        lastDenoised = out;
        return true;
    };
    env.uncond = &uncond;
    const int steps = static_cast<int>(sigmas.size()) - 1;
    env.onStep = [&](int step, int) {
        if (req.progress) req.progress(step, steps);
        if (req.emitPreview && req.previewMethod != "none" && req.previewEvery > 0 && step % req.previewEvery == 0)
            req.emitPreview(sampling::previewImage(tensor::Tensor(latent.shape(), lastDenoised), model->file.family, req.previewMaxSize));
    };

    const bool ran = sampling::runSampler(req.sampler.name, x, sigmas, env, req.sampler.options);
    if (!ran) {
        if (req.stop.stop_requested()) throw Interrupted();
        throw NodeError("the sampler stopped before the end");
    }
    SampleResult result;
    result.output = tensor::Tensor(latent.shape(), std::move(x));
    result.denoised = tensor::Tensor(latent.shape(), std::move(lastDenoised));
    return result;
}

}  // namespace gh::graph
