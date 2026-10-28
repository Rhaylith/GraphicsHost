#include "sampling/BrownianTree.h"
#include "sampling/Samplers.h"
#include "sampling/Schedulers.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::sampling;
using Catch::Approx;

namespace {

// Data that is N(0, 1) per element: the ideal denoiser at noise level sigma is x / (1 + sigma^2), and the exact path of the
// probability-flow ODE is x(sigma) = x_T * sqrt((1 + sigma^2) / (1 + sigma_T^2)).
DenoiseFn gaussianData() {
    return [](const Latent& x, float sigma, int, Latent& out) {
        out.resize(x.size());
        const double f = 1.0 / (1.0 + static_cast<double>(sigma) * sigma);
        for (size_t i = 0; i < x.size(); ++i) out[i] = static_cast<float>(x[i] * f);
        return true;
    };
}

// Data that is the single point c: the denoised estimate is c whatever the noise.
DenoiseFn pointData(float c) {
    return [c](const Latent& x, float, int, Latent& out) {
        out.assign(x.size(), c);
        return true;
    };
}

std::vector<float> karras(int steps, double smax = 14.6, double smin = 0.03) { return karrasSchedule(steps, smax, smin); }

Latent startFor(const std::vector<float>& sigmas, size_t n, uint64_t seed) {
    Latent x = NormalGenerator(RngKind::Philox, seed).normal(n);
    for (float& v : x) v *= sigmas.front() * 1.0f;   // sigma_T * noise for data that is N(0,1): variance 1 + sigma_T^2 is close for large sigma_T
    return x;
}

SamplerEnv env(DenoiseFn d, uint64_t seed = 1, size_t n = 0) {
    SamplerEnv e;
    e.denoise = std::move(d);
    auto gen = std::make_shared<NormalGenerator>(RngKind::Philox, seed);
    e.noise = [gen](float* out, size_t count, int, float, float) { gen->fill(out, count); };
    auto tree = std::make_shared<BrownianTree>(0.0, 20.0, n ? n : 1, seed);
    e.brownian = [tree](float* out, size_t count, float a, float b) {
        const auto v = tree->noise(a, b);
        for (size_t i = 0; i < count && i < v.size(); ++i) out[i] = v[i];
    };
    return e;
}

double rms(const Latent& x) {
    double s = 0;
    for (float v : x) s += static_cast<double>(v) * v;
    return std::sqrt(s / static_cast<double>(x.size()));
}

double maxError(const Latent& x, const Latent& y) {
    double e = 0;
    for (size_t i = 0; i < x.size(); ++i) e = std::max(e, static_cast<double>(std::fabs(x[i] - y[i])));
    return e;
}

}  // namespace

TEST_CASE("samplers: the list matches the reference and says what runs", "[unit][sampling][samplers]") {
    const auto& names = allSamplerNames();
    CHECK(names.size() == 34);
    CHECK(names.front() == "euler");
    int available = 0;
    for (const auto& n : names) available += samplerAvailable(n);
    CHECK(available == 20);
    CHECK(samplerAvailable("dpm++ 2M"));
    CHECK_FALSE(samplerAvailable("uni pc"));
    CHECK(samplerIsStochastic("euler ancestral"));
    CHECK(samplerIsStochastic("dpm++ 3M SDE"));
    CHECK_FALSE(samplerIsStochastic("heun"));
    CHECK_FALSE(samplerIsStochastic("no such sampler"));
    Latent x(4, 1.0f);
    auto e = env(gaussianData());
    CHECK_FALSE(runSampler("uni pc", x, karras(5), e));   // not available: refused, not silently replaced
    CHECK_FALSE(runSampler("euler", x, {1.0f}, e));       // a schedule needs at least two entries
}

TEST_CASE("samplers: deterministic samplers follow the exact path of the ODE", "[unit][sampling][samplers]") {
    const size_t n = 64;
    const auto sigmas = karras(40);
    const Latent start = startFor(sigmas, n, 11);
    Latent exact(n);
    for (size_t i = 0; i < n; ++i) exact[i] = static_cast<float>(start[i] * std::sqrt(1.0 / (1.0 + sigmas.front() * sigmas.front())));   // sigma = 0 at the end
    double previous = 1e9;
    for (const char* name : {"euler", "ddim", "heun", "dpm2", "lms", "ipndm", "ipndm v", "dpm++ 2M", "res multistep"}) {
        INFO(name);
        Latent x = start;
        int steps = 0;
        auto e = env(gaussianData());
        e.onStep = [&](int step, int total) {
            ++steps;
            CHECK(step == steps);
            CHECK(total == 40);
        };
        REQUIRE(runSampler(name, x, sigmas, e));
        CHECK(steps == 40);
        const double err = maxError(x, exact);
        CHECK(err < 0.05 * rms(exact) + 0.02);
        (void)previous;
    }
    // first order converges as the steps grow; the second-order solvers are better than euler for the same steps
    const auto errorAt = [&](const char* name, int steps) {
        const auto s = karras(steps);
        Latent x = startFor(s, n, 11);
        Latent ex(n);
        for (size_t i = 0; i < n; ++i) ex[i] = static_cast<float>(x[i] * std::sqrt(1.0 / (1.0 + s.front() * s.front())));
        auto e = env(gaussianData());
        runSampler(name, x, s, e);
        return maxError(x, ex);
    };
    CHECK(errorAt("euler", 80) < errorAt("euler", 10));
    CHECK(errorAt("heun", 10) < errorAt("euler", 10));
    CHECK(errorAt("dpm++ 2M", 10) < errorAt("euler", 10));
    CHECK(errorAt("dpm2", 10) < errorAt("euler", 10));
    CHECK(errorAt("lms", 10) < errorAt("euler", 10));
}

TEST_CASE("samplers: a point-mass data distribution is reached exactly by every sampler", "[unit][sampling][samplers]") {
    const auto sigmas = karras(15);
    const Latent start = startFor(sigmas, 40, 3);
    for (const auto& name : allSamplerNames()) {
        if (!samplerAvailable(name)) continue;
        INFO(name);
        Latent x = start;
        auto e = env(pointData(0.7f), 5, 40);
        REQUIRE(runSampler(name, x, sigmas, e));
        for (float v : x) REQUIRE(v == Approx(0.7f).margin(1e-3));
    }
}

TEST_CASE("samplers: stochastic samplers keep the variance of the data", "[unit][sampling][samplers]") {
    // data N(0,1): starting from noise with variance 1 + sigma_T^2, a correct sampler ends with variance 1
    const size_t n = 20000;
    const auto sigmas = karras(30, 10.0, 0.03);
    Latent start = NormalGenerator(RngKind::Philox, 99).normal(n);
    for (float& v : start) v *= static_cast<float>(std::sqrt(1.0 + sigmas.front() * sigmas.front()));
    for (const auto& name : allSamplerNames()) {
        if (!samplerAvailable(name) || name == "lcm") continue;   // lcm is built for few steps and distilled models
        INFO(name);
        Latent x = start;
        SamplerOptions opt;
        auto e = env(gaussianData(), 17, n);
        REQUIRE(runSampler(name, x, sigmas, e, opt));
        double mean = 0;
        for (float v : x) mean += v;
        mean /= static_cast<double>(n);
        CHECK(mean == Approx(0.0).margin(0.05));
        CHECK(rms(x) == Approx(1.0).margin(0.08));
    }
}

TEST_CASE("samplers: eta zero turns ancestral samplers into their deterministic forms", "[unit][sampling][samplers]") {
    const auto sigmas = karras(12);
    const Latent start = startFor(sigmas, 32, 4);
    SamplerOptions off;
    off.eta = 0.0;
    Latent a = start, b = start;
    auto e1 = env(gaussianData()), e2 = env(gaussianData());
    runSampler("euler ancestral", a, sigmas, e1, off);
    runSampler("euler", b, sigmas, e2);
    CHECK(maxError(a, b) < 1e-5);
    a = start;
    b = start;
    runSampler("res multistep ancestral", a, sigmas, e1, off);
    runSampler("res multistep", b, sigmas, e2);
    CHECK(maxError(a, b) < 1e-5);
    // the same noise gives the same picture; another seed another one
    SamplerOptions on;
    Latent c = start, d = start, f = start;
    auto n1 = env(gaussianData(), 1), n2 = env(gaussianData(), 1), n3 = env(gaussianData(), 2);
    runSampler("euler ancestral", c, sigmas, n1, on);
    runSampler("euler ancestral", d, sigmas, n2, on);
    runSampler("euler ancestral", f, sigmas, n3, on);
    CHECK(c == d);
    CHECK(maxError(c, f) > 1e-3);
    // sNoise scales the fresh noise
    Latent quiet = start;
    SamplerOptions silent;
    silent.sNoise = 0.0;
    auto n4 = env(gaussianData(), 1);
    runSampler("euler ancestral", quiet, sigmas, n4, silent);
    Latent plain = start;
    auto n5 = env(gaussianData());
    SamplerOptions zero;
    zero.eta = 0.0;
    runSampler("euler ancestral", plain, sigmas, n5, zero);
    CHECK(maxError(quiet, plain) > 1e-4);   // without the noise it is still not the deterministic path: the step is a smaller one
}

TEST_CASE("samplers: cancelling stops them, and CFG++ uses the unguided estimate", "[unit][sampling][samplers]") {
    const auto sigmas = karras(10);
    Latent x = startFor(sigmas, 8, 1);
    int calls = 0;
    SamplerEnv e;
    e.denoise = [&](const Latent& at, float, int, Latent& out) {
        out = at;
        return ++calls < 3;
    };
    CHECK_FALSE(runSampler("euler", x, sigmas, e));
    CHECK(calls == 3);

    // with no unguided estimate the cfg++ variant is plain euler
    Latent a = startFor(sigmas, 8, 1), b = a;
    auto e1 = env(gaussianData()), e2 = env(gaussianData());
    runSampler("euler cfg++", a, sigmas, e1);
    runSampler("euler", b, sigmas, e2);
    CHECK(maxError(a, b) < 1e-6);
    // when the guided and unguided estimates are the same, cfg++ is euler as well
    Latent u;
    SamplerEnv e3;
    e3.denoise = [&](const Latent& at, float sigma, int, Latent& out) {
        out.resize(at.size());
        for (size_t i = 0; i < at.size(); ++i) out[i] = static_cast<float>(at[i] / (1.0 + static_cast<double>(sigma) * sigma));
        u = out;
        return true;
    };
    e3.uncond = &u;
    Latent c = startFor(sigmas, 8, 1);
    runSampler("euler cfg++", c, sigmas, e3);
    CHECK(maxError(c, b) < 1e-4);
    // with a different unguided estimate the result differs
    SamplerEnv e4;
    Latent shifted;
    e4.denoise = [&](const Latent& at, float sigma, int, Latent& out) {
        out.resize(at.size());
        shifted.resize(at.size());
        for (size_t i = 0; i < at.size(); ++i) {
            out[i] = static_cast<float>(at[i] / (1.0 + static_cast<double>(sigma) * sigma));
            shifted[i] = out[i] + 0.5f;
        }
        return true;
    };
    e4.uncond = &shifted;
    Latent d = startFor(sigmas, 8, 1);
    runSampler("euler cfg++", d, sigmas, e4);
    CHECK(maxError(d, b) > 0.01);
}
