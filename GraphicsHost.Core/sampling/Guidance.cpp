#include "sampling/Guidance.h"

#include <algorithm>
#include <cmath>

namespace gh::sampling {

namespace {

double dot(const float* a, const float* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}

double stdev(const float* a, size_t n) {
    if (n < 2) return 0.0;
    double mean = 0;
    for (size_t i = 0; i < n; ++i) mean += a[i];
    mean /= static_cast<double>(n);
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += (a[i] - mean) * (a[i] - mean);
    return std::sqrt(s / static_cast<double>(n - 1));
}

}  // namespace

void cfgCombine(const float* cond, const float* uncond, double scale, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(uncond[i] + scale * (static_cast<double>(cond[i]) - uncond[i]));
}

void guide(const GuidanceTweaks& t, GuidanceState& state, const float* x, const float* cond, const float* uncond, double scale, size_t perSample, size_t batch, float* out) {
    const int step = state.step++;
    const size_t total = perSample * batch;
    if (step < t.zeroInitSteps) {   // a zero noise prediction: the estimate is the input itself
        for (size_t i = 0; i < total; ++i) out[i] = x[i];
        return;
    }
    std::vector<float> result(total);
    if (t.apg) {
        // update = cond - uncond, averaged over steps, capped, split into parts parallel and orthogonal to cond
        std::vector<float> diff(total);
        for (size_t i = 0; i < total; ++i) diff[i] = cond[i] - uncond[i];
        if (t.apgMomentum != 0.0) {
            if (state.apgAverage.size() != total) state.apgAverage.assign(total, 0.0f);
            for (size_t i = 0; i < total; ++i) {
                state.apgAverage[i] = static_cast<float>(diff[i] + t.apgMomentum * state.apgAverage[i]);
                diff[i] = state.apgAverage[i];
            }
        }
        for (size_t b = 0; b < batch; ++b) {
            float* d = diff.data() + b * perSample;
            const float* c = cond + b * perSample;
            if (t.apgNormThreshold > 0) {
                const double norm = std::sqrt(dot(d, d, perSample));
                const double factor = norm > 0 ? std::min(1.0, t.apgNormThreshold / norm) : 1.0;
                for (size_t i = 0; i < perSample; ++i) d[i] = static_cast<float>(d[i] * factor);
            }
            const double cc = dot(c, c, perSample);
            const double parallelCoeff = cc > 0 ? dot(d, c, perSample) / cc : 0.0;
            for (size_t i = 0; i < perSample; ++i) {
                const double parallel = parallelCoeff * c[i], orthogonal = d[i] - parallel;
                result[b * perSample + i] = static_cast<float>(c[i] + (scale - 1.0) * (orthogonal + t.apgEta * parallel));
            }
        }
    } else if (t.zeroStar) {
        for (size_t b = 0; b < batch; ++b) {
            const float* c = cond + b * perSample;
            const float* u = uncond + b * perSample;
            // work on the noise predictions: the optimised scale is the projection of the positive one onto the negative one
            std::vector<float> ec(perSample), eu(perSample);
            for (size_t i = 0; i < perSample; ++i) {
                ec[i] = x[b * perSample + i] - c[i];
                eu[i] = x[b * perSample + i] - u[i];
            }
            const double uu = dot(eu.data(), eu.data(), perSample);
            const double st = uu > 0 ? dot(ec.data(), eu.data(), perSample) / uu : 1.0;
            for (size_t i = 0; i < perSample; ++i) {
                const double e = eu[i] * st + scale * (ec[i] - eu[i] * st);
                result[b * perSample + i] = static_cast<float>(x[b * perSample + i] - e);
            }
        }
    } else {
        cfgCombine(cond, uncond, scale, result.data(), total);
    }
    if (t.rescale > 0.0) {
        for (size_t b = 0; b < batch; ++b) {
            std::vector<float> ec(perSample), eg(perSample);
            for (size_t i = 0; i < perSample; ++i) {
                ec[i] = x[b * perSample + i] - cond[b * perSample + i];
                eg[i] = x[b * perSample + i] - result[b * perSample + i];
            }
            const double sc = stdev(ec.data(), perSample), sg = stdev(eg.data(), perSample);
            const double factor = sg > 0 ? sc / sg : 1.0;
            for (size_t i = 0; i < perSample; ++i) {
                const double mixed = t.rescale * (eg[i] * factor) + (1.0 - t.rescale) * eg[i];
                result[b * perSample + i] = static_cast<float>(x[b * perSample + i] - mixed);
            }
        }
    }
    std::copy(result.begin(), result.end(), out);
}

void dualCfg(const float* a, const float* b, const float* neg, double cfgA, double cfgB, bool nested, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (nested) out[i] = static_cast<float>(neg[i] + cfgB * ((b[i] + cfgA * (static_cast<double>(a[i]) - b[i])) - neg[i]));
        else out[i] = static_cast<float>(neg[i] + cfgB * (static_cast<double>(b[i]) - neg[i]) + cfgA * (static_cast<double>(a[i]) - b[i]));
    }
}

void perpNegCfg(const float* pos, const float* neg, const float* empty, double cfg, double negScale, size_t perSample, size_t batch, float* out) {
    for (size_t b = 0; b < batch; ++b) {
        const size_t o = b * perSample;
        std::vector<float> p(perSample), n(perSample);
        for (size_t i = 0; i < perSample; ++i) {
            p[i] = pos[o + i] - empty[o + i];
            n[i] = neg[o + i] - empty[o + i];
        }
        const double pp = dot(p.data(), p.data(), perSample);
        const double k = pp > 0 ? dot(n.data(), p.data(), perSample) / pp : 0.0;
        for (size_t i = 0; i < perSample; ++i) {
            const double perpendicular = n[i] - k * p[i];
            out[o + i] = static_cast<float>(empty[o + i] + cfg * (p[i] - negScale * perpendicular));
        }
    }
}

}  // namespace gh::sampling
