#include "sampling/ModelSampling.h"

#include <algorithm>
#include <cmath>

namespace gh::sampling {

const char* toString(Prediction p) {
    switch (p) {
        case Prediction::Eps: return "eps";
        case Prediction::V: return "v prediction";
        case Prediction::X0: return "x0";
        case Prediction::Lcm: return "lcm";
        case Prediction::ImgToImg: return "img to img";
        case Prediction::Edm: return "edm";
        case Prediction::PlaygroundV25: return "playground v2.5";
    }
    return "eps";
}

std::optional<Prediction> parsePrediction(const std::string& s) {
    for (const Prediction p : {Prediction::Eps, Prediction::V, Prediction::X0, Prediction::Lcm, Prediction::ImgToImg, Prediction::Edm, Prediction::PlaygroundV25})
        if (s == toString(p)) return p;
    return std::nullopt;
}

std::vector<double> ModelSampling::sdAlphasCumprod() {
    // betas: the square of a linear ramp from sqrt(0.00085) to sqrt(0.012) over 1000 steps
    std::vector<double> ac(1000);
    const double lo = std::sqrt(0.00085), hi = std::sqrt(0.012);
    double product = 1.0;
    for (int i = 0; i < 1000; ++i) {
        const double r = lo + (hi - lo) * i / 999.0;
        product *= 1.0 - r * r;
        ac[static_cast<size_t>(i)] = product;
    }
    return ac;
}

double flowSigma(double t, double shift) { return shift * t / (1.0 + (shift - 1.0) * t); }

double fluxShift(double maxShift, double baseShift, int width, int height) {
    const double m = (maxShift - baseShift) / (4096.0 - 256.0);
    const double b = baseShift - m * 256.0;
    const double tokens = static_cast<double>(width) * height / 256.0;
    return std::exp(tokens * m + b);
}

ModelSampling::ModelSampling(SamplingParams params) : p_(params) {
    switch (p_.kind) {
        case SamplingKind::Discrete: {
            std::vector<double> ac = sdAlphasCumprod();
            if (p_.zeroSnr) {
                // shift and scale sqrt(alphas_cumprod) so that the last step has zero signal and the first is unchanged, then
                // leave a tiny remainder so the last sigma stays finite
                std::vector<double> root(ac.size());
                for (size_t i = 0; i < ac.size(); ++i) root[i] = std::sqrt(ac[i]);
                const double first = root.front(), last = root.back();
                for (double& r : root) r = (r - last) * first / (first - last);
                for (size_t i = 0; i < ac.size(); ++i) ac[i] = root[i] * root[i];
                ac.back() = 4.8973451890853435e-08;
            }
            discrete_.resize(ac.size());
            for (size_t i = 0; i < ac.size(); ++i) discrete_[i] = std::sqrt((1.0 - ac[i]) / ac[i]);
            sigmaMin_ = discrete_.front();
            sigmaMax_ = discrete_.back();
            break;
        }
        case SamplingKind::Edm:
            sigmaMax_ = p_.sigmaMax > 0 ? p_.sigmaMax : (p_.prediction == Prediction::V ? 500.0 : 120.0);
            sigmaMin_ = p_.sigmaMin > 0 ? p_.sigmaMin : (p_.prediction == Prediction::V ? 0.03 : 0.002);
            break;
        case SamplingKind::Flow:
            sigmaMax_ = 1.0;
            sigmaMin_ = flowSigma(1e-3, p_.shift);
            break;
    }
}

double ModelSampling::timestep(double sigma) const {
    switch (p_.kind) {
        case SamplingKind::Discrete: {
            const double logSigma = std::log(std::max(sigma, 1e-10));
            // the position between the two steps whose log sigmas enclose it
            if (logSigma <= std::log(discrete_.front())) return 0.0;
            if (logSigma >= std::log(discrete_.back())) return static_cast<double>(discrete_.size() - 1);
            const auto it = std::lower_bound(discrete_.begin(), discrete_.end(), sigma);
            const size_t hi = static_cast<size_t>(it - discrete_.begin()), lo = hi - 1;
            const double a = std::log(discrete_[lo]), b = std::log(discrete_[hi]);
            return static_cast<double>(lo) + (logSigma - a) / (b - a);
        }
        case SamplingKind::Edm: return 0.25 * std::log(std::max(sigma, 1e-10));
        case SamplingKind::Flow: {
            // invert sigma = shift * t / (1 + (shift - 1) * t)
            const double s = std::min(sigma, 1.0);
            return 1000.0 * s / (p_.shift - (p_.shift - 1.0) * s);
        }
    }
    return 0.0;
}

double ModelSampling::sigma(double t) const {
    switch (p_.kind) {
        case SamplingKind::Discrete: {
            const double clamped = std::clamp(t, 0.0, static_cast<double>(discrete_.size() - 1));
            const size_t lo = static_cast<size_t>(std::floor(clamped)), hi = std::min(lo + 1, discrete_.size() - 1);
            const double f = clamped - static_cast<double>(lo);
            return std::exp(std::log(discrete_[lo]) * (1.0 - f) + std::log(discrete_[hi]) * f);
        }
        case SamplingKind::Edm: return std::exp(4.0 * t);
        case SamplingKind::Flow: return flowSigma(t / 1000.0, p_.shift);
    }
    return 0.0;
}

double ModelSampling::inputScale(double sigma) const {
    switch (p_.kind) {
        case SamplingKind::Flow: return 1.0;
        case SamplingKind::Edm:
            if (p_.prediction == Prediction::Edm || p_.prediction == Prediction::PlaygroundV25) return 1.0 / std::sqrt(sigma * sigma + p_.sigmaData * p_.sigmaData);
            return 1.0 / std::sqrt(sigma * sigma + 1.0);
        case SamplingKind::Discrete:
            if (p_.prediction == Prediction::X0 || p_.prediction == Prediction::ImgToImg) return 1.0;
            return 1.0 / std::sqrt(sigma * sigma + 1.0);
    }
    return 1.0;
}

void ModelSampling::denoise(const float* x, const float* out, double sigma, float* result, size_t n) const {
    if (p_.kind == SamplingKind::Flow) {
        for (size_t i = 0; i < n; ++i) result[i] = static_cast<float>(x[i] - out[i] * sigma);
        return;
    }
    switch (p_.prediction) {
        case Prediction::Eps:
            for (size_t i = 0; i < n; ++i) result[i] = static_cast<float>(x[i] - out[i] * sigma);
            break;
        case Prediction::V: {
            const double a = 1.0 / (sigma * sigma + 1.0), b = sigma / std::sqrt(sigma * sigma + 1.0);
            for (size_t i = 0; i < n; ++i) result[i] = static_cast<float>(x[i] * a - out[i] * b);
            break;
        }
        case Prediction::X0:
        case Prediction::ImgToImg:
            for (size_t i = 0; i < n; ++i) result[i] = out[i];
            break;
        case Prediction::Lcm: {
            const double t = timestep(sigma);   // the discrete timestep
            const double scaled = t / 0.1;
            const double sd2 = p_.sigmaData * p_.sigmaData;
            const double cSkip = sd2 / (scaled * scaled + sd2), cOut = scaled / std::sqrt(scaled * scaled + sd2);
            for (size_t i = 0; i < n; ++i) result[i] = static_cast<float>(cOut * (x[i] - out[i] * sigma) + cSkip * x[i]);
            break;
        }
        case Prediction::Edm:
        case Prediction::PlaygroundV25: {
            const double sd2 = p_.sigmaData * p_.sigmaData, s2 = sigma * sigma;
            const double cSkip = sd2 / (s2 + sd2), cOut = sigma * p_.sigmaData / std::sqrt(s2 + sd2);
            for (size_t i = 0; i < n; ++i) result[i] = static_cast<float>(out[i] * cOut + x[i] * cSkip);
            break;
        }
    }
}

void ModelSampling::scaleNoise(double sigma, const float* noise, const float* latent, float* out, size_t n) const {
    if (p_.kind == SamplingKind::Flow) {
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(sigma * noise[i] + (1.0 - sigma) * (latent ? latent[i] : 0.0f));
        return;
    }
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(noise[i] * sigma + (latent ? latent[i] : 0.0f));
}

void ModelSampling::unscaleNoise(double sigma, const float* noisy, float* out, size_t n) const {
    const double d = p_.kind == SamplingKind::Flow ? (1.0 - sigma) : 1.0;
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(noisy[i] / (d == 0.0 ? 1.0 : d));
}

}  // namespace gh::sampling
