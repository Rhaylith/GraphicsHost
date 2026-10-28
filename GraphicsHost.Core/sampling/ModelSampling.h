#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace gh::sampling {

// How a model relates its input and output to the noise level (workflow plan, section 6.4: model sampling). Given the model's
// raw output at noise level sigma, `denoise` returns the estimate of the clean latent; the sampler works with that estimate only.
enum class Prediction { Eps, V, X0, Lcm, ImgToImg, Edm, PlaygroundV25 };
const char* toString(Prediction p);                              // "eps", "v prediction", "x0", "lcm", "img to img", "edm", "playground v2.5"
std::optional<Prediction> parsePrediction(const std::string& s);

enum class SamplingKind {
    Discrete,   // 1000 timesteps with a beta schedule (SD 1.x, SDXL)
    Edm,        // continuous noise levels between sigmaMin and sigmaMax, eps / v / edm / playground predictions
    Flow,       // rectified flow: noise level = shifted time in (0, 1] (SD3, FLUX, Wan ...)
};

struct SamplingParams {
    SamplingKind kind = SamplingKind::Discrete;
    Prediction prediction = Prediction::Eps;
    bool zeroSnr = false;        // Discrete: rescale the schedule so the last step has no signal left
    double sigmaMax = 0;         // Edm: noise range (0 = the kind's default)
    double sigmaMin = 0;
    double sigmaData = 0.5;      // Edm, Lcm
    double shift = 3.0;          // Flow
    bool operator==(const SamplingParams&) const = default;
};

class ModelSampling {
public:
    explicit ModelSampling(SamplingParams params = {});
    const SamplingParams& params() const { return p_; }

    double sigmaMax() const { return sigmaMax_; }
    double sigmaMin() const { return sigmaMin_; }
    // The model's own notion of time for a noise level and back. Discrete: 0..999 (fractional values interpolate in log sigma);
    // Edm: 0.25 * log(sigma); Flow: 1000 * t.
    double timestep(double sigma) const;
    double sigma(double timestep) const;
    // The noise levels of the 1000 discrete steps, ascending (Discrete only; empty otherwise).
    const std::vector<double>& discreteSigmas() const { return discrete_; }

    // The factor the model's input is scaled by before the call (1/sqrt(sigma^2 + 1) for the beta-schedule models).
    double inputScale(double sigma) const;
    // The estimate of the clean latent from the model output: out[i] from x[i] and modelOut[i]. `out` may alias `x`.
    void denoise(const float* x, const float* modelOut, double sigma, float* out, size_t count) const;
    // The latent at the start of a run: noise * sigma added to the latent (flow models blend instead).
    void scaleNoise(double sigma, const float* noise, const float* latent, float* out, size_t count) const;
    // The inverse of scaleNoise for a latent that is already noisy at `sigma` (used when a run is continued).
    void unscaleNoise(double sigma, const float* noisy, float* out, size_t count) const;

    // The default noise range and the schedule of 1000 betas of SD models.
    static std::vector<double> sdAlphasCumprod();

private:
    SamplingParams p_;
    double sigmaMax_ = 0, sigmaMin_ = 0;
    std::vector<double> discrete_;
};

// Flow shift of the FLUX family for an image size: exp(mu) with mu interpolated between baseShift at 256 tokens and maxShift
// at 4096 tokens (a token is a 16x16 block of pixels).
double fluxShift(double maxShift, double baseShift, int width, int height);

// The scale-and-shift of an arbitrary flow time t in (0, 1]: shift * t / (1 + (shift - 1) * t).
double flowSigma(double t, double shift);

}  // namespace gh::sampling
