#pragma once

#include "sampling/ModelSampling.h"

#include <optional>
#include <string>
#include <vector>

namespace gh::sampling {

// Sigma schedules: `steps + 1` noise levels from high to low, ending at 0. All of them are written from the papers and the
// definitions of the schedules; the named ones follow the model's own noise range (workflow plan, section 6.4).

enum class Scheduler { Normal, Karras, Exponential, SgmUniform, Simple, DdimUniform, Beta, LinearQuadratic, KlOptimal };
const char* toString(Scheduler s);                              // "normal", "karras", "exponential", "sgm uniform", "simple", "ddim uniform", "beta", "linear quadratic", "kl optimal"
std::optional<Scheduler> parseScheduler(const std::string& s);

// A named schedule for a model. Beta uses the default 0.6 / 0.6 distribution (see betaSchedule for others).
std::vector<float> makeSchedule(Scheduler s, const ModelSampling& model, int steps);

// The same, with "denoise" < 1 dropping the noisiest part: the schedule of steps / denoise is made and its last `steps` levels kept.
std::vector<float> makeScheduleDenoise(Scheduler s, const ModelSampling& model, int steps, double denoise);

// Parametric schedules.
std::vector<float> karrasSchedule(int steps, double sigmaMax, double sigmaMin, double rho = 7.0);               // Karras et al. 2022, eq. 5
std::vector<float> exponentialSchedule(int steps, double sigmaMax, double sigmaMin);                           // even in log sigma
std::vector<float> polyExponentialSchedule(int steps, double sigmaMax, double sigmaMin, double rho = 1.0);     // log sigma follows ramp^rho
std::vector<float> vpSchedule(int steps, double betaD = 19.9, double betaMin = 0.1, double epsS = 1e-3);      // variance preserving
std::vector<float> betaSchedule(const ModelSampling& model, int steps, double alpha = 0.6, double beta = 0.6);  // beta-distributed timesteps (Lee et al. 2024)
std::vector<float> laplaceSchedule(int steps, double sigmaMax, double sigmaMin, double mu = 0.0, double beta = 0.5);
std::vector<float> turboSchedule(const ModelSampling& model, int steps);                                      // 1 to 10 steps from timestep 999 down in hundreds

// Edits of a schedule.
void splitSigmas(const std::vector<float>& sigmas, int step, std::vector<float>& high, std::vector<float>& low);   // high = [0, step], low = [step, end]
void splitSigmasDenoise(const std::vector<float>& sigmas, double denoise, std::vector<float>& high, std::vector<float>& low);
std::vector<float> flipSigmas(const std::vector<float>& sigmas);                                                 // reversed; a leading 0 becomes 1e-4
std::vector<float> setFirstSigma(std::vector<float> sigmas, float sigma);
// Inserts `count` new levels between neighbours that lie in [startAt, endAt] (a negative bound is open); spacing is "linear", "cosine" or "sine".
std::vector<float> extendSigmas(const std::vector<float>& sigmas, int count, double startAt, double endAt, const std::string& spacing);

// The regularised incomplete beta function and its inverse (exposed for tests).
double betaCdf(double x, double a, double b);
double betaPpf(double p, double a, double b);

}  // namespace gh::sampling
