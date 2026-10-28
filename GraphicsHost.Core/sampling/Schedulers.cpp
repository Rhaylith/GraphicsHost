#include "sampling/Schedulers.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace gh::sampling {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> v(static_cast<size_t>(std::max(n, 0)));
    for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = n == 1 ? a : a + (b - a) * i / (n - 1);
    return v;
}

std::vector<float> finish(const std::vector<double>& sigmas) {
    std::vector<float> out;
    for (const double s : sigmas) out.push_back(static_cast<float>(s));
    out.push_back(0.0f);
    return out;
}

}  // namespace

const char* toString(Scheduler s) {
    switch (s) {
        case Scheduler::Normal: return "normal";
        case Scheduler::Karras: return "karras";
        case Scheduler::Exponential: return "exponential";
        case Scheduler::SgmUniform: return "sgm uniform";
        case Scheduler::Simple: return "simple";
        case Scheduler::DdimUniform: return "ddim uniform";
        case Scheduler::Beta: return "beta";
        case Scheduler::LinearQuadratic: return "linear quadratic";
        case Scheduler::KlOptimal: return "kl optimal";
    }
    return "normal";
}

std::optional<Scheduler> parseScheduler(const std::string& s) {
    for (const Scheduler k : {Scheduler::Normal, Scheduler::Karras, Scheduler::Exponential, Scheduler::SgmUniform, Scheduler::Simple, Scheduler::DdimUniform, Scheduler::Beta,
                              Scheduler::LinearQuadratic, Scheduler::KlOptimal})
        if (s == toString(k)) return k;
    return std::nullopt;
}

std::vector<float> karrasSchedule(int steps, double sigmaMax, double sigmaMin, double rho) {
    std::vector<double> s;
    const double a = std::pow(sigmaMax, 1.0 / rho), b = std::pow(sigmaMin, 1.0 / rho);
    for (const double r : linspace(0, 1, steps)) s.push_back(std::pow(a + r * (b - a), rho));
    return finish(s);
}

std::vector<float> exponentialSchedule(int steps, double sigmaMax, double sigmaMin) {
    std::vector<double> s;
    for (const double l : linspace(std::log(sigmaMax), std::log(sigmaMin), steps)) s.push_back(std::exp(l));
    return finish(s);
}

std::vector<float> polyExponentialSchedule(int steps, double sigmaMax, double sigmaMin, double rho) {
    std::vector<double> s;
    for (const double r : linspace(1, 0, steps)) s.push_back(std::exp(std::pow(r, rho) * (std::log(sigmaMax) - std::log(sigmaMin)) + std::log(sigmaMin)));
    return finish(s);
}

std::vector<float> vpSchedule(int steps, double betaD, double betaMin, double epsS) {
    std::vector<double> s;
    for (const double t : linspace(1, epsS, steps)) s.push_back(std::sqrt(std::exp(betaD * t * t / 2 + betaMin * t) - 1));
    return finish(s);
}

std::vector<float> laplaceSchedule(int steps, double sigmaMax, double sigmaMin, double mu, double beta) {
    std::vector<double> s;
    const double eps = 1e-5, lo = std::log(sigmaMin), hi = std::log(sigmaMax);
    for (const double x : linspace(0, 1, steps)) {
        const double d = 0.5 - x;
        const double sign = d > 0 ? 1.0 : d < 0 ? -1.0 : 0.0;
        double l = mu - beta * sign * std::log(1 - 2 * std::fabs(d) + eps);
        l = std::clamp(l, lo, hi);
        s.push_back(std::exp(l));
    }
    return finish(s);
}

std::vector<float> turboSchedule(const ModelSampling& model, int steps) {
    std::vector<double> s;
    for (int i = 0; i < std::clamp(steps, 1, 10); ++i) s.push_back(model.sigma((10 - i) * 100.0 - 1.0));   // timesteps 999, 899, 799 ... of the model
    return finish(s);
}

// ---- the beta distribution -------------------------------------------------------------------------------------------------

namespace {
// continued fraction of the incomplete beta function (modified Lentz)
double betaContinuedFraction(double x, double a, double b) {
    const int maxIter = 300;
    const double tiny = 1e-300, eps = 1e-14;
    double qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (std::fabs(d) < tiny) d = tiny;
    d = 1 / d;
    double h = d;
    for (int m = 1; m <= maxIter; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d;
        if (std::fabs(d) < tiny) d = tiny;
        c = 1 + aa / c;
        if (std::fabs(c) < tiny) c = tiny;
        d = 1 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d;
        if (std::fabs(d) < tiny) d = tiny;
        c = 1 + aa / c;
        if (std::fabs(c) < tiny) c = tiny;
        d = 1 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1) < eps) break;
    }
    return h;
}
}  // namespace

double betaCdf(double x, double a, double b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const double front = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return front * betaContinuedFraction(x, a, b) / a;
    return 1 - front * betaContinuedFraction(1 - x, b, a) / b;
}

double betaPpf(double p, double a, double b) {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    double lo = 0, hi = 1;
    for (int i = 0; i < 80; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (betaCdf(mid, a, b) < p) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

std::vector<float> betaSchedule(const ModelSampling& model, int steps, double alpha, double beta) {
    const int total = static_cast<int>(model.discreteSigmas().size()) - 1;
    std::vector<float> out;
    std::set<long long> seen;
    for (int i = 0; i < steps; ++i) {
        const double t = 1.0 - static_cast<double>(i) / steps;
        const long long ts = std::llround(betaPpf(t, alpha, beta) * total);
        if (!seen.insert(ts).second) continue;   // two steps on one timestep count once
        out.push_back(static_cast<float>(model.sigma(static_cast<double>(ts))));
    }
    out.push_back(0.0f);
    return out;
}

std::vector<float> makeSchedule(Scheduler s, const ModelSampling& m, int steps) {
    steps = std::max(steps, 1);
    const double smax = m.sigmaMax(), smin = m.sigmaMin();
    switch (s) {
        case Scheduler::Normal: {
            std::vector<double> v;
            for (const double t : linspace(m.timestep(smax), m.timestep(smin), steps)) v.push_back(m.sigma(t));
            return finish(v);
        }
        case Scheduler::SgmUniform: {
            std::vector<double> v;
            const auto ts = linspace(m.timestep(smax), m.timestep(smin), steps + 1);
            for (int i = 0; i < steps; ++i) v.push_back(m.sigma(ts[static_cast<size_t>(i)]));
            return finish(v);
        }
        case Scheduler::Karras: return karrasSchedule(steps, smax, smin);
        case Scheduler::Exponential: return exponentialSchedule(steps, smax, smin);
        case Scheduler::Simple: {
            const auto& all = m.discreteSigmas();
            if (all.empty()) return makeSchedule(Scheduler::Normal, m, steps);
            std::vector<double> v;
            const double stride = static_cast<double>(all.size()) / steps;
            for (int i = 0; i < steps; ++i) v.push_back(all[all.size() - 1 - static_cast<size_t>(i * stride)]);
            return finish(v);
        }
        case Scheduler::DdimUniform: {
            const auto& all = m.discreteSigmas();
            if (all.empty()) return makeSchedule(Scheduler::Normal, m, steps);
            const size_t stride = std::max<size_t>(all.size() / static_cast<size_t>(steps), 1);
            std::vector<double> v;
            for (size_t x = 1; x < all.size(); x += stride) v.push_back(all[x]);
            std::reverse(v.begin(), v.end());
            return finish(v);
        }
        case Scheduler::Beta:
            if (m.discreteSigmas().empty()) return makeSchedule(Scheduler::Normal, m, steps);
            return betaSchedule(m, steps);
        case Scheduler::LinearQuadratic: {
            // the noise falls linearly for the first half of the steps, then along a parabola with the same slope that meets zero
            // at the last step; written as levels of a flow schedule (1 at the start, 0 at the end) and mapped to the model's range
            const int n = steps, lin = std::max(n / 2, 1);
            const double threshold = 0.025, slope = threshold / lin;
            const int rest = n - lin;
            const double k = rest > 0 ? (1.0 - threshold - slope * rest) / (static_cast<double>(rest) * rest) : 0.0;
            std::vector<double> v;
            for (int i = 0; i < n; ++i) {
                const double q = i < lin ? i * slope : threshold + slope * (i - lin) + k * (i - lin) * static_cast<double>(i - lin);
                v.push_back(1.0 - q);
            }
            std::vector<float> out;
            for (const double level : v) out.push_back(static_cast<float>(m.params().kind == SamplingKind::Flow ? level : level * smax));
            out.push_back(0.0f);
            return out;
        }
        case Scheduler::KlOptimal: {
            std::vector<double> v;
            const double a = std::atan(smax), b = std::atan(smin);
            for (const double t : linspace(0, 1, steps)) v.push_back(std::tan((1 - t) * a + t * b));
            return finish(v);
        }
    }
    return {};
}

std::vector<float> makeScheduleDenoise(Scheduler s, const ModelSampling& m, int steps, double denoise) {
    if (denoise <= 0.0) return {};
    if (denoise >= 0.9999) return makeSchedule(s, m, steps);
    const int total = static_cast<int>(steps / denoise);
    const std::vector<float> all = makeSchedule(s, m, total);
    const size_t keep = static_cast<size_t>(steps) + 1;
    return std::vector<float>(all.end() - static_cast<std::ptrdiff_t>(std::min(keep, all.size())), all.end());
}

void splitSigmas(const std::vector<float>& sigmas, int step, std::vector<float>& high, std::vector<float>& low) {
    const int n = static_cast<int>(sigmas.size());
    step = std::clamp(step, 0, std::max(n - 1, 0));
    high.assign(sigmas.begin(), sigmas.begin() + std::min(step + 1, n));
    low.assign(sigmas.begin() + std::min(step, n), sigmas.end());
}

void splitSigmasDenoise(const std::vector<float>& sigmas, double denoise, std::vector<float>& high, std::vector<float>& low) {
    const int total = std::max(static_cast<int>(sigmas.size()) - 1, 0);
    const int stepsHigh = static_cast<int>(std::lround(total * (1.0 - std::clamp(denoise, 0.0, 1.0))));
    splitSigmas(sigmas, stepsHigh, high, low);
}

std::vector<float> flipSigmas(const std::vector<float>& sigmas) {
    std::vector<float> out(sigmas.rbegin(), sigmas.rend());
    if (!out.empty() && out.front() == 0.0f) out.front() = 1e-4f;
    return out;
}

std::vector<float> setFirstSigma(std::vector<float> sigmas, float sigma) {
    if (!sigmas.empty()) sigmas.front() = sigma;
    return sigmas;
}

std::vector<float> extendSigmas(const std::vector<float>& sigmas, int count, double startAt, double endAt, const std::string& spacing) {
    std::vector<float> out;
    const auto shape = [&](double t) {
        if (spacing == "cosine") return (1.0 - std::cos(kPi * t)) / 2.0;
        if (spacing == "sine") return std::sin(kPi * t / 2.0);
        return t;
    };
    for (size_t i = 0; i < sigmas.size(); ++i) {
        out.push_back(sigmas[i]);
        if (i + 1 == sigmas.size()) break;
        const double a = sigmas[i], b = sigmas[i + 1];
        const bool inside = (startAt < 0 || a >= startAt) && (endAt < 0 || a <= endAt);
        if (!inside) continue;
        for (int k = 1; k <= count; ++k) out.push_back(static_cast<float>(a + (b - a) * shape(static_cast<double>(k) / (count + 1))));
    }
    return out;
}

}  // namespace gh::sampling
