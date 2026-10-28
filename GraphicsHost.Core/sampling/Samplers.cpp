#include "sampling/Samplers.h"

#include <algorithm>
#include <cmath>

namespace gh::sampling {

namespace {

using V = Latent;

struct Ctx {
    V& x;
    const std::vector<float>& sigmas;
    SamplerEnv& env;
    const SamplerOptions& opt;
    size_t n;
    int steps() const { return static_cast<int>(sigmas.size()) - 1; }
    double sigma(int i) const { return sigmas[static_cast<size_t>(i)]; }

    bool model(const V& at, double sigma, int i, V& out) const { return env.denoise(at, static_cast<float>(sigma), i + 1, out); }
    void done(int i) const {
        if (env.onStep) env.onStep(i + 1, steps());
    }
    // standard normal noise for step i (zeros when the sampler has no source)
    V noise(int i, double from, double to) const {
        V v(n, 0.0f);
        if (env.noise) env.noise(v.data(), n, i + 1, static_cast<float>(from), static_cast<float>(to));
        return v;
    }
    V brownianNoise(int i, double from, double to) const {
        V v(n, 0.0f);
        if (env.brownian) env.brownian(v.data(), n, static_cast<float>(from), static_cast<float>(to));
        else if (env.noise) env.noise(v.data(), n, i + 1, static_cast<float>(from), static_cast<float>(to));
        return v;
    }
};

// the sigma to step down to and the noise to add back, for an ancestral step
void ancestralStep(double sigma, double sigmaNext, double eta, double& down, double& up) {
    if (eta <= 0.0 || sigma <= 0.0) {
        down = sigmaNext;
        up = 0.0;
        return;
    }
    up = std::min(sigmaNext, eta * std::sqrt(sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) / (sigma * sigma)));
    down = std::sqrt(std::max(sigmaNext * sigmaNext - up * up, 0.0));
}

double expm1d(double v) { return std::expm1(v); }

// ---- first order ---------------------------------------------------------------------------------------------------------

bool euler(Ctx& c, bool cfgpp) {
    V den;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        if (cfgpp && c.env.uncond && c.env.uncond->size() == c.n) {
            // CFG++: the direction comes from the unguided estimate, the guided estimate is where the step starts from
            const V& u = *c.env.uncond;
            for (size_t k = 0; k < c.n; ++k) c.x[k] = static_cast<float>(den[k] + (c.x[k] - u[k]) / s * sn);
        } else {
            const double dt = sn - s;
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>((c.x[k] - den[k]) / s * dt);
        }
        c.done(i);
    }
    return true;
}

bool eulerAncestral(Ctx& c, bool cfgpp) {
    V den;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        double down, up;
        ancestralStep(s, sn, c.opt.eta, down, up);
        if (cfgpp && c.env.uncond && c.env.uncond->size() == c.n) {
            const V& u = *c.env.uncond;
            for (size_t k = 0; k < c.n; ++k) c.x[k] = static_cast<float>(den[k] + (c.x[k] - u[k]) / s * down);
        } else {
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>((c.x[k] - den[k]) / s * (down - s));
        }
        if (sn > 0 && up > 0) {
            const V z = c.noise(i, s, sn);
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(z[k] * c.opt.sNoise * up);
        }
        c.done(i);
    }
    return true;
}

bool heun(Ctx& c) {
    V den, x2, den2;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1), dt = sn - s;
        if (sn == 0.0) {
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>((c.x[k] - den[k]) / s * dt);
        } else {
            x2.resize(c.n);
            for (size_t k = 0; k < c.n; ++k) x2[k] = c.x[k] + static_cast<float>((c.x[k] - den[k]) / s * dt);
            if (!c.model(x2, sn, i, den2)) return false;
            for (size_t k = 0; k < c.n; ++k) {
                const double d1 = (c.x[k] - den[k]) / s, d2 = (x2[k] - den2[k]) / sn;
                c.x[k] += static_cast<float>(0.5 * (d1 + d2) * dt);
            }
        }
        c.done(i);
    }
    return true;
}

bool dpm2(Ctx& c, bool ancestral) {
    V den, x2, den2;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        double down = sn, up = 0;
        if (ancestral) ancestralStep(s, sn, c.opt.eta, down, up);
        if (down == 0.0) {
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>((c.x[k] - den[k]) / s * (down - s));
        } else {
            const double mid = std::exp(0.5 * (std::log(s) + std::log(down)));   // the geometric mean
            const double dt1 = mid - s, dt2 = down - s;
            x2.resize(c.n);
            for (size_t k = 0; k < c.n; ++k) x2[k] = c.x[k] + static_cast<float>((c.x[k] - den[k]) / s * dt1);
            if (!c.model(x2, mid, i, den2)) return false;
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>((x2[k] - den2[k]) / mid * dt2);
        }
        if (ancestral && sn > 0 && up > 0) {
            const V z = c.noise(i, s, sn);
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(z[k] * c.opt.sNoise * up);
        }
        c.done(i);
    }
    return true;
}

// ---- multistep in sigma ----------------------------------------------------------------------------------------------------

// Coefficient j of the order-`order` Adams-Bashforth formula for the step from sigmas[i] to sigmas[i+1] when the previous
// derivatives sit at sigmas[i], sigmas[i-1] ...: the integral of the j-th Lagrange polynomial over the step.
double lmsCoefficient(int order, const std::vector<float>& t, int i, int j) {
    const auto basis = [&](double tau) {
        double prod = 1.0;
        for (int k = 0; k < order; ++k) {
            if (k == j) continue;
            prod *= (tau - t[static_cast<size_t>(i - k)]) / (static_cast<double>(t[static_cast<size_t>(i - j)]) - t[static_cast<size_t>(i - k)]);
        }
        return prod;
    };
    // composite Simpson: the integrand is a polynomial of degree < order, so a few hundred panels are exact to rounding
    const double a = t[static_cast<size_t>(i)], b = t[static_cast<size_t>(i + 1)];
    const int panels = 64;
    const double h = (b - a) / (2 * panels);
    double sum = basis(a) + basis(b);
    for (int p = 1; p < 2 * panels; ++p) sum += basis(a + h * p) * (p % 2 ? 4.0 : 2.0);
    return sum * h / 3.0;
}

bool lms(Ctx& c, int maxOrder, bool fixedCoefficients) {
    std::vector<V> ds;
    V den;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        V d(c.n);
        for (size_t k = 0; k < c.n; ++k) d[k] = static_cast<float>((c.x[k] - den[k]) / c.sigma(i));
        ds.push_back(std::move(d));
        if (static_cast<int>(ds.size()) > maxOrder) ds.erase(ds.begin());
        const int order = std::min(i + 1, maxOrder);
        const double dt = c.sigma(i + 1) - c.sigma(i);
        std::vector<double> coeff(static_cast<size_t>(order));
        if (fixedCoefficients) {
            // Adams-Bashforth with the coefficients of equal steps; the formula does not look at the step sizes
            static const double ab[4][4] = {{1, 0, 0, 0}, {1.5, -0.5, 0, 0}, {23.0 / 12, -16.0 / 12, 5.0 / 12, 0}, {55.0 / 24, -59.0 / 24, 37.0 / 24, -9.0 / 24}};
            for (int j = 0; j < order; ++j) coeff[static_cast<size_t>(j)] = ab[std::min(order, 4) - 1][j] * dt;
        } else {
            for (int j = 0; j < order; ++j) coeff[static_cast<size_t>(j)] = lmsCoefficient(order, c.sigmas, i, j);
        }
        for (int j = 0; j < order; ++j) {
            const V& dj = ds[ds.size() - 1 - static_cast<size_t>(j)];
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(coeff[static_cast<size_t>(j)] * dj[k]);
        }
        c.done(i);
    }
    return true;
}

// ---- DPM-Solver++ ----------------------------------------------------------------------------------------------------------

bool dpmpp2sAncestral(Ctx& c) {
    V den, x2, den2;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        double down, up;
        ancestralStep(s, sn, c.opt.eta, down, up);
        if (down == 0.0) {
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>((c.x[k] - den[k]) / s * (down - s));
        } else {
            const double t = -std::log(s), tNext = -std::log(down), h = tNext - t, mid = t + 0.5 * h;
            const double sMid = std::exp(-mid);
            x2.resize(c.n);
            for (size_t k = 0; k < c.n; ++k) x2[k] = static_cast<float>(sMid / s * c.x[k] - expm1d(t - mid) * den[k]);
            if (!c.model(x2, sMid, i, den2)) return false;
            for (size_t k = 0; k < c.n; ++k) c.x[k] = static_cast<float>(down / s * c.x[k] - expm1d(t - tNext) * den2[k]);
        }
        if (sn > 0 && up > 0) {
            const V z = c.noise(i, s, sn);
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(z[k] * c.opt.sNoise * up);
        }
        c.done(i);
    }
    return true;
}

bool dpmpp2m(Ctx& c) {
    V den, old;
    double tLast = 0;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        if (sn == 0.0) {
            c.x = den;
        } else {
            const double t = -std::log(s), tNext = -std::log(sn), h = tNext - t;
            const double ratio = sn / s, gain = -expm1d(-h);
            if (old.empty()) {
                for (size_t k = 0; k < c.n; ++k) c.x[k] = static_cast<float>(ratio * c.x[k] + gain * den[k]);
            } else {
                const double r = (t - tLast) / h;
                for (size_t k = 0; k < c.n; ++k) {
                    const double dd = (1.0 + 1.0 / (2.0 * r)) * den[k] - (1.0 / (2.0 * r)) * old[k];
                    c.x[k] = static_cast<float>(ratio * c.x[k] + gain * dd);
                }
            }
            tLast = t;
        }
        old = den;
        c.done(i);
    }
    return true;
}

bool dpmpp2mSde(Ctx& c) {
    V den, old;
    double hLast = 0;
    const double eta = c.opt.eta;
    const bool heunSolver = c.opt.solver == "heun";
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        double h = 0;
        if (sn == 0.0) {
            c.x = den;
        } else {
            const double t = -std::log(s), tn = -std::log(sn);
            h = tn - t;
            const double etaH = eta * h, ratio = sn / s * std::exp(-etaH), gain = -expm1d(-h - etaH);
            for (size_t k = 0; k < c.n; ++k) c.x[k] = static_cast<float>(ratio * c.x[k] + gain * den[k]);
            if (!old.empty()) {
                const double r = hLast / h;
                const double corr = heunSolver ? (gain / (-h - etaH) + 1.0) / r : 0.5 * gain / r;
                for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(corr * (den[k] - old[k]));
            }
            if (eta > 0 && c.opt.sNoise > 0) {
                const V z = c.brownianNoise(i, s, sn);
                const double scale = sn * std::sqrt(-expm1d(-2.0 * etaH)) * c.opt.sNoise;
                for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(z[k] * scale);
            }
        }
        old = den;
        hLast = h;
        c.done(i);
    }
    return true;
}

bool dpmppSde(Ctx& c) {
    V den, x2, den2;
    const double r = c.opt.r, fac = 1.0 / (2.0 * r), eta = c.opt.eta;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        if (sn == 0.0) {
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>((c.x[k] - den[k]) / s * (sn - s));
        } else {
            const double t = -std::log(s), tNext = -std::log(sn), h = tNext - t, mid = t + h * r;
            const double sMid = std::exp(-mid);
            double sd, su;
            ancestralStep(s, sMid, eta, sd, su);
            const double midDown = -std::log(sd);
            x2.resize(c.n);
            for (size_t k = 0; k < c.n; ++k) x2[k] = static_cast<float>(sd / s * c.x[k] - expm1d(t - midDown) * den[k]);
            {
                const V z = c.brownianNoise(i, s, sMid);
                for (size_t k = 0; k < c.n; ++k) x2[k] += static_cast<float>(z[k] * c.opt.sNoise * su);
            }
            if (!c.model(x2, sMid, i, den2)) return false;
            ancestralStep(s, sn, eta, sd, su);
            const double tDown = -std::log(sd);
            for (size_t k = 0; k < c.n; ++k) {
                const double dd = (1.0 - fac) * den[k] + fac * den2[k];
                c.x[k] = static_cast<float>(sd / s * c.x[k] - expm1d(t - tDown) * dd);
            }
            const V z = c.brownianNoise(i, s, sn);
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(z[k] * c.opt.sNoise * su);
        }
        c.done(i);
    }
    return true;
}

bool dpmpp3mSde(Ctx& c) {
    V den, d1v, d2v;
    double h1 = 0, h2 = 0;
    bool have1 = false, have2 = false;
    const double eta = c.opt.eta;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        double h = 0;
        if (sn == 0.0) {
            c.x = den;
        } else {
            const double t = -std::log(s), tn = -std::log(sn);
            h = tn - t;
            const double hEta = h * (eta + 1.0);
            const double ratio = sn / s * std::exp(-h * eta), gain = -expm1d(-hEta);
            for (size_t k = 0; k < c.n; ++k) c.x[k] = static_cast<float>(ratio * c.x[k] + gain * den[k]);
            const double p2 = expm1d(-hEta) / hEta + 1.0;   // phi_2 as the formulas of the paper define it: (e^{-h} - 1)/h + 1
            if (have2) {
                const double r0 = h1 / h, r1 = h2 / h;
                const double p3 = p2 / hEta - 0.5;
                for (size_t k = 0; k < c.n; ++k) {
                    const double d10 = (den[k] - d1v[k]) / r0, d11 = (d1v[k] - d2v[k]) / r1;
                    const double d1 = d10 + (d10 - d11) * r0 / (r0 + r1);
                    const double d2 = (d10 - d11) / (r0 + r1);
                    c.x[k] += static_cast<float>(p2 * d1 - p3 * d2);
                }
            } else if (have1) {
                const double r = h1 / h;
                for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(p2 * (den[k] - d1v[k]) / r);
            }
            if (eta > 0 && c.opt.sNoise > 0) {
                const V z = c.brownianNoise(i, s, sn);
                const double scale = sn * std::sqrt(-expm1d(-2.0 * h * eta)) * c.opt.sNoise;
                for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(z[k] * scale);
            }
        }
        d2v = d1v;
        d1v = den;
        have2 = have1;
        have1 = true;
        h2 = h1;
        h1 = h;
        c.done(i);
    }
    return true;
}

// ---- RES, LCM ----------------------------------------------------------------------------------------------------------------

double phi1(double t) { return t == 0.0 ? 1.0 : expm1d(t) / t; }
double phi2(double t) { return t == 0.0 ? 0.5 : (phi1(t) - 1.0) / t; }

bool resMultistep(Ctx& c, bool ancestral) {
    V den, old;
    double oldSigmaDown = 0;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double s = c.sigma(i), sn = c.sigma(i + 1);
        double down, up;
        ancestralStep(s, sn, ancestral ? c.opt.eta : 0.0, down, up);
        if (down == 0.0 || old.empty()) {
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>((c.x[k] - den[k]) / s * (down - s));
        } else {
            const double t = -std::log(s), tOld = -std::log(oldSigmaDown), tNext = -std::log(down), tPrev = -std::log(c.sigma(i - 1));
            const double h = tNext - t, c2 = (tPrev - tOld) / h;
            const double p1 = phi1(-h), p2 = phi2(-h);
            double b1 = p1 - p2 / c2, b2 = p2 / c2;
            if (!std::isfinite(b1)) b1 = 0.0;
            if (!std::isfinite(b2)) b2 = 0.0;
            const double decay = std::exp(-h);
            for (size_t k = 0; k < c.n; ++k) c.x[k] = static_cast<float>(decay * c.x[k] + h * (b1 * den[k] + b2 * old[k]));
        }
        if (ancestral && sn > 0 && up > 0) {
            const V z = c.noise(i, s, sn);
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(z[k] * c.opt.sNoise * up);
        }
        old = den;
        oldSigmaDown = down;
        c.done(i);
    }
    return true;
}

bool lcm(Ctx& c) {
    V den;
    for (int i = 0; i < c.steps(); ++i) {
        if (!c.model(c.x, c.sigma(i), i, den)) return false;
        const double sn = c.sigma(i + 1);
        c.x = den;
        if (sn > 0) {
            const V z = c.noise(i, c.sigma(i), sn);
            for (size_t k = 0; k < c.n; ++k) c.x[k] += static_cast<float>(z[k] * sn);
        }
        c.done(i);
    }
    return true;
}

struct Entry {
    const char* name;
    bool stochastic;
    bool (*run)(Ctx&);
};

const Entry kEntries[] = {
    {"euler", false, [](Ctx& c) { return euler(c, false); }},
    {"euler cfg++", false, [](Ctx& c) { return euler(c, true); }},
    {"euler ancestral", true, [](Ctx& c) { return eulerAncestral(c, false); }},
    {"euler ancestral cfg++", true, [](Ctx& c) { return eulerAncestral(c, true); }},
    {"heun", false, heun},
    {"dpm2", false, [](Ctx& c) { return dpm2(c, false); }},
    {"dpm2 ancestral", true, [](Ctx& c) { return dpm2(c, true); }},
    {"lms", false, [](Ctx& c) { return lms(c, std::clamp(c.opt.order, 1, 100), false); }},
    {"dpm++ 2S ancestral", true, dpmpp2sAncestral},
    {"dpm++ SDE", true, dpmppSde},
    {"dpm++ 2M", false, dpmpp2m},
    {"dpm++ 2M SDE", true, dpmpp2mSde},
    {"dpm++ 3M SDE", true, dpmpp3mSde},
    {"ddpm", true, [](Ctx& c) { SamplerOptions o = c.opt; o.eta = 1.0; Ctx d{c.x, c.sigmas, c.env, o, c.n}; return eulerAncestral(d, false); }},
    {"ddim", false, [](Ctx& c) { return euler(c, false); }},
    {"lcm", true, lcm},
    {"ipndm", false, [](Ctx& c) { return lms(c, 4, true); }},
    {"ipndm v", false, [](Ctx& c) { return lms(c, 4, false); }},
    {"res multistep", false, [](Ctx& c) { return resMultistep(c, false); }},
    {"res multistep ancestral", true, [](Ctx& c) { return resMultistep(c, true); }},
};

const Entry* lookup(const std::string& name) {
    for (const Entry& e : kEntries)
        if (name == e.name) return &e;
    return nullptr;
}

}  // namespace

const std::vector<std::string>& allSamplerNames() {
    static const std::vector<std::string> names = {
        "euler", "euler cfg++", "euler ancestral", "euler ancestral cfg++", "heun", "heun++", "dpm2", "dpm2 ancestral", "lms", "dpm fast", "dpm adaptive",
        "dpm++ 2S ancestral", "dpm++ 2S ancestral cfg++", "dpm++ SDE", "dpm++ 2M", "dpm++ 2M cfg++", "dpm++ 2M SDE", "dpm++ 3M SDE", "ddpm", "ddim", "lcm", "ipndm",
        "ipndm v", "deis", "uni pc", "uni pc bh2", "res multistep", "res multistep ancestral", "res multistep cfg++", "gradient estimation", "er sde", "seeds 2", "seeds 3", "sa solver"};
    return names;
}

bool samplerAvailable(const std::string& name) { return lookup(name) != nullptr; }

bool samplerIsStochastic(const std::string& name) {
    const Entry* e = lookup(name);
    return e && e->stochastic;
}

bool runSampler(const std::string& name, Latent& x, const std::vector<float>& sigmas, SamplerEnv& env, const SamplerOptions& options) {
    const Entry* entry = lookup(name);
    if (!entry || sigmas.size() < 2 || !env.denoise) return false;
    Ctx c{x, sigmas, env, options, x.size()};
    return entry->run(c);
}

}  // namespace gh::sampling
