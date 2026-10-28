#include "sampling/BrownianTree.h"

#include <algorithm>
#include <cmath>

namespace gh::sampling {

BrownianTree::BrownianTree(double t0, double t1, size_t size, uint64_t seed, RngKind kind)
    : t0_(std::min(t0, t1)), t1_(std::max(t0, t1)), size_(size), seed_(seed), kind_(kind) {}

std::vector<float> BrownianTree::draw(uint64_t node) const {
    // splitmix64 of (seed, node) keys a short stream per node
    uint64_t z = seed_ + node * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return NormalGenerator(kind_, z).normal(size_);
}

std::vector<float> BrownianTree::value(double t) const {
    t = std::clamp(t, t0_, t1_);
    const double range = t1_ - t0_;
    std::vector<float> wlo(size_, 0.0f);
    std::vector<float> whi = draw(1);
    for (float& v : whi) v *= static_cast<float>(std::sqrt(range));
    double lo = t0_, hi = t1_;
    uint64_t node = 1;
    for (int depth = 0; depth < 56 && hi - lo > range * 1e-12; ++depth) {
        const double mid = 0.5 * (lo + hi);
        const std::vector<float> z = draw(node * 2 + 0x100000000ull);   // the bridge noise of this split
        std::vector<float> wmid(size_);
        const float sd = static_cast<float>(std::sqrt((hi - lo) / 4.0));
        for (size_t i = 0; i < size_; ++i) wmid[i] = 0.5f * (wlo[i] + whi[i]) + sd * z[i];
        if (t < mid) {
            hi = mid;
            whi = std::move(wmid);
            node = node * 2;
        } else {
            lo = mid;
            wlo = std::move(wmid);
            node = node * 2 + 1;
        }
    }
    std::vector<float> out(size_);
    const double f = hi > lo ? (t - lo) / (hi - lo) : 0.0;
    for (size_t i = 0; i < size_; ++i) out[i] = static_cast<float>(wlo[i] + (whi[i] - wlo[i]) * f);
    return out;
}

std::vector<float> BrownianTree::increment(double a, double b) const {
    const std::vector<float> wa = value(a), wb = value(b);
    std::vector<float> out(size_);
    for (size_t i = 0; i < size_; ++i) out[i] = wb[i] - wa[i];
    return out;
}

std::vector<float> BrownianTree::noise(double a, double b) const {
    std::vector<float> d = increment(a, b);
    const double span = std::fabs(b - a);
    if (span == 0.0) return std::vector<float>(size_, 0.0f);
    const float inv = static_cast<float>(1.0 / std::sqrt(span));
    for (float& v : d) v *= inv;
    return d;
}

}  // namespace gh::sampling
