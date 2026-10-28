#pragma once

#include "sampling/Rng.h"

#include <cstdint>
#include <vector>

namespace gh::sampling {

// A Brownian motion that can be asked for its value at any time, in any order, and always gives the same answer: the interval
// is halved again and again, each midpoint drawn from the Brownian bridge with noise taken from a generator keyed by the node
// of the tree. SDE samplers use it so that their noise does not depend on the step sizes they happen to take.
class BrownianTree {
public:
    BrownianTree(double t0, double t1, size_t size, uint64_t seed, RngKind kind = RngKind::Philox);

    // W(b) - W(a) for t0 <= a, b <= t1 (negative when b < a).
    std::vector<float> increment(double a, double b) const;
    // The increment divided by sqrt(|b - a|): standard normal noise whatever the step. a == b gives zeros.
    std::vector<float> noise(double a, double b) const;

private:
    std::vector<float> value(double t) const;
    std::vector<float> draw(uint64_t node) const;

    double t0_, t1_;
    size_t size_;
    uint64_t seed_;
    RngKind kind_;
};

}  // namespace gh::sampling
