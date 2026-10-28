#include "graph/Executor.h"
#include "graph/Handles.h"
#include "sampling/BrownianTree.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::sampling;
using Catch::Approx;
using nlohmann::json;

namespace {
struct Stats {
    double mean = 0, sd = 0;
};
Stats statsOf(const std::vector<float>& v) {
    double m = 0;
    for (float x : v) m += x;
    m /= static_cast<double>(v.size());
    double s = 0;
    for (float x : v) s += (x - m) * (x - m);
    return {m, std::sqrt(s / static_cast<double>(v.size()))};
}
}  // namespace

TEST_CASE("philox: the Random123 known answers", "[unit][sampling][rng]") {
    uint32_t out[4];
    const uint32_t zero[4] = {0, 0, 0, 0}, key0[2] = {0, 0};
    Philox4x32::block(zero, key0, out);
    CHECK(out[0] == 0x6627e8d5u);
    CHECK(out[1] == 0xe169c58du);
    CHECK(out[2] == 0xbc57ac4cu);
    CHECK(out[3] == 0x9b00dbd8u);
    const uint32_t ones[4] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, keyOnes[2] = {0xffffffffu, 0xffffffffu};
    Philox4x32::block(ones, keyOnes, out);
    CHECK(out[0] == 0x408f276du);
    CHECK(out[1] == 0x41c83b0eu);
    CHECK(out[2] == 0xa20bc7c6u);
    CHECK(out[3] == 0x6d5451fdu);
}

TEST_CASE("rng: names", "[unit][sampling][rng]") {
    CHECK(parseRng("philox") == RngKind::Philox);
    CHECK(parseRng("cpu-compatible") == RngKind::CpuCompatible);
    CHECK_FALSE(parseRng("mersenne").has_value());
    CHECK(std::string(toString(RngKind::Philox)) == "philox");
}

TEST_CASE("rng: normal numbers are normal, repeatable and seed dependent", "[unit][sampling][rng]") {
    for (const RngKind kind : {RngKind::CpuCompatible, RngKind::Philox}) {
        INFO(toString(kind));
        for (const size_t n : {size_t{5}, size_t{16}, size_t{37}, size_t{100000}}) {
            const auto a = NormalGenerator(kind, 42).normal(n);
            CHECK(a == NormalGenerator(kind, 42).normal(n));
            CHECK(a != NormalGenerator(kind, 43).normal(n));
            for (float x : a) REQUIRE(std::isfinite(x));
        }
        const Stats s = statsOf(NormalGenerator(kind, 7).normal(200000));
        CHECK(s.mean == Approx(0.0).margin(0.01));
        CHECK(s.sd == Approx(1.0).margin(0.01));
    }
    // the two generators are different streams
    CHECK(NormalGenerator(RngKind::CpuCompatible, 1).normal(32) != NormalGenerator(RngKind::Philox, 1).normal(32));
    // a count that is not a multiple of 16 redraws the last block: its start equals the multiple-of-16 prefix only for whole blocks
    const auto whole = NormalGenerator(RngKind::CpuCompatible, 9).normal(32);
    const auto tail = NormalGenerator(RngKind::CpuCompatible, 9).normal(40);
    CHECK(std::equal(whole.begin(), whole.begin() + 32, tail.begin()) == false);   // 40 draws 40 uniforms first, so the stream differs
    // small counts use the scalar path with a cached second value: two calls of 1 equal one call of 2
    NormalGenerator one(RngKind::CpuCompatible, 5), two(RngKind::CpuCompatible, 5);
    const float a = one.normal(1)[0], b = one.normal(1)[0];
    const auto both = two.normal(2);
    CHECK(a == both[0]);
    CHECK(b == both[1]);
    // philox is counter based: asking for 8 then 8 equals 16 at once
    NormalGenerator p1(RngKind::Philox, 3), p2(RngKind::Philox, 3);
    auto first = p1.normal(8);
    const auto second = p1.normal(8);
    first.insert(first.end(), second.begin(), second.end());
    CHECK(first == p2.normal(16));
}

TEST_CASE("brownian tree: consistent in any order, with the variance of a Brownian motion", "[unit][sampling][brownian]") {
    const BrownianTree tree(0.0, 10.0, 4, 123);
    const auto ab = tree.increment(1.0, 3.0), bc = tree.increment(3.0, 7.5), ac = tree.increment(1.0, 7.5);
    for (size_t i = 0; i < 4; ++i) CHECK(ab[i] + bc[i] == Approx(ac[i]).margin(1e-5));
    CHECK(tree.increment(3.0, 1.0)[0] == Approx(-ab[0]).margin(1e-6));
    CHECK(tree.increment(2.0, 2.0) == std::vector<float>(4, 0.0f));
    // asking again, or in another order, gives the same numbers
    CHECK(tree.noise(5.0, 6.0) == tree.noise(5.0, 6.0));
    const auto late = tree.noise(8.0, 9.0);
    const BrownianTree other(0.0, 10.0, 4, 123);
    other.noise(1.0, 2.0);
    CHECK(other.noise(8.0, 9.0) == late);
    CHECK(BrownianTree(0.0, 10.0, 4, 124).noise(8.0, 9.0) != late);

    // normalised increments over many separate intervals are standard normal
    const BrownianTree wide(0.0, 1000.0, 1, 5);
    std::vector<float> samples;
    for (int i = 0; i < 4000; ++i) samples.push_back(wide.noise(i * 0.25, i * 0.25 + 0.2)[0]);
    const Stats s = statsOf(samples);
    CHECK(s.mean == Approx(0.0).margin(0.06));
    CHECK(s.sd == Approx(1.0).margin(0.06));
    // the reversed interval flips the sign
    CHECK(wide.noise(10.0, 9.0)[0] == Approx(-wide.noise(9.0, 10.0)[0]).margin(1e-5));
}

TEST_CASE("noise nodes: random noise carries seed and generator, no noise says none", "[unit][sampling][nodes]") {
    using namespace gh::graph;
    OutputCache cache(*CachePolicy::parse("none"));
    const auto run = [&](const json& nodes) {
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", json::array({"1"})}});
        return Executor(Registry::builtin(), cache).run(job);
    };
    auto r = run({{"1", {{"type", "sampling.noiseRandom"}, {"inputs", {{"seed", 77}, {"generator", "philox"}}}}}});
    REQUIRE(r.ok());
    auto h = r.outputs.at("1")[0].asObject<NoiseHandle>();
    CHECK(h->seed == 77);
    CHECK(h->rng == "philox");
    CHECK_FALSE(h->none);
    CHECK(r.outputs.at("1")[0].type == ValueType::Noise);

    r = run({{"1", {{"type", "sampling.noiseRandom"}, {"inputs", {{"seed", 1}}}}}});   // "setting default" without services
    CHECK(r.outputs.at("1")[0].asObject<NoiseHandle>()->rng == "cpu-compatible");

    r = run({{"1", {{"type", "sampling.noiseNone"}, {"inputs", json::object()}}}});
    REQUIRE(r.ok());
    CHECK(r.outputs.at("1")[0].asObject<NoiseHandle>()->none);
}
