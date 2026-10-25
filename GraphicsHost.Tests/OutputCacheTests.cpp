#include "graph/OutputCache.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::graph;
using nlohmann::json;

namespace {
Registry registry() {
    Registry r;
    r.add(NodeDef("t.n").input(Int("a").def(5)).input(Float("f").def(1.0)).input(String("s").def("x")).input(Socket("up", ValueType::Int).optional()).output(Out("v", ValueType::Int)));
    r.add(NodeDef("t.file").input(String("path").def("a.png")).fingerprint([](const auto& lit) { return "mtime:" + lit.at("path").template get<std::string>(); }));
    r.add(NodeDef("t.live").fingerprint([](const auto&) { return std::string("*"); }));
    return r;
}
JobNode node(const char* type, json inputs = json::object()) {
    JobNode n;
    n.type = type;
    for (const auto& [k, v] : inputs.items()) n.inputs[k] = v;
    return n;
}
NodeOutputs ints(int64_t v) { return {Value::makeInt(v)}; }
}  // namespace

TEST_CASE("cache key: stable for equal nodes, different when anything that matters differs", "[unit][cache]") {
    const Registry r = registry();
    const auto key = [&](const JobNode& n, std::map<std::string, Upstream> up = {}) { return nodeCacheKey(n, r, up); };
    const std::string base = key(node("t.n", {{"a", 5}, {"s", "x"}}));
    CHECK(base.size() == 64);
    CHECK(key(node("t.n")) == base);                          // defaults filled in
    CHECK(key(node("t.n", {{"s", "x"}, {"a", 5}})) == base);  // order of inputs does not matter
    CHECK(key(node("t.n", {{"a", 5.0}})) == base);            // 5 and 5.0 are the same INT
    CHECK(key(node("t.n", {{"f", 1}})) == base);              // and 1 is the same FLOAT as 1.0
    CHECK(key(node("t.n", {{"a", 6}})) != base);
    CHECK(key(node("t.n", {{"s", "y"}})) != base);

    const std::map<std::string, Upstream> up1 = {{"up", {"k1", 0}}}, up2 = {{"up", {"k2", 0}}}, up1b = {{"up", {"k1", 1}}};
    const std::string withUp = key(node("t.n"), up1);
    CHECK(withUp != base);
    CHECK(key(node("t.n"), up1) == withUp);
    CHECK(key(node("t.n"), up2) != withUp);   // the source changed
    CHECK(key(node("t.n"), up1b) != withUp);  // another output of the same source

    // the node's own fingerprint is part of the key; "*" means never cache; unknown types are never cached
    CHECK(key(node("t.file", {{"path", "a.png"}})) != key(node("t.file", {{"path", "b.png"}})));
    CHECK(key(node("t.file")) == key(node("t.file", {{"path", "a.png"}})));
    CHECK(key(node("t.live")).empty());
    CHECK(key(node("plugin.unknown")).empty());
}

TEST_CASE("cache policy: text form", "[unit][cache]") {
    CHECK(CachePolicy::parse("last")->kind == CachePolicy::Kind::Last);
    CHECK(CachePolicy::parse("none")->kind == CachePolicy::Kind::None);
    CHECK(CachePolicy::parse("lru:20")->entries == 20);
    CHECK(CachePolicy::parse("ram:16")->freeRamGb == 16);
    CHECK(CachePolicy::parse("ram:1.5")->toString() == "ram:1.5");
    CHECK(CachePolicy::parse("lru:20")->toString() == "lru:20");
    CHECK_FALSE(CachePolicy::parse("lru").has_value());
    CHECK_FALSE(CachePolicy::parse("lru:0").has_value());
    CHECK_FALSE(CachePolicy::parse("lru:x").has_value());
    CHECK_FALSE(CachePolicy::parse("fifo:3").has_value());
}

TEST_CASE("cache policy last: keeps what the most recent job used", "[unit][cache]") {
    OutputCache cache(*CachePolicy::parse("last"));
    cache.beginJob();
    cache.put("a", ints(1));
    cache.put("b", ints(2));
    cache.endJob();
    CHECK(cache.size() == 2);

    cache.beginJob();
    REQUIRE(cache.find("a") != nullptr);
    CHECK(cache.find("a")->at(0).asInt() == 1);
    CHECK(cache.find("zzz") == nullptr);
    cache.put("c", ints(3));
    cache.endJob();   // b was not used: gone
    CHECK(cache.size() == 2);
    cache.beginJob();
    CHECK(cache.find("b") == nullptr);
    CHECK(cache.find("c") != nullptr);
    cache.endJob();   // a was not used this time
    CHECK(cache.size() == 1);
}

TEST_CASE("cache policy lru: keeps the most recently used entries", "[unit][cache]") {
    OutputCache cache(*CachePolicy::parse("lru:2"));
    cache.beginJob();
    cache.put("a", ints(1));
    cache.put("b", ints(2));
    cache.put("c", ints(3));
    cache.endJob();
    CHECK(cache.size() == 2);   // a was the oldest
    cache.beginJob();
    CHECK(cache.find("a") == nullptr);
    CHECK(cache.find("b") != nullptr);   // b is now the most recent
    cache.put("d", ints(4));
    cache.endJob();                      // evicts c
    cache.beginJob();
    CHECK(cache.find("b") != nullptr);
    CHECK(cache.find("c") == nullptr);
    CHECK(cache.find("d") != nullptr);
}

TEST_CASE("cache policy ram: evicts while free memory is under the threshold; none keeps nothing", "[unit][cache]") {
    uint64_t freeGb = 4;   // each call after an eviction round sees one more GB
    OutputCache ram(*CachePolicy::parse("ram:6"), [&] { return (freeGb++) * 1024ull * 1024 * 1024; });
    ram.beginJob();
    ram.put("a", ints(1));
    ram.put("b", ints(2));
    ram.put("c", ints(3));
    ram.endJob();
    CHECK(ram.size() == 1);   // 4 GB free: evict a; 5 GB: evict b; 6 GB: enough

    OutputCache plenty(*CachePolicy::parse("ram:1"), [] { return 64ull << 30; });
    plenty.beginJob();
    plenty.put("a", ints(1));
    plenty.endJob();
    CHECK(plenty.size() == 1);

    OutputCache none(*CachePolicy::parse("none"));
    none.beginJob();
    none.put("a", ints(1));
    CHECK(none.find("a") == nullptr);
    none.endJob();
    CHECK(none.size() == 0);

    // switching the policy applies to what is stored
    OutputCache cache;
    cache.beginJob();
    cache.put("a", ints(1));
    cache.setPolicy(*CachePolicy::parse("none"));
    CHECK(cache.size() == 0);
    CHECK(cache.policy().kind == CachePolicy::Kind::None);

    // empty keys (nodes that must always run) are never stored; sizes are tracked
    OutputCache sized;
    sized.beginJob();
    sized.put("", ints(1));
    CHECK(sized.size() == 0);
    sized.put("img", {Value::makeTensor(ValueType::Image, gh::tensor::Tensor({1, 8, 8, 3}))});
    CHECK(sized.bytes() >= 8 * 8 * 3 * sizeof(float));
    sized.clear();
    CHECK(sized.bytes() == 0);

    CHECK(OutputCache::systemFreeRamBytes() > 0);
}
