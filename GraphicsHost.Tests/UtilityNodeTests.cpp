#include "RepoRoot.h"
#include "graph/Executor.h"
#include "graph/ReferenceDiff.h"
#include "nodes/MathExpression.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <fstream>

using namespace gh::graph;
using nlohmann::json;

namespace {
ExecResult runJob(const json& nodes, json outputs = json::array()) {
    static const Registry& registry = Registry::builtin();
    OutputCache cache(*CachePolicy::parse("none"));
    const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", outputs}});
    return Executor(registry, cache).run(job);
}
json n(const char* type, json inputs = json::object()) { return {{"type", type}, {"inputs", inputs}}; }
json ref(const char* id, int out = 0) { return json::array({id, out}); }

// runs one utility node with literal inputs and returns one of its outputs as text
std::string eval(const char* type, json inputs, int output = 0) {
    const auto r = runJob({{"1", n(type, inputs)}, {"2", n("util.previewAny", {{"value", ref("1", output)}})}});
    if (!r.ok()) return "ERROR: " + r.errorMessage;
    return r.outputs.at("2")[0].asString();
}
}  // namespace

TEST_CASE("math: expressions evaluate with the usual precedence", "[unit][nodes][math]") {
    using gh::nodes::evaluateMath;
    const std::map<char, double> v = {{'a', 10}, {'b', 4}, {'c', 0.5}};
    CHECK(evaluateMath("a + b * 2", v) == 18);
    CHECK(evaluateMath("(a + b) * 2", v) == 28);
    CHECK(evaluateMath("a / b", v) == 2.5);
    CHECK(evaluateMath("a % 4", v) == 2);
    CHECK(evaluateMath("2 ^ 3 ^ 2", v) == 512);
    CHECK(evaluateMath("-a + 3", v) == -7);
    CHECK(evaluateMath("a * 1.5", v) == 15);
    CHECK(evaluateMath("min(a, b) + max(a, b)", v) == 14);
    CHECK(evaluateMath("round(2.5) + floor(2.9) + ceil(2.1) + abs(-1) + sqrt(16)", v) == 3 + 2 + 3 + 1 + 4);
    CHECK(evaluateMath("a \xC3\x97 b \xE2\x88\x92 1 \xC3\xB7 c", v) == 38);   // x - /  written as the signs in the reference
    CHECK(evaluateMath("  7  ", v) == 7);
    CHECK_THROWS_WITH(evaluateMath("a / 0", v), Catch::Matchers::ContainsSubstring("division by zero"));
    CHECK_THROWS_WITH(evaluateMath("z + 1", v), Catch::Matchers::ContainsSubstring("unknown name 'z'"));
    CHECK_THROWS_WITH(evaluateMath("(1 + 2", v), Catch::Matchers::ContainsSubstring("parenthesis"));
    CHECK_THROWS_WITH(evaluateMath("1 +", v), Catch::Matchers::ContainsSubstring("ends too early"));
    CHECK_THROWS_WITH(evaluateMath("3 3", v), Catch::Matchers::ContainsSubstring("unexpected"));
    CHECK_THROWS_WITH(evaluateMath("min(1)", v), Catch::Matchers::ContainsSubstring("takes 2"));
    CHECK_THROWS_WITH(evaluateMath("nope(1)", v), Catch::Matchers::ContainsSubstring("unknown function"));
    CHECK_THROWS_WITH(evaluateMath("sqrt(-1)", v), Catch::Matchers::ContainsSubstring("negative"));
}

TEST_CASE("utility nodes: constants pass their value on", "[unit][nodes][utility]") {
    CHECK(eval("util.int", {{"value", 1024}}) == "1024");
    CHECK(eval("util.float", {{"value", 7.5}}) == "7.5");
    CHECK(eval("util.string", {{"value", "a fox"}}) == "a fox");
    CHECK(eval("util.bool", {{"value", true}}) == "true");
    CHECK(eval("util.primitive", {{"value", 3}}) == "3");
    CHECK(eval("util.primitive", {{"value", "text"}}) == "text");
    CHECK(eval("util.reroute", {{"value", 9}}) == "9");
    const auto r = runJob({{"1", n("util.note", {{"text", "hello"}})}, {"2", n("util.int", {{"value", 1}})}, {"3", n("util.previewAny", {{"value", ref("2")}})}});
    CHECK(r.ok());
    CHECK(r.ran.size() == 2);   // the note is not executed
}

TEST_CASE("utility nodes: text tools", "[unit][nodes][utility]") {
    CHECK(eval("util.textJoin", {{"a", "a red fox"}, {"b", "in snow"}}) == "a red fox, in snow");
    CHECK(eval("util.textJoin", {{"a", "x"}, {"b", "y"}, {"separator", " BREAK "}}) == "x BREAK y");
    CHECK(eval("util.textReplace", {{"text", "a cat, a cat"}, {"find", "cat"}, {"replace", "dog"}}) == "a dog, a dog");
    CHECK(eval("util.textReplace", {{"text", "a cat"}, {"find", "cat"}}) == "a ");
    CHECK(eval("util.textReplace", {{"text", "a cat"}, {"find", ""}, {"replace", "x"}}) == "a cat");
    CHECK(eval("util.textReplace", {{"text", "black cat"}, {"find", "(black|white) cat"}, {"replace", "$1 dog"}, {"regex", true}}) == "black dog");
    CHECK(eval("util.textReplace", {{"text", "x"}, {"find", "("}, {"regex", true}}).find("ERROR: the pattern is not valid") == 0);
    CHECK(eval("util.textSlice", {{"text", "abcdefgh"}, {"start", 0}, {"end", 3}}) == "abc");
    CHECK(eval("util.textSlice", {{"text", "abcdefgh"}, {"start", 2}}) == "cdefgh");   // the default end is -1: the end
    CHECK(eval("util.textSlice", {{"text", "abcdefgh"}, {"start", -3}}) == "fgh");
    CHECK(eval("util.textSlice", {{"text", "abcdefgh"}, {"start", 5}, {"end", 2}}) == "");
    CHECK(eval("util.textSlice", {{"text", "abc"}, {"start", 0}, {"end", 99}}) == "abc");
}

TEST_CASE("utility nodes: compare, math and image size", "[unit][nodes][utility]") {
    CHECK(eval("util.compare", {{"a", 1280}, {"b", 1024}, {"operation", ">"}}) == "true");
    CHECK(eval("util.compare", {{"a", 1024}, {"b", 1024.0}, {"operation", "="}}) == "true");
    CHECK(eval("util.compare", {{"a", 5}, {"b", 7}, {"operation", "\xE2\x89\xA5"}}) == "false");   // >=
    CHECK(eval("util.compare", {{"a", "apple"}, {"b", "banana"}, {"operation", "<"}}) == "true");
    CHECK(eval("util.compare", {{"a", "x"}, {"b", "x"}, {"operation", "\xE2\x89\xA0"}}) == "false");   // !=
    CHECK(eval("util.math", {{"expression", "a * 1.5"}, {"a", 1024}}) == "1536");
    CHECK(eval("util.math", {{"expression", "a / 3"}, {"a", 10}}, 1) == "3");   // the whole-number output
    CHECK(eval("util.math", {{"expression", "a +"}}).find("ERROR: the expression ends too early") == 0);

    const auto size = runJob({{"1", n("util.int", {{"value", 1}})}, {"2", n("util.imageSize", {{"image", ref("1")}})}, {"3", n("util.previewAny", {{"value", ref("2", 0)}})}});
    CHECK_FALSE(size.ok());   // an INT is not an image: the node reports it
}

TEST_CASE("utility nodes: switch runs only the chosen branch", "[unit][nodes][utility]") {
    OutputCache cache(*CachePolicy::parse("none"));
    int ran = 0;
    Registry r;
    for (const NodeDef* d : Registry::builtin().all()) r.add(*d);
    r.add(NodeDef("t.counted").input(Int("v").def(0)).output(Out("v", ValueType::Int)).run([&ran](NodeContext& c) {
        ++ran;
        return NodeResult{{c.get("v")}};
    }));
    const auto go = [&](bool cond) {
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"},
                                          {"version", 1},
                                          {"nodes", {{"1", n("t.counted", {{"v", 10}})}, {"2", n("t.counted", {{"v", 20}})},
                                                     {"3", n("util.switch", {{"condition", cond}, {"onTrue", ref("1")}, {"onFalse", ref("2")}})},
                                                     {"4", n("util.previewAny", {{"value", ref("3")}})}}}});
        ran = 0;
        return Executor(r, cache).run(job);
    };
    auto yes = go(true);
    REQUIRE(yes.ok());
    CHECK(yes.outputs.at("4")[0].asString() == "10");
    CHECK(ran == 1);
    auto no = go(false);
    REQUIRE(no.ok());
    CHECK(no.outputs.at("4")[0].asString() == "20");
    CHECK(ran == 1);

    const auto missing = runJob({{"3", n("util.switch", {{"condition", true}})}, {"4", n("util.previewAny", {{"value", ref("3")}})}});
    CHECK_FALSE(missing.ok());
    CHECK(missing.errorInput == "onTrue");
}

TEST_CASE("utility nodes: the registry holds them and can write their reference rows", "[unit][nodes][utility][reference-rows]") {
    const Registry& r = Registry::builtin();
    CHECK(r.find("util.switch") != nullptr);
    CHECK(r.size() == 16);

    // writes the rows for the document: GH_WRITE_REFERENCE=<file> GraphicsHost.Tests.exe "[reference-rows]"
    if (const char* out = std::getenv("GH_WRITE_REFERENCE")) std::ofstream(out, std::ios::binary) << renderReference(r);
}
