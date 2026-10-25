#include "graph/Executor.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>

using namespace gh::graph;
using nlohmann::json;

namespace {

struct Rig {
    Registry registry;
    std::map<std::string, int> runs;   // by node type
    OutputCache cache;
    int fileVersion = 1;

    Rig() {
        auto count = [this](const char* type) { ++runs[type]; };
        registry.add(NodeDef("t.const").input(Int("value").def(0)).output(Out("v", ValueType::Int)).run([=](NodeContext& c) {
            count("t.const");
            return NodeResult{{Value::makeInt(c.get("value").asInt())}};
        }));
        registry.add(NodeDef("t.add").input(Int("a").def(0)).input(Int("b").def(0)).output(Out("v", ValueType::Int)).run([=](NodeContext& c) {
            count("t.add");
            return NodeResult{{Value::makeInt(c.get("a").asInt() + c.get("b").asInt())}};
        }));
        registry.add(NodeDef("t.range").input(Int("n").def(3)).output(Out("v", ValueType::Int).list()).run([=](NodeContext& c) {
            count("t.range");
            ValueList items;
            for (int64_t i = 0; i < c.get("n").asInt(); ++i) items.push_back(Value::makeInt(i));
            return NodeResult{{Value::makeList(ValueType::Int, items)}};
        }));
        registry.add(NodeDef("t.sum").acceptsLists().input(Int("v")).output(Out("v", ValueType::Int)).run([=](NodeContext& c) {
            count("t.sum");
            int64_t total = 0;
            for (const Value& v : c.get("v").list()) total += v.asInt();
            return NodeResult{{Value::makeInt(total)}};
        }));
        registry.add(NodeDef("t.pick")
                         .input(Bool("first").def(true))
                         .input(Int("a").lazy())
                         .input(Int("b").lazy())
                         .output(Out("v", ValueType::Int))
                         .run([=](NodeContext& c) {
                             count("t.pick");
                             const char* want = c.get("first").asBool() ? "a" : "b";
                             if (!c.has(want)) return NodeResult{{}, {want}, std::nullopt};
                             return NodeResult{{c.get(want)}};
                         }));
        registry.add(NodeDef("t.double").input(Int("v")).output(Out("v", ValueType::Int)).expands().run([=](NodeContext& c) {
            count("t.double");
            Expansion e;
            JobNode one;   // inner "x" adds the parent's input to itself
            one.type = "t.add";
            one.inputs["a"] = c.links.at("v");
            one.inputs["b"] = c.links.at("v");
            e.graph.nodes["x"] = one;
            JobNode two;
            two.type = "t.add";
            two.inputs["a"] = NodeRef{"x", 0};
            two.inputs["b"] = json(0);
            e.graph.nodes["y"] = two;
            e.outputs = {NodeRef{"y", 0}};
            NodeResult r;
            r.expansion = std::move(e);
            return r;
        }));
        registry.add(NodeDef("t.badexpand").input(Int("v")).output(Out("v", ValueType::Int)).run([=](NodeContext&) {
            Expansion e;
            JobNode bad;
            bad.type = "t.nonexistent";
            e.graph.nodes["x"] = bad;
            e.outputs = {NodeRef{"x", 0}};
            NodeResult r;
            r.expansion = std::move(e);
            return r;
        }));
        registry.add(NodeDef("t.fail").input(Int("v").def(0)).output(Out("v", ValueType::Int)).run([=](NodeContext&) -> NodeResult { throw NodeError("it broke"); }));
        registry.add(NodeDef("t.file").input(String("name").def("a")).output(Out("v", ValueType::Int)).fingerprint([this](const auto&) { return std::to_string(fileVersion); }).run([=](NodeContext&) {
            count("t.file");
            return NodeResult{{Value::makeInt(fileVersion)}};
        }));
        registry.add(NodeDef("t.live").output(Out("v", ValueType::Int)).fingerprint([](const auto&) { return std::string("*"); }).run([=](NodeContext&) {
            count("t.live");
            return NodeResult{{Value::makeInt(1)}};
        }));
        registry.add(NodeDef("t.out").input(Int("v")).output(Out("v", ValueType::Int)).isOutput().run([=](NodeContext& c) {
            count("t.out");
            return NodeResult{{c.get("v")}};
        }));
    }

    ExecResult run(const json& nodes, json outputs = json::array(), RunOptions options = {}) {
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", outputs}});
        return Executor(registry, cache).run(job, options);
    }
};

json n(const char* type, json inputs = json::object()) { return {{"type", type}, {"inputs", inputs}}; }
json ref(const char* id, int out = 0) { return json::array({id, out}); }

}  // namespace

TEST_CASE("executor: runs the graph and returns the outputs of the requested nodes", "[unit][executor]") {
    Rig rig;
    const auto r = rig.run({{"1", n("t.const", {{"value", 2}})}, {"2", n("t.const", {{"value", 40}})}, {"3", n("t.add", {{"a", ref("1")}, {"b", ref("2")}})}, {"4", n("t.out", {{"v", ref("3")}})}});
    REQUIRE(r.ok());
    CHECK(r.outputs.at("4")[0].asInt() == 42);
    CHECK(r.ran == std::vector<std::string>{"1", "2", "3", "4"});
    CHECK(r.millis.size() == 4);
    CHECK(rig.runs["t.add"] == 1);
}

TEST_CASE("executor: a partial run only runs what the requested output needs", "[unit][executor]") {
    Rig rig;
    const auto r = rig.run({{"1", n("t.const", {{"value", 1}})}, {"2", n("t.const", {{"value", 2}})}, {"3", n("t.out", {{"v", ref("1")}})}, {"4", n("t.out", {{"v", ref("2")}})}}, json::array({"4"}));
    REQUIRE(r.ok());
    CHECK(r.ran == std::vector<std::string>{"2", "4"});
    CHECK(r.outputs.count("3") == 0);
}

TEST_CASE("executor: unchanged nodes come from the cache on the next run", "[unit][executor]") {
    Rig rig;
    json graph = {{"1", n("t.const", {{"value", 5}})}, {"2", n("t.add", {{"a", ref("1")}, {"b", 1}})}, {"3", n("t.out", {{"v", ref("2")}})}};
    REQUIRE(rig.run(graph).ok());
    CHECK(rig.runs["t.add"] == 1);

    auto again = rig.run(graph);
    REQUIRE(again.ok());
    CHECK(again.cached == std::vector<std::string>{"3"});   // the output node itself is cached, nothing upstream needs to be visited
    CHECK(again.ran.empty());
    CHECK(rig.runs["t.add"] == 1);
    CHECK(again.outputs.at("3")[0].asInt() == 6);

    // change one literal: only that node and what depends on it run again
    graph["1"]["inputs"]["value"] = 7;
    auto changed = rig.run(graph);
    CHECK(changed.ran == std::vector<std::string>{"1", "2", "3"});
    CHECK(changed.outputs.at("3")[0].asInt() == 8);

    graph["2"]["inputs"]["b"] = 10;   // the const is unchanged
    auto partial = rig.run(graph);
    CHECK(partial.cached == std::vector<std::string>{"1"});
    CHECK(partial.ran == std::vector<std::string>{"2", "3"});

    // fingerprints: a file that changed on disk, and a node that must always run
    Rig files;
    json g = {{"1", n("t.file")}, {"2", n("t.live")}, {"3", n("t.add", {{"a", ref("1")}, {"b", ref("2")}})}, {"4", n("t.out", {{"v", ref("3")}})}};
    REQUIRE(files.run(g).ok());
    REQUIRE(files.run(g).ok());
    CHECK(files.runs["t.file"] == 1);   // cached
    CHECK(files.runs["t.live"] == 2);   // always run
    CHECK(files.runs["t.add"] == 2);    // depends on the live node, so it never caches either
    files.fileVersion = 2;
    auto updated = files.run(g);
    CHECK(files.runs["t.file"] == 2);
    CHECK(updated.outputs.at("4")[0].asInt() == 3);

    // policy none: nothing is kept between jobs
    Rig none;
    none.cache.setPolicy(*CachePolicy::parse("none"));
    REQUIRE(none.run(graph).ok());
    REQUIRE(none.run(graph).ok());
    CHECK(none.runs["t.add"] == 2);
}

TEST_CASE("executor: lazy inputs are evaluated only when the node asks", "[unit][executor]") {
    Rig rig;
    json g = {{"1", n("t.const", {{"value", 10}})}, {"2", n("t.const", {{"value", 20}})}, {"3", n("t.pick", {{"first", true}, {"a", ref("1")}, {"b", ref("2")}})}, {"4", n("t.out", {{"v", ref("3")}})}};
    auto r = rig.run(g);
    REQUIRE(r.ok());
    CHECK(r.outputs.at("4")[0].asInt() == 10);
    CHECK(r.ran == std::vector<std::string>{"1", "3", "4"});   // the const feeding "b" never ran
    CHECK(rig.runs["t.const"] == 1);
    CHECK(rig.runs["t.pick"] == 2);   // once to ask, once to answer

    g["3"]["inputs"]["first"] = false;
    r = rig.run(g);
    REQUIRE(r.ok());
    CHECK(r.outputs.at("4")[0].asInt() == 20);
    CHECK(r.ran.size() == 3);
    CHECK(r.ran[0] == "2");   // the other branch, now asked for; node 1 is not needed this time
}

TEST_CASE("executor: list outputs map over nodes that take single values", "[unit][executor]") {
    Rig rig;
    // range(4) -> add(a = item, b = 10) -> sum
    const auto r = rig.run({{"1", n("t.range", {{"n", 4}})}, {"2", n("t.add", {{"a", ref("1")}, {"b", 10}})}, {"3", n("t.sum", {{"v", ref("2")}})}, {"4", n("t.out", {{"v", ref("3")}})}});
    REQUIRE(r.ok());
    CHECK(rig.runs["t.add"] == 4);
    CHECK(r.outputs.at("4")[0].asInt() == 46);   // (0+1+2+3) + 4 * 10

    // two lists of equal length are walked together: (0+0) + (1+1) + (2+2)
    const auto two = rig.run({{"1", n("t.range", {{"n", 3}})}, {"5", n("t.range", {{"n", 3}})}, {"6", n("t.add", {{"a", ref("1")}, {"b", ref("5")}})}, {"2", n("t.sum", {{"v", ref("6")}})}, {"3", n("t.out", {{"v", ref("2")}})}});
    REQUIRE(two.ok());
    CHECK(two.outputs.at("3")[0].asInt() == 6);

    // unequal lengths fail, naming the node
    const auto bad = rig.run({{"1", n("t.range", {{"n", 3}})}, {"5", n("t.range", {{"n", 2}})}, {"6", n("t.add", {{"a", ref("1")}, {"b", ref("5")}})}, {"2", n("t.sum", {{"v", ref("6")}})}, {"3", n("t.out", {{"v", ref("2")}})}});
    CHECK(bad.status == ExecResult::Status::Failed);
    CHECK(bad.errorNode == "6");
    CHECK(bad.errorMessage.find("differ in length") != std::string::npos);
}

TEST_CASE("executor: a node can expand into a sub-graph with stable ids", "[unit][executor]") {
    Rig rig;
    json g = {{"1", n("t.const", {{"value", 21}})}, {"2", n("t.double", {{"v", ref("1")}})}, {"3", n("t.out", {{"v", ref("2")}})}};
    auto r = rig.run(g);
    REQUIRE(r.ok());
    CHECK(r.outputs.at("3")[0].asInt() == 42);
    CHECK(std::find(r.ran.begin(), r.ran.end(), "2:x") != r.ran.end());
    CHECK(std::find(r.ran.begin(), r.ran.end(), "2:y") != r.ran.end());
    CHECK(rig.runs["t.add"] == 2);

    // the same job again: everything cached, the expansion is not even asked for
    r = rig.run(g);
    CHECK(r.ran.empty());
    CHECK(rig.runs["t.double"] == 1);

    // a different input value re-runs the node and its inner nodes
    g["1"]["inputs"]["value"] = 5;
    r = rig.run(g);
    REQUIRE(r.ok());
    CHECK(r.outputs.at("3")[0].asInt() == 10);

    // an invalid expansion is a failure of the expanding node, with the reason
    const auto bad = rig.run({{"1", n("t.const")}, {"2", n("t.badexpand", {{"v", ref("1")}})}, {"3", n("t.out", {{"v", ref("2")}})}});
    CHECK(bad.status == ExecResult::Status::Failed);
    CHECK(bad.errorNode == "2");
    CHECK(bad.errorMessage.find("expansion is invalid") != std::string::npos);
}

TEST_CASE("executor: failures end the job and name the node", "[unit][executor]") {
    Rig rig;
    const auto r = rig.run({{"1", n("t.const", {{"value", 1}})}, {"2", n("t.fail", {{"v", ref("1")}})}, {"3", n("t.out", {{"v", ref("2")}})}});
    CHECK(r.status == ExecResult::Status::Failed);
    CHECK_FALSE(r.ok());
    CHECK(r.errorNode == "2");
    CHECK(r.errorMessage == "it broke");
    CHECK(rig.runs["t.out"] == 0);

    const auto loop = rig.run({{"1", n("t.add", {{"a", ref("2")}})}, {"2", n("t.add", {{"a", ref("1")}})}, {"3", n("t.out", {{"v", ref("1")}})}});
    CHECK(loop.status == ExecResult::Status::Failed);
    CHECK(loop.errorMessage.find("loop") != std::string::npos);

    const auto unknown = rig.run({{"1", n("plugin.nothing")}, {"2", n("t.out", {{"v", ref("1")}})}});
    CHECK(unknown.status == ExecResult::Status::Failed);
    CHECK(unknown.errorNode == "1");
}
