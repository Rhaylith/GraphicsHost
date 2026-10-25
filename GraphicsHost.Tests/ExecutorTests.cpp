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
    std::stop_source stopSource;
    std::function<void(int)> onStep;   // called inside t.steps after each step

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
        registry.add(NodeDef("t.steps").input(Int("steps").def(5)).output(Out("v", ValueType::Int)).run([=](NodeContext& c) {
            count("t.steps");
            const int n = static_cast<int>(c.get("steps").asInt());
            for (int i = 1; i <= n; ++i) {
                if (c.stop.stop_requested()) throw Interrupted();
                if (c.progress) c.progress(i, n);
                if (c.preview) c.preview(Value::makeTensor(ValueType::Image, gh::tensor::Tensor({1, 2, 2, 3})));
                if (onStep) onStep(i);
            }
            return NodeResult{{Value::makeInt(n)}};
        }));
        registry.add(NodeDef("t.out").input(Int("v")).output(Out("v", ValueType::Int)).isOutput().run([=](NodeContext& c) {
            count("t.out");
            return NodeResult{{c.get("v")}};
        }));
    }

    ExecResult run(const json& nodes, json outputs = json::array(), RunOptions options = {}) {
        if (!options.stop.stop_possible()) options.stop = stopSource.get_token();
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

namespace {
struct Recorder {
    std::vector<json> events;
    std::vector<Event> raw;
    EventBus bus;
    EventBus::Id id;
    Recorder() {
        id = bus.subscribe([this](const Event& e) {
            events.push_back(e.toJson());
            raw.push_back(e);
        });
    }
    std::vector<std::string> types() const {
        std::vector<std::string> out;
        for (const auto& e : events) out.push_back(e["type"].get<std::string>() + (e.contains("node") ? ":" + e["node"].get<std::string>() : ""));
        return out;
    }
};
}  // namespace

TEST_CASE("events: a job reports its nodes, progress, previews and summaries", "[unit][executor][events]") {
    Rig rig;
    Recorder rec;
    RunOptions options;
    options.jobId = "job-1";
    options.events = &rec.bus;
    const auto r = rig.run({{"1", n("t.steps", {{"steps", 3}})}, {"2", n("t.out", {{"v", ref("1")}})}}, json::array(), options);
    REQUIRE(r.ok());
    CHECK(rec.types() == std::vector<std::string>{"job.started", "node.started:1", "node.progress:1", "node.preview:1", "node.progress:1", "node.preview:1", "node.progress:1", "node.preview:1",
                                                  "node.finished:1", "node.started:2", "node.finished:2", "job.finished"});
    CHECK(rec.events.front()["total"] == 2);
    CHECK(rec.events.front()["job"] == "job-1");
    CHECK(rec.events[2]["step"] == 1);
    CHECK(rec.events[2]["steps"] == 3);
    REQUIRE(rec.raw[3].preview != nullptr);
    CHECK(rec.raw[3].preview->type == ValueType::Image);
    CHECK(rec.events[8]["outputs"][0] == "INT 3");
    CHECK(rec.events[8]["done"] == 1);
    CHECK(rec.events[8]["millis"].is_number());
    CHECK(rec.events.back()["status"] == "done");
    CHECK(rec.events.back()["done"] == 2);

    // the next run is entirely cached: one event for the output node
    rec.events.clear();
    const auto again = rig.run({{"1", n("t.steps", {{"steps", 3}})}, {"2", n("t.out", {{"v", ref("1")}})}}, json::array(), options);
    REQUIRE(again.ok());
    CHECK(rec.types() == std::vector<std::string>{"job.started", "node.cached:2", "job.finished"});
}

TEST_CASE("events: errors name the node and the input", "[unit][executor][events]") {
    Rig rig;
    rig.registry.add(NodeDef("t.picky").input(Int("v").def(0)).output(Out("v", ValueType::Int)).run([](NodeContext&) -> NodeResult { throw NodeError("must not be zero", "v"); }));
    Recorder rec;
    RunOptions options;
    options.events = &rec.bus;
    const auto r = rig.run({{"1", n("t.picky")}, {"2", n("t.out", {{"v", ref("1")}})}}, json::array(), options);
    CHECK(r.status == ExecResult::Status::Failed);
    CHECK(r.errorNode == "1");
    CHECK(r.errorInput == "v");
    CHECK(rec.types() == std::vector<std::string>{"job.started", "node.started:1", "node.error:1", "job.finished"});
    CHECK(rec.events[2]["message"] == "must not be zero");
    CHECK(rec.events[2]["input"] == "v");
    CHECK(rec.events[2]["nodeType"] == "t.picky");
    CHECK(rec.events.back()["status"] == "failed");
    CHECK(rec.events.back()["message"] == "must not be zero");
}

TEST_CASE("events: interrupting stops between nodes and inside a node's loop", "[unit][executor][events]") {
    Rig rig;
    // inside a node: stop after step 2 of 10
    rig.onStep = [&](int i) {
        if (i == 2) rig.stopSource.request_stop();
    };
    Recorder rec;
    RunOptions options;
    options.events = &rec.bus;
    const auto r = rig.run({{"1", n("t.steps", {{"steps", 10}})}, {"2", n("t.out", {{"v", ref("1")}})}}, json::array(), options);
    CHECK(r.status == ExecResult::Status::Interrupted);
    CHECK(r.errorNode.empty());
    CHECK(rig.runs["t.out"] == 0);
    CHECK(rec.events.back()["status"] == "interrupted");
    int progress = 0;
    for (const auto& e : rec.events) progress += e["type"] == "node.progress";
    CHECK(progress == 2);

    // between nodes: a token that is already set runs nothing
    Rig rig2;
    std::stop_source stopped;
    stopped.request_stop();
    RunOptions pre;
    pre.stop = stopped.get_token();
    const auto none = rig2.run({{"1", n("t.const", {{"value", 1}})}, {"2", n("t.out", {{"v", ref("1")}})}}, json::array(), pre);
    CHECK(none.status == ExecResult::Status::Interrupted);
    CHECK(rig2.runs.empty());

    // an interrupted job's finished nodes stay cached for the next run
    Rig rig3;
    rig3.onStep = [&](int) {};
    json g = {{"1", n("t.const", {{"value", 1}})}, {"2", n("t.steps", {{"steps", 3}})}, {"3", n("t.add", {{"a", ref("1")}, {"b", ref("2")}})}, {"4", n("t.out", {{"v", ref("3")}})}};
    rig3.onStep = [&](int i) {
        if (i == 1) rig3.stopSource.request_stop();
    };
    CHECK(rig3.run(g).status == ExecResult::Status::Interrupted);
    rig3.stopSource = std::stop_source();
    rig3.onStep = nullptr;
    const auto resumed = rig3.run(g);
    REQUIRE(resumed.ok());
    CHECK(rig3.runs["t.const"] == 1);   // finished before the interrupt, reused
    CHECK(resumed.outputs.at("4")[0].asInt() == 4);
}

TEST_CASE("events: the bus delivers to every subscriber until they leave", "[unit][events]") {
    EventBus bus;
    int a = 0, b = 0;
    const auto ia = bus.subscribe([&](const Event&) { ++a; });
    bus.subscribe([&](const Event&) { ++b; });
    Event e;
    bus.publish(e);
    bus.unsubscribe(ia);
    bus.publish(e);
    CHECK(a == 1);
    CHECK(b == 2);
    CHECK(std::string(eventName(Event::Kind::NodePreview)) == "node.preview");
}
