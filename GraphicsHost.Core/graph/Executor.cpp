#include "graph/Executor.h"

#include "graph/Validator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

namespace gh::graph {

using Json = nlohmann::json;

namespace {

// A failure tied to the node it happened in. Thrown by the innermost node; outer evaluations let it pass.
struct NodeFailure : std::runtime_error {
    NodeFailure(std::string n, const std::string& message, std::string in = {}) : std::runtime_error(message), node(std::move(n)), input(std::move(in)) {}
    std::string node;
    std::string input;   // the input to blame, when known
};

constexpr int kMaxExpansionDepth = 32;

Value literalValue(const InputDef& in, const Json& v) {
    const auto scalar = [&](const Json& j) -> Value {
        switch (in.type) {
            case ValueType::Int: return Value::makeInt(static_cast<int64_t>(std::llround(j.get<double>())));
            case ValueType::Float: return Value::makeFloat(j.get<double>());
            case ValueType::Boolean: return Value::makeBool(j.get<bool>());
            case ValueType::String:
            case ValueType::Combo: return Value::makeString(j.get<std::string>());
            case ValueType::Any:   // the type is whatever the JSON says
                if (j.is_boolean()) return Value::makeBool(j.get<bool>());
                if (j.is_number_integer()) return Value::makeInt(j.get<int64_t>());
                if (j.is_number()) return Value::makeFloat(j.get<double>());
                if (j.is_string()) return Value::makeString(j.get<std::string>());
                throw NodeError("'" + in.name + "' takes a number, text or true/false");
            default: throw NodeError("'" + in.name + "' takes a link, not a typed value");
        }
    };
    if (v.is_array()) {
        ValueList items;
        for (const auto& j : v) items.push_back(scalar(j));
        return Value::makeList(in.type, std::move(items));
    }
    return scalar(v);
}

class Run {
public:
    Run(const Registry& registry, OutputCache& cache, const JobGraph& job, const RunOptions& options)
        : registry_(registry), cache_(cache), nodes_(job.nodes.begin(), job.nodes.end()), job_(job), options_(options) {}

    ExecResult go() {
        ExecResult result;
        try {
            const Plan plan = makePlan(job_, registry_);
            total_ = static_cast<int>(plan.order.size());
            cache_.beginJob();
            Event started;
            started.kind = Event::Kind::JobStarted;
            started.total = total_;
            emit(started);
            // demand-driven: a root that is in the cache keeps everything behind it from being visited
            for (const auto& root : plan.roots) result.outputs[root] = evaluate(root);
        } catch (const NodeFailure& f) {
            result.status = ExecResult::Status::Failed;
            result.errorNode = f.node;
            result.errorInput = f.input;
            result.errorMessage = f.what();
            Event e;
            e.kind = Event::Kind::NodeError;
            e.node = f.node;
            e.input = f.input;
            e.message = f.what();
            if (nodes_.count(f.node)) e.type = nodes_.at(f.node).type;
            emit(e);
        } catch (const Interrupted&) {
            result.status = ExecResult::Status::Interrupted;
        } catch (const std::exception& e) {   // PlanError, or something unexpected: the job fails, the process does not
            result.status = ExecResult::Status::Failed;
            result.errorMessage = e.what();
        }
        Event finished;
        finished.kind = Event::Kind::JobFinished;
        finished.status = result.status == ExecResult::Status::Done ? "done" : result.status == ExecResult::Status::Failed ? "failed" : "interrupted";
        finished.message = result.errorMessage;
        finished.done = done_;
        finished.total = total_;
        emit(finished);
        cache_.endJob();
        result.ran = std::move(ran_);
        result.cached = std::move(cachedNodes_);
        result.millis = std::move(millis_);
        return result;
    }

private:
    using Outputs = std::shared_ptr<const NodeOutputs>;

    const NodeDef& definition(const std::string& id) const {
        const NodeDef* def = registry_.find(nodes_.at(id).type);
        if (!def) throw NodeFailure(id, "unknown node type '" + nodes_.at(id).type + "'");
        return *def;
    }

    // The cache key of a node, "" when it must always run. Computed from the structure of the graph alone.
    const std::string& keyOf(const std::string& id) {
        const auto known = keys_.find(id);
        if (known != keys_.end()) return known->second;
        keys_[id] = "";   // guards against loops through expansions; replaced below
        const JobNode& node = nodes_.at(id);
        std::map<std::string, Upstream> upstream;
        bool cacheable = true;
        for (const auto& [name, input] : node.inputs)
            if (const NodeRef* ref = std::get_if<NodeRef>(&input)) {
                if (!nodes_.count(ref->node)) throw NodeFailure(id, "input '" + name + "' is linked to node " + ref->node + ", which does not exist");
                const std::string& k = keyOf(ref->node);
                if (k.empty()) cacheable = false;
                upstream[name] = {k, ref->output};
            }
        return keys_[id] = cacheable ? nodeCacheKey(node, registry_, upstream) : std::string();
    }

    const NodeOutputs& evaluate(const std::string& id) {
        if (const auto done = results_.find(id); done != results_.end()) return *done->second;
        if (options_.stop.stop_requested()) throw Interrupted();
        if (!active_.insert(id).second) throw NodeFailure(id, "the graph loops back to this node");
        struct Leave {
            std::set<std::string>& active;
            const std::string& id;
            ~Leave() { active.erase(id); }
        } leave{active_, id};

        const NodeDef& def = definition(id);
        const std::string& key = keyOf(id);
        if (!key.empty()) {
            if (const Outputs hit = cache_.find(key)) {
                cachedNodes_.push_back(id);
                ++done_;
                Event e = nodeEvent(Event::Kind::NodeCached, id);
                e.summaries = summaries(*hit);
                emit(e);
                return *(results_[id] = hit);
            }
        }

        NodeOutputs outputs = runNode(id, def);
        const Outputs stored = std::make_shared<const NodeOutputs>(std::move(outputs));
        if (!key.empty()) cache_.put(key, *stored);
        ++done_;
        Event e = nodeEvent(Event::Kind::NodeFinished, id);
        e.summaries = summaries(*stored);
        e.millis = millis_[id];
        emit(e);
        return *(results_[id] = stored);
    }

    void emit(Event e) const {
        if (!options_.events) return;
        e.jobId = options_.jobId;
        options_.events->publish(e);
    }
    Event nodeEvent(Event::Kind kind, const std::string& id) const {
        Event e;
        e.kind = kind;
        e.node = id;
        e.type = nodes_.at(id).type;
        e.done = done_;
        e.total = total_;
        return e;
    }
    static std::vector<std::string> summaries(const NodeOutputs& outputs) {
        std::vector<std::string> out;
        for (const Value& v : outputs) out.push_back(v.summary());
        return out;
    }

    // The value feeding one linked input.
    Value linked(const std::string& id, const std::string& input, const NodeRef& ref) {
        const NodeOutputs& source = evaluate(ref.node);
        if (ref.output < 0 || ref.output >= static_cast<int>(source.size()))
            throw NodeFailure(id, "input '" + input + "' reads output " + std::to_string(ref.output) + " of node " + ref.node + ", which has " + std::to_string(source.size()), input);
        return source[ref.output];
    }

    NodeOutputs runNode(const std::string& id, const NodeDef& def) {
        const JobNode node = nodes_.at(id);   // copy: expansions add to nodes_
        if (!def.runFn()) throw NodeFailure(id, "node type '" + node.type + "' cannot run");
        NodeContext ctx;
        ctx.nodeId = id;
        ctx.jobId = options_.jobId;
        ctx.stop = options_.stop;
        ctx.progress = [this, id](int step, int steps) {
            Event e = nodeEvent(Event::Kind::NodeProgress, id);
            e.step = step;
            e.steps = steps;
            emit(e);
        };
        ctx.preview = [this, id](const Value& image) {
            Event e = nodeEvent(Event::Kind::NodePreview, id);
            e.preview = std::make_shared<const Value>(image);
            emit(e);
        };

        std::set<std::string> lazyPending;
        try {
            for (const InputDef& in : def.inputs()) {
                if (in.kind == InputKind::Hidden) continue;
                const auto given = node.inputs.find(in.name);
                if (given == node.inputs.end()) {
                    if (in.widget && !in.def.is_null()) ctx.inputs[in.name] = literalValue(in, in.def);
                } else if (const NodeRef* ref = std::get_if<NodeRef>(&given->second)) {
                    ctx.links[in.name] = *ref;
                    if (in.lazy) lazyPending.insert(in.name);
                    else ctx.inputs[in.name] = linked(id, in.name, *ref);
                } else {
                    ctx.inputs[in.name] = literalValue(in, std::get<Json>(given->second));
                }
            }
        } catch (const NodeFailure&) {
            throw;
        } catch (const Interrupted&) {
            throw;
        } catch (const NodeError& e) {
            throw NodeFailure(id, e.what(), e.input);
        } catch (const std::exception& e) {
            throw NodeFailure(id, e.what());
        }

        emit(nodeEvent(Event::Kind::NodeStarted, id));

        // list mapping: run once per item; lists must have one length (a single item is repeated)
        std::size_t count = 1;
        bool mapped = false;
        if (!def.takesLists()) {
            for (const auto& [name, v] : ctx.inputs)
                if (v.isList()) {
                    mapped = true;
                    count = std::max(count, v.list().size());
                }
            for (const auto& [name, v] : ctx.inputs)
                if (v.isList() && v.list().size() != count && v.list().size() != 1)
                    throw NodeFailure(id, "list inputs differ in length (" + std::to_string(count) + " and " + std::to_string(v.list().size()) + ")");
        }
        if (!mapped) return invoke(id, def, node, ctx, lazyPending);

        std::vector<NodeOutputs> perItem;
        for (std::size_t i = 0; i < count; ++i) {
            NodeContext item = ctx;
            for (auto& [name, v] : item.inputs)
                if (v.isList()) v = v.list().size() == 1 ? v.list()[0] : v.list()[i];
            perItem.push_back(invoke(id, def, node, item, lazyPending));
        }
        NodeOutputs combined;
        for (std::size_t o = 0; o < def.outputs().size(); ++o) {
            ValueList items;
            for (const auto& r : perItem) {
                if (o >= r.size()) continue;
                if (def.outputs()[o].isList && r[o].isList()) items.insert(items.end(), r[o].list().begin(), r[o].list().end());
                else items.push_back(r[o]);
            }
            combined.push_back(Value::makeList(def.outputs()[o].type, std::move(items)));
        }
        return combined;
    }

    NodeOutputs invoke(const std::string& id, const NodeDef& def, const JobNode&, NodeContext ctx, std::set<std::string> lazyPending) {
        const auto started = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration waited{};
        NodeResult res;
        for (std::size_t round = 0;; ++round) {
            if (options_.stop.stop_requested()) throw Interrupted();
            try {
                res = def.runFn()(ctx);
            } catch (const Interrupted&) {
                throw;
            } catch (const NodeFailure&) {
                throw;
            } catch (const NodeError& e) {
                throw NodeFailure(id, e.what(), e.input);
            } catch (const std::exception& e) {
                throw NodeFailure(id, e.what());
            }
            if (res.needInputs.empty()) break;
            if (round > def.inputs().size()) throw NodeFailure(id, "the node keeps asking for inputs");
            for (const auto& name : res.needInputs) {
                if (!lazyPending.count(name)) {
                    if (ctx.has(name)) continue;   // already there
                    throw NodeFailure(id, "the node asked for '" + name + "', which is not a linked lazy input");
                }
                const auto t0 = std::chrono::steady_clock::now();
                ctx.inputs[name] = linked(id, name, ctx.links.at(name));
                waited += std::chrono::steady_clock::now() - t0;
                lazyPending.erase(name);
            }
        }
        if (res.expansion) {
            NodeOutputs out = expand(id, *res.expansion);
            record(id, started, waited);
            return out;
        }
        if (res.outputs.size() != def.outputs().size())
            throw NodeFailure(id, "the node returned " + std::to_string(res.outputs.size()) + " outputs, its definition has " + std::to_string(def.outputs().size()));
        record(id, started, waited);
        return std::move(res.outputs);
    }

    void record(const std::string& id, std::chrono::steady_clock::time_point started, std::chrono::steady_clock::duration waited) {
        const auto elapsed = std::chrono::steady_clock::now() - started - waited;
        millis_[id] += std::chrono::duration<double, std::milli>(elapsed).count();
        if (std::find(ran_.begin(), ran_.end(), id) == ran_.end()) ran_.push_back(id);
    }

    NodeOutputs expand(const std::string& id, const Expansion& exp) {
        if (static_cast<int>(std::count(id.begin(), id.end(), ':')) >= kMaxExpansionDepth) throw NodeFailure(id, "nodes expand into nodes that expand, too deep");
        const std::string prefix = id + ":";
        const auto rename = [&](const NodeRef& ref) { return exp.graph.nodes.count(ref.node) ? NodeRef{prefix + ref.node, ref.output} : ref; };
        for (const auto& [innerId, inner] : exp.graph.nodes) {
            JobNode copy = inner;
            for (auto& [name, input] : copy.inputs)
                if (const NodeRef* ref = std::get_if<NodeRef>(&input)) input = rename(*ref);
            nodes_[prefix + innerId] = std::move(copy);
        }
        // the spliced nodes must be sound before anything runs
        JobGraph merged;
        merged.nodes.insert(nodes_.begin(), nodes_.end());
        for (const auto& issue : validate(merged, registry_).issues)
            if (issue.node.rfind(prefix, 0) == 0) throw NodeFailure(id, "its expansion is invalid: node " + issue.node.substr(prefix.size()) + (issue.input.empty() ? "" : ", input '" + issue.input + "'") + ": " + issue.message);
        NodeOutputs out;
        for (const NodeRef& ref : exp.outputs) {
            const NodeRef real = rename(ref);
            if (!nodes_.count(real.node)) throw NodeFailure(id, "its expansion returns node " + ref.node + ", which does not exist");
            out.push_back(linked(id, "<expansion>", real));
        }
        return out;
    }

    const Registry& registry_;
    OutputCache& cache_;
    std::map<std::string, JobNode, NodeIdLess> nodes_;
    const JobGraph& job_;
    const RunOptions& options_;
    std::map<std::string, Outputs> results_;
    std::map<std::string, std::string> keys_;
    std::set<std::string> active_;
    std::vector<std::string> ran_, cachedNodes_;
    std::map<std::string, double, NodeIdLess> millis_;
    int done_ = 0, total_ = 0;
};

}  // namespace

ExecResult Executor::run(const JobGraph& job, const RunOptions& options) {
    return Run(registry_, cache_, job, options).go();
}

}  // namespace gh::graph
