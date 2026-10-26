#include "graph/Validator.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace gh::graph {

using Json = nlohmann::json;

Json ValidationResult::toJson() const {
    Json list = Json::array();
    for (const auto& i : issues) list.push_back({{"node", i.node}, {"input", i.input}, {"code", i.code}, {"message", i.message}, {"details", i.details}});
    return {{"ok", ok()}, {"issues", list}};
}

namespace {

std::string number(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.10g", v);
    return buf;
}

class Checker {
public:
    Checker(const JobGraph& job, const Registry& registry, const ValidateOptions& options) : job_(job), registry_(registry), options_(options) {}

    ValidationResult run() {
        for (const auto& [id, node] : job_.nodes) checkNode(id, node);
        checkCycles();
        for (const auto& id : job_.outputs)
            if (!job_.nodes.count(id)) add(id, "", "unknown_output", "The job asks for the output of node " + id + ", which does not exist.");
        if (job_.outputs.empty()) {
            const bool any = std::any_of(job_.nodes.begin(), job_.nodes.end(), [&](const auto& n) {
                const NodeDef* def = registry_.find(n.second.type);
                return def && def->outputNode();
            });
            if (!any && !job_.nodes.empty()) add("", "", "no_outputs", "Nothing to run: the workflow has no output node (for example Save Image or Preview Image).");
        }
        return std::move(result_);
    }

private:
    void add(const std::string& node, const std::string& input, const char* code, std::string message, Json details = Json::object()) {
        result_.issues.push_back({node, input, code, std::move(message), std::move(details)});
    }

    void checkNode(const std::string& id, const JobNode& node) {
        const NodeDef* def = registry_.find(node.type);
        if (!def) {
            add(id, "", "unknown_type", "Unknown node type '" + node.type + "'. A plugin may be missing.", {{"type", node.type}});
            return;
        }
        for (const auto& [name, value] : node.inputs)
            if (!def->findInput(name)) add(id, name, "unknown_input", node.type + " has no input '" + name + "'.");

        std::map<std::string, Json> literals;
        for (const InputDef& in : def->inputs()) {
            if (in.kind == InputKind::Hidden) continue;
            const auto it = node.inputs.find(in.name);
            if (it == node.inputs.end()) {
                // a widget with a default needs no value; a required socket needs a link
                if (in.kind == InputKind::Required && (!in.widget || in.def.is_null()))
                    add(id, in.name, "missing_input", "'" + in.name + "' needs " + (in.widget ? "a value or a link." : "a link."));
                else if (in.widget && !in.def.is_null()) literals[in.name] = in.def;
                continue;
            }
            if (const NodeRef* ref = std::get_if<NodeRef>(&it->second)) checkLink(id, in, *ref);
            else {
                const Json& v = std::get<Json>(it->second);
                if (checkValue(id, in, v)) literals[in.name] = v;
            }
        }
        if (def->validateFn())
            for (const auto& message : def->validateFn()(literals)) add(id, "", "custom", message);
    }

    void checkLink(const std::string& id, const InputDef& in, const NodeRef& ref) {
        const auto src = job_.nodes.find(ref.node);
        if (src == job_.nodes.end()) {
            add(id, in.name, "bad_link", "'" + in.name + "' is linked to node " + ref.node + ", which does not exist.");
            return;
        }
        const NodeDef* srcDef = registry_.find(src->second.type);
        if (!srcDef) return;   // reported on that node
        if (ref.output < 0 || ref.output >= static_cast<int>(srcDef->outputs().size())) {
            add(id, in.name, "bad_link", "'" + in.name + "' is linked to output " + std::to_string(ref.output) + " of " + srcDef->id() + ", which has " + std::to_string(srcDef->outputs().size()) + ".");
            return;
        }
        const OutputDef& out = srcDef->outputs()[ref.output];
        if (!linkCompatible(out.type, in.type))
            add(id, in.name, "type_mismatch", "'" + in.name + "' takes " + typeId(in.type) + " but is linked to " + out.name + " (" + typeId(out.type) + ") of node " + ref.node + ".",
                {{"expected", typeId(in.type)}, {"got", typeId(out.type)}});
    }

    // Returns true when the literal is acceptable.
    bool checkValue(const std::string& id, const InputDef& in, const Json& v) {
        if (!in.widget) {
            add(id, in.name, "bad_value", "'" + in.name + "' takes a link, not a typed value.");
            return false;
        }
        switch (in.type) {
            case ValueType::Int:
            case ValueType::Float: return checkNumber(id, in, v);
            case ValueType::Boolean:
                if (!v.is_boolean()) return fail(id, in, "must be true or false.");
                return true;
            case ValueType::String:
                if (!v.is_string()) return fail(id, in, "must be text.");
                return true;
            case ValueType::Combo: return checkCombo(id, in, v);
            default: return true;
        }
    }

    bool fail(const std::string& id, const InputDef& in, const std::string& what) {
        add(id, in.name, "bad_value", "'" + in.name + "' " + what);
        return false;
    }

    bool checkNumber(const std::string& id, const InputDef& in, const Json& v) {
        if (!v.is_number()) return fail(id, in, "must be a number.");
        double x = v.get<double>();
        if (in.type == ValueType::Int && x != std::floor(x)) return fail(id, in, "must be a whole number.");
        const double step = in.step.value_or(0.0);
        if (step > 0) x = std::round((x - in.min.value_or(0.0)) / step) * step + in.min.value_or(0.0);
        const bool low = in.min && x < *in.min - 1e-9, high = in.max && x > *in.max + 1e-9;
        if (low || high) {
            Json details = Json::object();
            if (in.min) details["min"] = *in.min;
            if (in.max) details["max"] = *in.max;
            std::string range = in.min && in.max ? "between " + number(*in.min) + " and " + number(*in.max) : in.min ? "at least " + number(*in.min) : "at most " + number(*in.max);
            add(id, in.name, "out_of_range", "'" + in.name + "' must be " + range + " (got " + number(x) + ").", details);
            return false;
        }
        return true;
    }

    bool checkCombo(const std::string& id, const InputDef& in, const Json& v) {
        if (!v.is_string()) return fail(id, in, "must be text.");
        const std::string value = v.get<std::string>();
        if (!in.options.empty() && std::find(in.options.begin(), in.options.end(), value) == in.options.end()) {
            add(id, in.name, "unknown_option", "'" + value + "' is not one of the choices for '" + in.name + "'.", {{"options", in.options}, {"value", value}});
            return false;
        }
        if (value.empty() && in.kind == InputKind::Optional) return true;   // an optional model left empty
        if (!in.modelFolder.empty() && options_.modelExists && !options_.modelExists(in.modelFolder, value)) {
            add(id, in.name, "missing_model", "The model '" + value + "' was not found in the " + in.modelFolder + " folder.", {{"folder", in.modelFolder}, {"name", value}});
            return false;
        }
        return true;
    }

    // Depth-first search over the links (with an explicit stack, so long chains are fine); every node on a cycle is
    // reported once.
    void checkCycles() {
        enum State { Unseen, Open, Done };
        struct Frame {
            std::string id;
            std::map<std::string, JobInput>::const_iterator next, end;
        };
        std::map<std::string, State> state;
        std::set<std::string> reported;
        std::vector<Frame> stack;
        const auto enter = [&](const std::string& id) {
            state[id] = Open;
            const JobNode& node = job_.nodes.at(id);
            stack.push_back({id, node.inputs.begin(), node.inputs.end()});
        };
        for (const auto& [rootId, rootNode] : job_.nodes) {
            if (state[rootId] != Unseen) continue;
            enter(rootId);
            while (!stack.empty()) {
                Frame& frame = stack.back();
                if (frame.next == frame.end) {
                    state[frame.id] = Done;
                    stack.pop_back();
                    continue;
                }
                const NodeRef* ref = std::get_if<NodeRef>(&(frame.next++)->second);
                if (!ref || !job_.nodes.count(ref->node)) continue;
                if (state[ref->node] == Open) {
                    bool inLoop = false;
                    for (const Frame& f : stack) {
                        inLoop = inLoop || f.id == ref->node;
                        if (inLoop && reported.insert(f.id).second) add(f.id, "", "cycle", "Node " + f.id + " is part of a loop: its result depends on itself.");
                    }
                } else if (state[ref->node] == Unseen) {
                    enter(ref->node);
                }
            }
        }
    }

    const JobGraph& job_;
    const Registry& registry_;
    const ValidateOptions& options_;
    ValidationResult result_;
};

}  // namespace

ValidationResult validate(const JobGraph& job, const Registry& registry, const ValidateOptions& options) {
    return Checker(job, registry, options).run();
}

}  // namespace gh::graph
