#include "graph/Workflow.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>

namespace gh::graph {

using Json = nlohmann::json;

namespace {

[[noreturn]] void bad(const std::string& message) { throw std::runtime_error("workflow: " + message); }

std::string str(const Json& j, const char* key, const std::string& fallback = "") {
    const auto it = j.find(key);
    return it != j.end() && it->is_string() ? it->get<std::string>() : fallback;
}
int integer(const Json& j, const char* key, int fallback = 0) {
    const auto it = j.find(key);
    return it != j.end() && it->is_number() ? it->get<int>() : fallback;
}
bool boolean(const Json& j, const char* key, bool fallback = false) {
    const auto it = j.find(key);
    return it != j.end() && it->is_boolean() ? it->get<bool>() : fallback;
}
template <size_t N>
std::array<double, N> numbers(const Json& j, const char* key) {
    std::array<double, N> out{};
    const auto it = j.find(key);
    if (it != j.end() && it->is_array())
        for (size_t i = 0; i < N && i < it->size(); ++i)
            if ((*it)[i].is_number()) out[i] = (*it)[i].get<double>();
    return out;
}

Json unknownKeys(const Json& object, std::initializer_list<const char*> known) {
    Json rest = Json::object();
    for (const auto& [key, value] : object.items())
        if (std::find_if(known.begin(), known.end(), [&](const char* k) { return key == k; }) == known.end()) rest[key] = value;
    return rest;
}

void keepUnknown(Json& out, const Json& unknown) {
    for (const auto& [key, value] : unknown.items()) out[key] = value;
}

const char* modeName(NodeMode m) {
    switch (m) {
        case NodeMode::Muted: return "muted";
        case NodeMode::Bypassed: return "bypassed";
        default: return "active";
    }
}

NodeMode parseMode(const std::string& s) {
    if (s.empty() || s == "active") return NodeMode::Active;
    if (s == "muted") return NodeMode::Muted;
    if (s == "bypassed") return NodeMode::Bypassed;
    bad("unknown node mode '" + s + "'");
}

Json nodeJson(const WNode& n) {
    Json j = Json::object();
    keepUnknown(j, n.unknown);
    j["id"] = n.id;
    j["type"] = n.type;
    if (!n.title.empty()) j["title"] = n.title;
    j["pos"] = n.pos;
    j["size"] = n.size;
    j["mode"] = modeName(n.mode);
    j["color"] = n.color.empty() ? Json(nullptr) : Json(n.color);
    j["collapsed"] = n.collapsed;
    j["pinned"] = n.pinned;
    j["widgets"] = n.widgets;
    j["inputsAsSockets"] = n.inputsAsSockets;
    j["order"] = n.order;
    return j;
}

WNode nodeFromJson(const Json& j) {
    if (!j.is_object()) bad("a node is not an object");
    WNode n;
    n.id = integer(j, "id");
    n.type = str(j, "type");
    if (n.type.empty()) bad("node " + std::to_string(n.id) + " has no type");
    n.title = str(j, "title");
    n.pos = numbers<2>(j, "pos");
    n.size = numbers<2>(j, "size");
    n.mode = parseMode(str(j, "mode"));
    n.color = str(j, "color");
    n.collapsed = boolean(j, "collapsed");
    n.pinned = boolean(j, "pinned");
    if (const auto it = j.find("widgets"); it != j.end()) {
        if (!it->is_object()) bad("node " + std::to_string(n.id) + ": widgets must be an object");
        n.widgets = *it;
    }
    if (const auto it = j.find("inputsAsSockets"); it != j.end() && it->is_array())
        for (const auto& s : *it)
            if (s.is_string()) n.inputsAsSockets.push_back(s.get<std::string>());
    n.order = integer(j, "order");
    n.unknown = unknownKeys(j, {"id", "type", "title", "pos", "size", "mode", "color", "collapsed", "pinned", "widgets", "inputsAsSockets", "order"});
    return n;
}

Json linkJson(const WLink& l) {
    return {{"id", l.id}, {"from", Json::array({l.fromNode, l.fromOutput})}, {"to", Json::array({l.toNode, l.toInput})}, {"type", l.type}};
}

WLink linkFromJson(const Json& j) {
    const auto from = j.find("from");
    const auto to = j.find("to");
    if (from == j.end() || !from->is_array() || from->size() != 2 || !(*from)[0].is_number() || !(*from)[1].is_number())
        bad("a link needs \"from\": [node, output]");
    if (to == j.end() || !to->is_array() || to->size() != 2 || !(*to)[0].is_number() || !(*to)[1].is_string())
        bad("a link needs \"to\": [node, input name]");
    WLink l;
    l.id = integer(j, "id");
    l.fromNode = (*from)[0].get<int>();
    l.fromOutput = (*from)[1].get<int>();
    l.toNode = (*to)[0].get<int>();
    l.toInput = (*to)[1].get<std::string>();
    l.type = str(j, "type");
    return l;
}

void bodyJson(const WBody& body, Json& out) {
    Json nodes = Json::array(), links = Json::array();
    for (const auto& n : body.nodes) nodes.push_back(nodeJson(n));
    for (const auto& l : body.links) links.push_back(linkJson(l));
    out["nodes"] = std::move(nodes);
    out["links"] = std::move(links);
}

WBody bodyFromJson(const Json& j) {
    WBody body;
    if (const auto it = j.find("nodes"); it != j.end()) {
        if (!it->is_array()) bad("\"nodes\" must be a list");
        for (const auto& n : *it) body.nodes.push_back(nodeFromJson(n));
    }
    if (const auto it = j.find("links"); it != j.end()) {
        if (!it->is_array()) bad("\"links\" must be a list");
        for (const auto& l : *it) body.links.push_back(linkFromJson(l));
    }
    return body;
}

Json portsJson(const std::vector<WPort>& ports) {
    Json list = Json::array();
    for (const auto& p : ports) list.push_back({{"name", p.name}, {"type", p.type}});
    return list;
}

std::vector<WPort> portsFromJson(const Json& j, const char* key) {
    std::vector<WPort> ports;
    if (const auto it = j.find(key); it != j.end() && it->is_array())
        for (const auto& p : *it) ports.push_back({str(p, "name"), str(p, "type", "*")});
    return ports;
}

// ---- migrations: one function per version step, applied in order -------------------------------------------------

void migrateNodeList(Json& nodes) {
    if (!nodes.is_array()) return;
    for (auto& n : nodes) {
        if (!n.is_object()) continue;
        if (!n.contains("mode")) {
            if (boolean(n, "bypassed")) n["mode"] = "bypassed";
            else if (boolean(n, "muted")) n["mode"] = "muted";
        }
        n.erase("bypassed");
        n.erase("muted");
    }
}

// Version 0 (before the format was frozen) had no format/version keys and flagged nodes with "muted"/"bypassed" booleans.
void migrate0to1(Json& d) {
    d["format"] = kWorkflowFormat;
    d["version"] = 1;
    if (d.contains("nodes")) migrateNodeList(d["nodes"]);
    if (const auto it = d.find("subgraphs"); it != d.end() && it->is_array())
        for (auto& sg : *it)
            if (sg.is_object() && sg.contains("nodes")) migrateNodeList(sg["nodes"]);
}

using Migration = void (*)(Json&);
constexpr Migration kMigrations[] = {migrate0to1};   // kMigrations[v] upgrades version v to v + 1
static_assert(sizeof(kMigrations) / sizeof(kMigrations[0]) == kWorkflowVersion);

// ---- conversion ---------------------------------------------------------------------------------------------------

struct Port {
    std::string name;
    ValueType type = ValueType::Any;
};

struct Source {
    std::string node;
    int output = 0;
};

struct Pass {
    std::string node;   // the output is whatever feeds this input of this node
    std::string input;
};

struct FlatNode {
    std::string id;
    std::string type;
    Json widgets = Json::object();
    NodeMode mode = NodeMode::Active;
    std::vector<Port> ins, outs;   // from the registry or the subgraph definition; used to rewire bypassed nodes
    std::map<int, Pass> pass;      // pass-through outputs (reroutes, subgraph boundaries)
    bool constant = false;         // a primitive: its output is its "value" widget
    bool known = true;             // the registry knows the type
};

struct Flat {
    std::map<std::string, FlatNode, NodeIdLess> nodes;
    std::map<std::pair<std::string, std::string>, Source> links;   // (target node, input) -> source
};

constexpr const char* kRerouteType = "util.reroute";
constexpr const char* kPrimitiveType = "util.primitive";

ValueType portType(const std::string& id) {
    ValueType t = ValueType::Any;
    parseTypeId(id, t);
    return t;
}

class Converter {
public:
    Converter(const Workflow& wf, const Registry& registry) : wf_(wf), registry_(registry) {}

    JobGraph run() {
        flatten(wf_.body, "", {});
        JobGraph job;
        for (const auto& [id, node] : flat_.nodes) {
            if (node.mode != NodeMode::Active || node.constant || !node.pass.empty() || node.type[0] == '$' || node.type.rfind(kSubgraphPrefix, 0) == 0) continue;
            JobNode out;
            out.type = node.type;
            const NodeDef* def = node.known ? registry_.find(node.type) : nullptr;
            for (const auto& [name, value] : node.widgets.items()) {
                if (def && !def->findInput(name)) continue;   // editor-only widgets such as the seed control
                out.inputs[name] = value;
            }
            for (const auto& [key, source] : flat_.links) {
                if (key.first != id) continue;
                const Resolved r = resolve(source.node, source.output, 0);
                if (r.kind == Resolved::Link) out.inputs[key.second] = NodeRef{r.node, r.output};
                else if (r.kind == Resolved::Literal) out.inputs[key.second] = r.literal;
                else out.inputs.erase(key.second);   // fed by a muted node: the input is empty
            }
            job.nodes.emplace(id, std::move(out));
        }
        job.clientId = wf_.id;
        return job;
    }

private:
    struct Resolved {
        enum Kind { None, Link, Literal } kind = None;
        std::string node;
        int output = 0;
        Json literal;
    };

    const WSubgraph* subgraph(const std::string& id) const {
        for (const auto& sg : wf_.subgraphs)
            if (sg.id == id) return &sg;
        return nullptr;
    }

    void flatten(const WBody& body, const std::string& prefix, std::vector<std::string> open) {
        const auto idOf = [&](int id) { return prefix + std::to_string(id); };
        for (const WNode& n : body.nodes) {
            FlatNode f;
            f.id = idOf(n.id);
            f.type = n.type;
            f.widgets = n.widgets;
            f.mode = n.mode;
            if (n.type.rfind(kSubgraphPrefix, 0) == 0) {
                const std::string sgId = n.type.substr(std::string(kSubgraphPrefix).size());
                const WSubgraph* sg = subgraph(sgId);
                if (!sg) throw ConvertError("node " + f.id + " uses subgraph '" + sgId + "', which is not in the workflow");
                if (std::find(open.begin(), open.end(), sgId) != open.end()) throw ConvertError("subgraph '" + sg->name + "' contains itself");
                for (const auto& p : sg->inputs) f.ins.push_back({p.name, portType(p.type)});
                for (const auto& p : sg->outputs) f.outs.push_back({p.name, portType(p.type)});
                // outputs read what the inner output boundary receives; inner nodes read from the input boundary
                FlatNode in, out;
                in.id = f.id + ":in";
                in.type = "$subgraph-in";
                out.id = f.id + ":out";
                out.type = "$subgraph-out";
                for (size_t i = 0; i < sg->inputs.size(); ++i) in.pass[static_cast<int>(i)] = {f.id, sg->inputs[i].name};
                for (size_t i = 0; i < sg->outputs.size(); ++i) f.pass[static_cast<int>(i)] = {out.id, sg->outputs[i].name};
                flat_.nodes.emplace(in.id, std::move(in));
                flat_.nodes.emplace(out.id, std::move(out));
                auto deeper = open;
                deeper.push_back(sgId);
                flatten(sg->body, f.id + ":", deeper);
                innerLinks(*sg, f.id);
            } else if (n.type == kRerouteType) {
                f.known = false;
                f.ins = {{"value", ValueType::Any}};
                f.outs = {{"value", ValueType::Any}};
                f.pass[0] = {f.id, "value"};
            } else if (n.type == kPrimitiveType) {
                f.known = false;
                f.constant = true;
                f.outs = {{"value", ValueType::Any}};
            } else if (const NodeDef* def = registry_.find(n.type)) {
                for (const auto& in : def->inputs()) f.ins.push_back({in.name, in.type});
                for (const auto& o : def->outputs()) f.outs.push_back({o.name, o.type});
            } else {
                f.known = false;
            }
            flat_.nodes.emplace(f.id, std::move(f));
        }
        for (const WLink& l : body.links) {
            if (l.fromNode == kSubgraphInput || l.fromNode == kSubgraphOutput || l.toNode == kSubgraphInput || l.toNode == kSubgraphOutput) continue;   // see innerLinks
            flat_.links[{idOf(l.toNode), l.toInput}] = {idOf(l.fromNode), l.fromOutput};
        }
    }

    // Links of a subgraph body that touch its boundary, attached to the instance's pseudo nodes.
    void innerLinks(const WSubgraph& sg, const std::string& instance) {
        const std::string prefix = instance + ":";
        for (const WLink& l : sg.body.links) {
            const bool fromBoundary = l.fromNode == kSubgraphInput;
            const bool toBoundary = l.toNode == kSubgraphOutput;
            if (!fromBoundary && !toBoundary) continue;
            const Source from = fromBoundary ? Source{instance + ":in", l.fromOutput} : Source{prefix + std::to_string(l.fromNode), l.fromOutput};
            const std::string to = toBoundary ? instance + ":out" : prefix + std::to_string(l.toNode);
            flat_.links[{to, l.toInput}] = from;
        }
    }

    Resolved inputValue(const std::string& nodeId, const std::string& input, int depth) const {
        const auto link = flat_.links.find({nodeId, input});
        if (link != flat_.links.end()) return resolve(link->second.node, link->second.output, depth + 1);
        const auto node = flat_.nodes.find(nodeId);
        if (node != flat_.nodes.end() && node->second.widgets.contains(input)) {
            Resolved r;
            r.kind = Resolved::Literal;
            r.literal = node->second.widgets[input];
            return r;
        }
        return {};
    }

    Resolved resolve(const std::string& nodeId, int output, int depth) const {
        if (depth > 256) throw ConvertError("links loop through reroutes or subgraph boundaries at node " + nodeId);
        const auto it = flat_.nodes.find(nodeId);
        if (it == flat_.nodes.end()) return {};
        const FlatNode& n = it->second;
        if (n.mode == NodeMode::Muted) return {};
        if (n.mode == NodeMode::Bypassed) {
            const ValueType want = output < static_cast<int>(n.outs.size()) ? n.outs[output].type : ValueType::Any;
            for (const Port& in : n.ins) {
                if (!linkCompatible(in.type, want) || !flat_.links.count({nodeId, in.name})) continue;
                return inputValue(nodeId, in.name, depth);
            }
            throw ConvertError("node " + nodeId + " (" + n.type + ") is bypassed but has no linked input of type " + typeId(want));
        }
        if (n.constant) {
            Resolved r;
            if (n.widgets.contains("value")) {
                r.kind = Resolved::Literal;
                r.literal = n.widgets["value"];
            }
            return r;
        }
        if (const auto p = n.pass.find(output); p != n.pass.end()) return inputValue(p->second.node, p->second.input, depth);
        Resolved r;
        r.kind = Resolved::Link;
        r.node = nodeId;
        r.output = output;
        return r;
    }

    const Workflow& wf_;
    const Registry& registry_;
    Flat flat_;
};

}  // namespace

Json toJson(const Workflow& wf) {
    Json j = Json::object();
    keepUnknown(j, wf.unknown);
    j["format"] = kWorkflowFormat;
    j["version"] = kWorkflowVersion;
    j["id"] = wf.id;
    j["revision"] = wf.revision;
    bodyJson(wf.body, j);
    Json reroutes = Json::array(), groups = Json::array(), subgraphs = Json::array();
    for (const auto& r : wf.reroutes) reroutes.push_back({{"id", r.id}, {"pos", r.pos}, {"link", r.link}});
    for (const auto& g : wf.groups) groups.push_back({{"id", g.id}, {"title", g.title}, {"bounds", g.bounds}, {"color", g.color}});
    for (const auto& sg : wf.subgraphs) {
        Json s = Json::object();
        keepUnknown(s, sg.unknown);
        s["id"] = sg.id;
        s["name"] = sg.name;
        s["inputs"] = portsJson(sg.inputs);
        s["outputs"] = portsJson(sg.outputs);
        bodyJson(sg.body, s);
        subgraphs.push_back(std::move(s));
    }
    j["reroutes"] = std::move(reroutes);
    j["groups"] = std::move(groups);
    j["subgraphs"] = std::move(subgraphs);
    j["view"] = wf.view;
    j["extra"] = wf.extra;
    return j;
}

Json migrateWorkflow(Json d) {
    if (!d.is_object()) bad("not a JSON object");
    if (d.contains("format") && d["format"] != kWorkflowFormat) bad("the format is '" + str(d, "format") + "', not " + kWorkflowFormat);
    const int version = d.contains("version") && d["version"].is_number() ? d["version"].get<int>() : 0;
    if (version > kWorkflowVersion) bad("version " + std::to_string(version) + " is newer than this program understands (" + std::to_string(kWorkflowVersion) + ")");
    if (version < 0) bad("negative version");
    for (int v = version; v < kWorkflowVersion; ++v) kMigrations[v](d);
    return d;
}

Workflow workflowFromJson(const Json& document) {
    const Json d = migrateWorkflow(document);
    Workflow wf;
    wf.id = str(d, "id");
    wf.revision = integer(d, "revision");
    wf.body = bodyFromJson(d);
    if (const auto it = d.find("reroutes"); it != d.end() && it->is_array())
        for (const auto& r : *it) wf.reroutes.push_back({integer(r, "id"), numbers<2>(r, "pos"), integer(r, "link")});
    if (const auto it = d.find("groups"); it != d.end() && it->is_array())
        for (const auto& g : *it) wf.groups.push_back({integer(g, "id"), str(g, "title"), numbers<4>(g, "bounds"), str(g, "color")});
    if (const auto it = d.find("subgraphs"); it != d.end() && it->is_array()) {
        for (const auto& s : *it) {
            WSubgraph sg;
            sg.id = str(s, "id");
            if (sg.id.empty()) bad("a subgraph has no id");
            sg.name = str(s, "name");
            sg.inputs = portsFromJson(s, "inputs");
            sg.outputs = portsFromJson(s, "outputs");
            sg.body = bodyFromJson(s);
            sg.unknown = unknownKeys(s, {"id", "name", "inputs", "outputs", "nodes", "links"});
            wf.subgraphs.push_back(std::move(sg));
        }
    }
    if (const auto it = d.find("view"); it != d.end() && it->is_object()) wf.view = *it;
    if (const auto it = d.find("extra"); it != d.end() && it->is_object()) wf.extra = *it;
    wf.unknown = unknownKeys(d, {"format", "version", "id", "revision", "nodes", "links", "reroutes", "groups", "subgraphs", "view", "extra"});
    return wf;
}

JobGraph toJob(const Workflow& workflow, const Registry& registry) {
    return Converter(workflow, registry).run();
}

}  // namespace gh::graph
