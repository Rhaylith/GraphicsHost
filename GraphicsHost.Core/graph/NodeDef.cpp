#include "graph/NodeDef.h"

#include "nodes/Builtin.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace gh::graph {

using Json = nlohmann::json;

const InputDef* NodeDef::findInput(const std::string& name) const {
    for (const auto& in : inputs_)
        if (in.name == name) return &in;
    return nullptr;
}

void Registry::add(NodeDef def) {
    if (def.id().empty()) throw std::invalid_argument("a node needs an id");
    if (nodes_.count(def.id())) throw std::invalid_argument("node id registered twice: " + def.id());
    std::set<std::string> seen;
    for (const auto& in : def.inputs()) {
        if (in.name.empty()) throw std::invalid_argument(def.id() + ": an input has no name");
        if (!seen.insert(in.name).second) throw std::invalid_argument(def.id() + ": input name used twice: " + in.name);
    }
    for (const auto& out : def.outputs())
        if (out.name.empty()) throw std::invalid_argument(def.id() + ": an output has no name");
    const std::string id = def.id();
    nodes_.emplace(id, std::move(def));
}

const NodeDef* Registry::find(const std::string& id) const {
    const auto it = nodes_.find(id);
    return it == nodes_.end() ? nullptr : &it->second;
}

std::vector<const NodeDef*> Registry::all() const {
    std::vector<const NodeDef*> list;
    for (const auto& [id, def] : nodes_) list.push_back(&def);
    std::sort(list.begin(), list.end(), [](const NodeDef* a, const NodeDef* b) {
        if (a->categoryName() != b->categoryName()) return a->categoryName() < b->categoryName();
        return a->displayName() < b->displayName();
    });
    return list;
}

namespace {
Json inputJson(const InputDef& in) {
    Json j = {{"name", in.name}, {"type", typeId(in.type)}, {"tooltip", in.tip}};
    j["kind"] = in.kind == InputKind::Required ? "required" : in.kind == InputKind::Optional ? "optional" : "hidden";
    j["widget"] = in.widget;
    if (!in.def.is_null()) j["default"] = in.def;
    if (in.min) j["min"] = *in.min;
    if (in.max) j["max"] = *in.max;
    if (in.step) j["step"] = *in.step;
    if (in.round) j["round"] = *in.round;
    if (in.widget && (in.type == ValueType::Int || in.type == ValueType::Float)) j["display"] = in.display;
    if (in.seed) j["seedControl"] = true;
    if (in.multiline) j["multiline"] = true;
    if (in.dynamicPrompts) j["dynamicPrompts"] = true;
    if (!in.options.empty()) j["options"] = in.options;
    if (!in.optionTips.empty()) j["optionTips"] = in.optionTips;
    if (!in.modelFolder.empty()) j["modelFolder"] = in.modelFolder;
    if (in.lazy) j["lazy"] = true;
    return j;
}
}  // namespace

Json Registry::catalogue() const {
    Json types = Json::array();
    for (const ValueType t : allValueTypes()) types.push_back({{"id", typeId(t)}, {"color", typeColor(t)}});
    Json nodes = Json::array();
    for (const NodeDef* def : all()) {
        Json inputs = Json::array(), outputs = Json::array();
        for (const auto& in : def->inputs()) inputs.push_back(inputJson(in));
        for (const auto& out : def->outputs()) outputs.push_back({{"name", out.name}, {"type", typeId(out.type)}, {"tooltip", out.tip}, {"list", out.isList}});
        Json node = {{"id", def->id()},           {"display", def->displayName()}, {"category", def->categoryName()},
                     {"description", def->descriptionText()}, {"inputs", inputs}, {"outputs", outputs}, {"outputNode", def->outputNode()}};
        if (def->takesLists()) node["acceptsLists"] = true;
        if (def->canExpand()) node["expands"] = true;
        if (def->isDeprecated()) node["deprecated"] = true;
        if (def->isExperimental()) node["experimental"] = true;
        if (!def->aliases().empty()) node["searchAliases"] = def->aliases();
        nodes.push_back(std::move(node));
    }
    return {{"version", 1}, {"types", types}, {"nodes", nodes}};
}

const Registry& Registry::builtin() {
    static const Registry registry = [] {
        Registry r;
        nodes::registerBuiltinNodes(r);
        return r;
    }();
    return registry;
}

}  // namespace gh::graph
