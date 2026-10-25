#include "graph/JobGraph.h"

#include <cctype>
#include <stdexcept>

namespace gh::graph {

using Json = nlohmann::json;

bool NodeIdLess::operator()(const std::string& a, const std::string& b) const {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (std::isdigit(static_cast<unsigned char>(a[i])) && std::isdigit(static_cast<unsigned char>(b[j]))) {
            size_t ei = i, ej = j;
            while (ei < a.size() && std::isdigit(static_cast<unsigned char>(a[ei]))) ++ei;
            while (ej < b.size() && std::isdigit(static_cast<unsigned char>(b[ej]))) ++ej;
            // compare the digit runs as numbers: strip zeros, then length, then text
            size_t si = i, sj = j;
            while (si + 1 < ei && a[si] == '0') ++si;
            while (sj + 1 < ej && b[sj] == '0') ++sj;
            const size_t li = ei - si, lj = ej - sj;
            if (li != lj) return li < lj;
            const int c = a.compare(si, li, b, sj, lj);
            if (c != 0) return c < 0;
            i = ei;
            j = ej;
        } else {
            if (a[i] != b[j]) return a[i] < b[j];
            ++i;
            ++j;
        }
    }
    return (a.size() - i) < (b.size() - j);
}

Json toJson(const JobGraph& g) {
    Json nodes = Json::object();
    for (const auto& [id, node] : g.nodes) {
        Json inputs = Json::object();
        for (const auto& [name, in] : node.inputs) {
            if (const auto* ref = std::get_if<NodeRef>(&in)) inputs[name] = Json::array({ref->node, ref->output});
            else inputs[name] = std::get<Json>(in);
        }
        nodes[id] = {{"type", node.type}, {"inputs", inputs}};
    }
    Json doc = {{"format", kJobFormat}, {"version", kJobVersion}, {"nodes", nodes}};
    if (!g.outputs.empty()) doc["outputs"] = g.outputs;
    if (!g.clientId.empty() || !g.clientLabel.empty()) doc["client"] = {{"id", g.clientId}, {"label", g.clientLabel}};
    return doc;
}

JobGraph jobFromJson(const Json& doc) {
    if (!doc.is_object()) throw std::runtime_error("a job is a JSON object");
    if (doc.value("format", std::string()) != kJobFormat) throw std::runtime_error("this is not a GraphicsHost job (format must be \"graphicshost.job\")");
    const int version = doc.value("version", 0);
    if (version < 1 || version > kJobVersion) throw std::runtime_error("job version " + std::to_string(version) + " is not supported (this program reads up to " + std::to_string(kJobVersion) + ")");
    if (!doc.contains("nodes") || !doc["nodes"].is_object()) throw std::runtime_error("the job has no \"nodes\" object");
    JobGraph g;
    for (const auto& [id, node] : doc["nodes"].items()) {
        if (!node.is_object() || !node.contains("type") || !node["type"].is_string()) throw std::runtime_error("node " + id + " has no \"type\"");
        JobNode n;
        n.type = node["type"].get<std::string>();
        if (node.contains("inputs")) {
            if (!node["inputs"].is_object()) throw std::runtime_error("node " + id + ": \"inputs\" must be an object");
            for (const auto& [name, value] : node["inputs"].items()) {
                // a link is [nodeId, outputIndex]; the values of widgets are never lists, so a list of this shape is a link
                if (value.is_array() && value.size() == 2 && (value[0].is_string() || value[0].is_number_integer()) && value[1].is_number_integer()) {
                    n.inputs[name] = NodeRef{value[0].is_string() ? value[0].get<std::string>() : std::to_string(value[0].get<int64_t>()), value[1].get<int>()};
                } else {
                    n.inputs[name] = value;
                }
            }
        }
        g.nodes[id] = std::move(n);
    }
    if (doc.contains("outputs")) {
        if (!doc["outputs"].is_array()) throw std::runtime_error("\"outputs\" must be a list of node ids");
        for (const auto& o : doc["outputs"]) g.outputs.push_back(o.is_string() ? o.get<std::string>() : std::to_string(o.get<int64_t>()));
    }
    if (doc.contains("client") && doc["client"].is_object()) {
        g.clientId = doc["client"].value("id", std::string());
        g.clientLabel = doc["client"].value("label", std::string());
    }
    return g;
}

}  // namespace gh::graph
