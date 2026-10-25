#pragma once

#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace gh::graph {

// Node ids are strings ("4", "12", and "7:3" or "7.2" inside expanded subgraphs). They sort the way people read them:
// digit runs compare as numbers, so "2" comes before "10".
struct NodeIdLess {
    bool operator()(const std::string& a, const std::string& b) const;
};

// A link: the output `output` of node `node`.
struct NodeRef {
    std::string node;
    int output = 0;
    bool operator==(const NodeRef&) const = default;
};

// An input of a job node: a literal value (a JSON number, string or boolean) or a link.
using JobInput = std::variant<nlohmann::json, NodeRef>;

struct JobNode {
    std::string type;
    std::map<std::string, JobInput> inputs;
};

// The job format (workflow plan, section 4.4): what the engine runs, what the API accepts, and what is embedded in
// outputs. All widgets are plain inputs, all reroutes, constants, muted and bypassed nodes and subgraphs are gone.
struct JobGraph {
    std::map<std::string, JobNode, NodeIdLess> nodes;
    std::vector<std::string> outputs;   // empty: every output node
    std::string clientId;
    std::string clientLabel;
};

inline constexpr const char* kJobFormat = "graphicshost.job";
inline constexpr int kJobVersion = 1;

nlohmann::json toJson(const JobGraph& graph);
// Throws std::runtime_error with a message for people when the document is not a job.
JobGraph jobFromJson(const nlohmann::json& document);

}  // namespace gh::graph
