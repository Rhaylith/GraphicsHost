#pragma once

#include "graph/JobGraph.h"
#include "graph/NodeDef.h"

#include <nlohmann/json.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace gh::graph {

// The editor format (workflow plan, section 4.3): everything the canvas needs to come back exactly as it was left.
// Unknown keys are kept (plugins put their data in "extra"), widget values are stored by name.

enum class NodeMode { Active, Muted, Bypassed };

struct WNode {
    int id = 0;
    std::string type;
    std::string title;
    std::array<double, 2> pos{0, 0};
    std::array<double, 2> size{0, 0};
    NodeMode mode = NodeMode::Active;
    std::string color;
    bool collapsed = false;
    bool pinned = false;
    nlohmann::json widgets = nlohmann::json::object();
    std::vector<std::string> inputsAsSockets;   // widgets shown as sockets
    int order = 0;
    nlohmann::json unknown = nlohmann::json::object();   // keys this version does not know, written back unchanged
};

struct WLink {
    int id = 0;
    int fromNode = 0;
    int fromOutput = 0;
    int toNode = 0;
    std::string toInput;
    std::string type;
};

struct WReroute {
    int id = 0;
    std::array<double, 2> pos{0, 0};
    int link = 0;
};

struct WGroup {
    int id = 0;
    std::string title;
    std::array<double, 4> bounds{0, 0, 0, 0};
    std::string color;
};

struct WBody {
    std::vector<WNode> nodes;
    std::vector<WLink> links;
};

struct WPort {
    std::string name;
    std::string type;   // a ValueType id such as "MODEL"
};

// Inside a subgraph a link may start at the pseudo node kSubgraphInput (its output index is the subgraph input's index)
// and end at the pseudo node kSubgraphOutput (its input name is the subgraph output's name).
inline constexpr int kSubgraphInput = -10;
inline constexpr int kSubgraphOutput = -20;
// A node of type "subgraph:<id>" is an instance of the subgraph with that id. Its inputs and outputs are the subgraph's.
inline constexpr const char* kSubgraphPrefix = "subgraph:";

struct WSubgraph {
    std::string id;
    std::string name;
    std::vector<WPort> inputs;
    std::vector<WPort> outputs;
    WBody body;
    nlohmann::json unknown = nlohmann::json::object();
};

struct Workflow {
    std::string id;
    int revision = 0;
    WBody body;
    std::vector<WReroute> reroutes;
    std::vector<WGroup> groups;
    std::vector<WSubgraph> subgraphs;
    nlohmann::json view = nlohmann::json::object();
    nlohmann::json extra = nlohmann::json::object();
    nlohmann::json unknown = nlohmann::json::object();
};

inline constexpr const char* kWorkflowFormat = "graphicshost.workflow";
inline constexpr int kWorkflowVersion = 1;

nlohmann::json toJson(const Workflow& workflow);
// Migrates old documents first. Throws std::runtime_error with a message for people when the document is not a workflow
// or comes from a newer version.
Workflow workflowFromJson(const nlohmann::json& document);
// Brings a document of any older version up to kWorkflowVersion (returns it unchanged when it is current).
nlohmann::json migrateWorkflow(nlohmann::json document);

class ConvertError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Workflow -> job (section 4.4): subgraphs are flattened (inner node ids are "<instance>:<inner>"), muted nodes are
// dropped, bypassed nodes are short-circuited (each output takes the first linked input of the same type), reroute
// and primitive constant nodes are folded into their links, widgets become plain inputs. A node type the registry
// does not know is kept as it is; the validator reports it. Throws ConvertError when a bypass has nothing to pass on,
// a subgraph is missing or contains itself, or links loop through pass-through nodes.
JobGraph toJob(const Workflow& workflow, const Registry& registry);

}  // namespace gh::graph
