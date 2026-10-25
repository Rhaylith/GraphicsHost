#pragma once

#include "graph/JobGraph.h"
#include "graph/Value.h"

#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

namespace gh::graph {

// A node's own failure (bad input it could only detect while running, a failed model call ...). The executor reports it
// against the node.
class NodeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Thrown by a node that noticed the job was interrupted (its stop token is set); the executor ends the job quietly.
class Interrupted : public std::exception {
public:
    const char* what() const noexcept override { return "interrupted"; }
};

// A sub-graph a node returns instead of values (node expansion): the executor validates it, runs it with ids derived from
// the node's own id, and the node's outputs are the listed outputs of the sub-graph.
struct Expansion {
    JobGraph graph;
    std::vector<NodeRef> outputs;
};

struct NodeResult {
    std::vector<Value> outputs;
    // Lazy inputs the node needs after all: the executor evaluates them and calls the node again.
    std::vector<std::string> needInputs;
    std::optional<Expansion> expansion;
};

// What a node can see while it runs.
struct NodeContext {
    std::string nodeId;
    std::string jobId;
    std::map<std::string, Value> inputs;   // widgets and linked inputs; lazy inputs only after they were requested
    std::map<std::string, NodeRef> links;  // where the linked inputs come from (a node that expands refers to these in its sub-graph)
    std::stop_token stop;                  // set when the job is interrupted: long loops check it
    // Reports progress inside the node (a sampler's step); both may be empty.
    std::function<void(int step, int steps)> progress;
    // Offers a preview picture (an IMAGE value) to the clients that watch the job.
    std::function<void(const Value& image)> preview;

    bool has(const std::string& name) const { return inputs.count(name) != 0; }
    const Value& get(const std::string& name) const {
        const auto it = inputs.find(name);
        if (it == inputs.end()) throw NodeError("missing input '" + name + "'");
        return it->second;
    }
};

using NodeFn = std::function<NodeResult(NodeContext&)>;

}  // namespace gh::graph
