#pragma once

#include "graph/JobGraph.h"
#include "graph/NodeDef.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <string>
#include <vector>

namespace gh::graph {

// One thing wrong with a job, tied to a node and, when it applies, to one of its inputs, so the editor can mark the
// socket. `code` is stable (clients switch on it); `message` is for people.
//   unknown_type, unknown_input, missing_input, bad_link, type_mismatch, bad_value, out_of_range, unknown_option,
//   missing_model, custom, cycle, unknown_output, no_outputs
struct ValidationIssue {
    std::string node;     // empty: the job as a whole
    std::string input;    // empty: the node as a whole
    std::string code;
    std::string message;
    nlohmann::json details = nlohmann::json::object();   // e.g. {"folder":"loras","name":"x"} for missing_model, {"min":1,"max":10}
};

struct ValidationResult {
    std::vector<ValidationIssue> issues;
    bool ok() const { return issues.empty(); }
    nlohmann::json toJson() const;
};

struct ValidateOptions {
    // Does a model file exist in a model folder ("checkpoints", "loras" ...)? Unset: names are not checked.
    std::function<bool(const std::string& folder, const std::string& name)> modelExists;
};

// Checks the whole job and reports every problem, not just the first: unknown node types and inputs, required inputs
// without a link or value, links to missing nodes or outputs, link types that do not fit, values of the wrong kind or out
// of range (after rounding to the input's step), combo values that are not offered, missing model files, custom node
// checks, cycles, and output ids that do not exist.
ValidationResult validate(const JobGraph& job, const Registry& registry, const ValidateOptions& options = {});

}  // namespace gh::graph
