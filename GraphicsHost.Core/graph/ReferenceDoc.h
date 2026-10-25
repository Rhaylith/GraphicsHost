#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace gh::graph {

// The tables of docs/workflow-node-reference.md read back as data, so tests and the docs generator can compare the
// document with the code.
struct ReferenceDoc {
    std::map<std::string, std::string> common;                                          // C.<key> -> tooltip
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> options;   // O.<table> -> (option, tooltip); "a / b" rows are split
    std::map<std::string, std::string> settings;                                        // S rows: dotted key -> tooltip (a row may name several keys)
};

ReferenceDoc parseReferenceDoc(const std::string& markdown);

}  // namespace gh::graph
