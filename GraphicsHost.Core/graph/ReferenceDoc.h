#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace gh::graph {

// One row of a node table: the parameters it names (`names` is empty and `loose` set when the cell is a description such
// as "encoder1 … encoder4"), the type cell and the tooltip cell, as written.
struct DocParam {
    std::vector<std::string> names;
    bool loose = false;
    std::string type;
    std::string tip;
};

struct DocNode {
    std::string display;
    std::string id;
    std::string section;   // "N1" ...
    std::vector<DocParam> params;
};

// The tables of docs/workflow-node-reference.md read back as data, so tests and the docs generator can compare the
// document with the code.
struct ReferenceDoc {
    std::map<std::string, std::string> common;                                          // C.<key> -> tooltip
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> options;   // O.<table> -> (option, tooltip); "a / b" rows are split
    std::map<std::string, std::string> settings;                                        // S rows: dotted key -> tooltip (a row may name several keys)
    std::vector<DocNode> nodes;                                                         // N tables, in document order
};

ReferenceDoc parseReferenceDoc(const std::string& markdown);

}  // namespace gh::graph
