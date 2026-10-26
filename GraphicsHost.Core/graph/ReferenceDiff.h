#pragma once

#include "graph/NodeDef.h"
#include "graph/ReferenceDoc.h"

#include <string>
#include <vector>

namespace gh::graph {

// The result of comparing the node registry with docs/workflow-node-reference.md.
struct ReferenceDiff {
    std::vector<std::string> problems;       // registered nodes that disagree with the document (names, types, ranges, defaults, tooltips)
    std::vector<std::string> undocumented;   // registered nodes the document does not list
    std::vector<std::string> pending;        // nodes the document lists that are not registered yet
};

// Every registered node must match its rows in the document: display name, every non-hidden input present, the type
// cell (kind, range, step, default, choices, model folder, socket type) and the tooltip (a shared "→ C.key" tip must
// equal the shared text; other cells must equal the code's text). Rows that only describe several parameters at once
// are checked for the names they give.
ReferenceDiff diffReference(const Registry& registry, const ReferenceDoc& doc);

// The registry rendered as markdown tables in the document's format (one table per category), to start or refresh the
// rows of newly written nodes. Rendering and then parsing and diffing gives no problems.
std::string renderReference(const Registry& registry);

}  // namespace gh::graph
