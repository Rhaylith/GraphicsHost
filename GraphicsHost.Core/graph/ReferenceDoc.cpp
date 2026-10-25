#include "graph/ReferenceDoc.h"

#include <sstream>

namespace gh::graph {

namespace {

bool starts(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

std::string trimmed(std::string s) {
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r") - b + 1);
}

// "| a | b |" -> {"a", "b"} (the text of a cell never contains " | ")
std::vector<std::string> cells(const std::string& line) {
    std::vector<std::string> out;
    std::string body = trimmed(line);
    if (starts(body, "|")) body.erase(0, 1);
    if (!body.empty() && body.back() == '|') body.pop_back();
    size_t at = 0;
    while (true) {
        const size_t bar = body.find(" | ", at);
        if (bar == std::string::npos) {
            out.push_back(trimmed(body.substr(at)));
            break;
        }
        out.push_back(trimmed(body.substr(at, bar - at)));
        at = bar + 3;
    }
    return out;
}

}  // namespace

ReferenceDoc parseReferenceDoc(const std::string& markdown) {
    ReferenceDoc doc;
    std::istringstream in(markdown);
    std::string line, section, table;
    while (std::getline(in, line)) {
        if (starts(line, "## ")) {
            section = starts(line, "## C.") ? "C" : starts(line, "## O.") ? "O" : starts(line, "## S.") ? "S" : "";
            table.clear();
            continue;
        }
        if (section == "O" && starts(line, "### O.")) {
            const size_t end = line.find_first_of(" \t", 6);
            table = trimmed(line.substr(6, end == std::string::npos ? std::string::npos : end - 6));
            doc.options[table];
            continue;
        }
        if (!starts(line, "| ")) continue;
        const auto c = cells(line);
        if (c.size() < 2) continue;
        if (section == "S" && c.size() >= 3 && starts(c[0], "`")) {
            // the keys at the start of the first cell: `a`, `b` (words after them, such as "(existing)", are not keys)
            size_t at = 0;
            while (at < c[0].size() && c[0][at] == '`') {
                const size_t close = c[0].find('`', at + 1);
                if (close == std::string::npos) break;
                doc.settings[c[0].substr(at + 1, close - at - 1)] = c[2];
                at = close + 1;
                if (c[0].compare(at, 2, ", ") != 0) break;
                at += 2;
            }
        } else if (section == "C" && starts(c[0], "`C.") && c[0].back() == '`') {
            doc.common[c[0].substr(3, c[0].size() - 4)] = c[1];
        } else if (section == "O" && !table.empty() && c[0] != "Option") {
            size_t at = 0;
            while (true) {
                const size_t slash = c[0].find(" / ", at);
                doc.options[table].emplace_back(c[0].substr(at, slash == std::string::npos ? std::string::npos : slash - at), c[1]);
                if (slash == std::string::npos) break;
                at = slash + 3;
            }
        }
    }
    return doc;
}

}  // namespace gh::graph
