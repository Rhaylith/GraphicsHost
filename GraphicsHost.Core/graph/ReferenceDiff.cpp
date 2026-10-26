#include "graph/ReferenceDiff.h"

#include "graph/Tips.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>

namespace gh::graph {

using Json = nlohmann::json;

namespace {

const std::string kMinus = "\xE2\x88\x92";   // U+2212, used in the document for negative numbers
const std::string kDots = "\xE2\x80\xA6";    // U+2026
const std::string kArrow = "\xE2\x86\x92";   // U+2192
const std::string kMiddot = "\xC2\xB7";      // U+00B7

std::string num(double v) {
    char buf[40];
    if (v == std::floor(v) && std::fabs(v) < 1e18) std::snprintf(buf, sizeof buf, "%.0f", v);
    else std::snprintf(buf, sizeof buf, "%.10g", v);
    std::string s = buf;
    if (!s.empty() && s[0] == '-') s = kMinus + s.substr(1);
    return s;
}

std::optional<double> parseNum(std::string s) {
    for (size_t at; (at = s.find(kMinus)) != std::string::npos;) s.replace(at, kMinus.size(), "-");
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    if (s.empty()) return std::nullopt;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') return std::nullopt;
    return v;
}

std::vector<std::string> split(const std::string& s, const std::string& sep) {
    std::vector<std::string> out;
    size_t at = 0;
    while (true) {
        const size_t p = s.find(sep, at);
        out.push_back(s.substr(at, p == std::string::npos ? std::string::npos : p - at));
        if (p == std::string::npos) break;
        at = p + sep.size();
    }
    return out;
}

std::string stripStars(std::string s) {
    while (s.size() >= 2 && s.front() == '*' && s.back() == '*') s = s.substr(1, s.size() - 2);
    return s;
}

std::string defaultText(const InputDef& in) {
    if (in.def.is_null()) return "";
    if (in.def.is_boolean()) return in.def.get<bool>() ? "true" : "false";
    if (in.def.is_number()) return num(in.def.get<double>());
    if (in.def.is_string()) return in.def.get<std::string>();
    return in.def.dump();
}

std::string typeCell(const InputDef& in) {
    const std::string dflt = defaultText(in);
    const std::string sep = " " + kMiddot + " ";
    if (!in.widget || in.type == ValueType::Any) {
        const std::string name = in.type == ValueType::Any ? "any" : typeId(in.type);
        return name + (in.kind == InputKind::Optional ? " (optional)" : "");
    }
    switch (in.type) {
        case ValueType::Int:
        case ValueType::Float: {
            std::string s = in.type == ValueType::Int ? "int" : "float";
            if (in.min && in.max) s += " " + num(*in.min) + " " + kDots + " " + num(*in.max);
            else if (in.min) s += " \xE2\x89\xA5 " + num(*in.min);
            else if (in.max) s += " \xE2\x89\xA4 " + num(*in.max);
            if (in.step) s += sep + num(*in.step);
            if (!dflt.empty()) s += sep + "**" + dflt + "**";
            return s;
        }
        case ValueType::Boolean: return "bool" + (dflt.empty() ? std::string() : sep + "**" + dflt + "**");
        case ValueType::Combo: {
            if (!in.modelFolder.empty()) return "model list *" + in.modelFolder + "*";
            std::string s = "combo ";
            for (size_t i = 0; i < in.options.size(); ++i) s += (i ? ", " : "") + in.options[i];
            if (!dflt.empty()) s += sep + "**" + dflt + "**";
            return s;
        }
        case ValueType::String: return "string" + (dflt.empty() ? std::string() : sep + "**" + dflt + "**");
        default: return typeId(in.type);
    }
}

// The key of a shared tip whose text this is, or "".
std::string sharedKeyOf(const std::string& tip) {
    for (const auto& c : tips::allCommon)
        if (tip == c.text) return c.key;
    return "";
}

std::string tipCell(const InputDef& in) {
    const std::string key = sharedKeyOf(in.tip);
    return key.empty() ? in.tip : kArrow + " C." + key;
}

// What the document's tooltip cell says the tooltip is; nullopt when the cell points somewhere this checker cannot follow.
std::optional<std::string> resolveTip(const std::string& cell, std::string& problem) {
    const std::string arrow = kArrow + " C.";
    if (cell.rfind(arrow, 0) != 0) {
        if (cell.rfind(kArrow, 0) == 0) return std::nullopt;
        return cell;
    }
    size_t end = arrow.size();
    while (end < cell.size() && std::isalnum(static_cast<unsigned char>(cell[end]))) ++end;
    const std::string key = cell.substr(arrow.size(), end - arrow.size());
    const char* shared = tips::common(key);
    if (!shared) {
        problem = "the reference points to the shared tooltip C." + key + ", which does not exist";
        return std::nullopt;
    }
    const std::string rest = cell.substr(end);
    if (rest.empty()) return std::string(shared);
    if (rest.rfind(" + ", 0) == 0) return tips::common(key, rest.substr(3));
    return std::nullopt;   // "→ C.x / C.y (...)", "→ as Set Area." ...
}

// Compares the type cell of a single-parameter row with the input; returns what differs.
void checkType(const std::string& cell, const InputDef& in, const std::string& where, std::vector<std::string>& problems) {
    const std::string sep = " " + kMiddot + " ";
    const auto parts = split(cell, sep);
    std::string head = parts.empty() ? "" : parts[0];
    if (head.size() > 11 && head.compare(head.size() - 11, 11, " (optional)") == 0) head.erase(head.size() - 11);
    const auto bad = [&](const std::string& what) { problems.push_back(where + ": " + what); };

    if (head.rfind("int", 0) == 0 || head.rfind("float", 0) == 0) {
        const bool isInt = head.rfind("int", 0) == 0;
        if (!in.widget || (isInt ? in.type != ValueType::Int : in.type != ValueType::Float)) return bad("the reference says '" + head.substr(0, head.find(' ')) + "'");
        const size_t dots = head.find(kDots);
        if (dots != std::string::npos) {
            const auto lo = parseNum(head.substr(head.find(' ') + 1, dots - head.find(' ') - 1));
            const auto hi = parseNum(head.substr(dots + kDots.size()));
            if (lo && (!in.min || std::fabs(*in.min - *lo) > 1e-9)) bad("the minimum differs from the reference (" + num(*lo) + ")");
            if (hi && (!in.max || std::fabs(*in.max - *hi) > 1e-9)) bad("the maximum differs from the reference (" + num(*hi) + ")");
        }
        for (size_t i = 1; i < parts.size(); ++i) {
            if (parts[i].rfind("**", 0) == 0) {
                const auto d = parseNum(stripStars(parts[i]));
                if (d && (!in.def.is_number() || std::fabs(in.def.get<double>() - *d) > 1e-9)) bad("the default differs from the reference (" + num(*d) + ")");
            } else if (const auto s = parseNum(parts[i])) {
                if (!in.step || std::fabs(*in.step - *s) > 1e-9) bad("the step differs from the reference (" + num(*s) + ")");
            }
        }
    } else if (head == "bool") {
        if (in.type != ValueType::Boolean) return bad("the reference says bool");
        if (parts.size() > 1) {
            const std::string d = stripStars(parts[1]);
            if ((d == "true" || d == "false") && defaultText(in) != d) bad("the default differs from the reference (" + d + ")");
        }
    } else if (head.rfind("combo ", 0) == 0) {
        if (in.type != ValueType::Combo) return bad("the reference says combo");
        const std::string list = head.substr(6);
        if (list.find(kDots) == std::string::npos && list.find('(') == std::string::npos && !in.options.empty()) {
            std::vector<std::string> docOptions = split(list, ", ");
            if (docOptions != in.options) bad("the choices differ from the reference");
        }
        if (parts.size() > 1) {
            const std::string d = stripStars(parts[1]);
            if (d.find(' ') == std::string::npos && !in.def.is_null() && defaultText(in) != d) bad("the default differs from the reference (" + d + ")");
        }
    } else if (head.rfind("model list", 0) == 0) {
        const size_t star = head.find('*');
        if (in.type != ValueType::Combo) return bad("the reference says a model list");
        if (star != std::string::npos && head.find(" + ") == std::string::npos) {
            const size_t close = head.find('*', star + 1);
            const std::string folder = head.substr(star + 1, close - star - 1);
            if (in.modelFolder != folder) bad("the model folder differs from the reference (" + folder + ")");
        }
    } else if (head == "string") {
        if (in.type != ValueType::String) bad("the reference says string");
    } else if (head == "any") {
        if (in.type != ValueType::Any) bad("the reference says any");
    } else if (!head.empty() && std::all_of(head.begin(), head.end(), [](unsigned char ch) { return std::isupper(ch) || ch == '_'; })) {
        ValueType t = ValueType::Any;
        if (parseTypeId(head, t) && t != in.type) bad("the type differs from the reference (" + head + ")");
    }
}

void checkNode(const NodeDef& def, const DocNode& doc, std::vector<std::string>& problems) {
    if (def.displayName() != doc.display) problems.push_back(def.id() + ": the display name differs from the reference ('" + doc.display + "')");
    std::set<std::string> documented;
    bool loose = false;
    for (const DocParam& p : doc.params) {
        loose = loose || p.loose;
        for (const std::string& name : p.names) {
            const InputDef* in = def.findInput(name);
            const std::string where = def.id() + "." + name;
            if (!in) {
                problems.push_back(where + ": the reference lists a parameter that does not exist");
                continue;
            }
            documented.insert(name);
            std::string problem;
            const auto tip = resolveTip(p.tip, problem);
            if (!problem.empty()) problems.push_back(where + ": " + problem);
            else if (tip && *tip != in->tip) problems.push_back(where + ": the tooltip differs from the reference");
            if (p.names.size() == 1) checkType(p.type, *in, where, problems);
        }
    }
    if (!loose)
        for (const InputDef& in : def.inputs())
            if (in.kind != InputKind::Hidden && !documented.count(in.name)) problems.push_back(def.id() + "." + in.name + ": the reference does not list this input");
}

}  // namespace

ReferenceDiff diffReference(const Registry& registry, const ReferenceDoc& doc) {
    ReferenceDiff diff;
    std::map<std::string, const DocNode*> byId;
    for (const DocNode& n : doc.nodes) byId[n.id] = &n;
    for (const NodeDef* def : registry.all()) {
        const auto it = byId.find(def->id());
        if (it == byId.end()) diff.undocumented.push_back(def->id());
        else checkNode(*def, *it->second, diff.problems);
    }
    for (const DocNode& n : doc.nodes)
        if (!registry.find(n.id)) diff.pending.push_back(n.id);
    return diff;
}

std::string renderReference(const Registry& registry) {
    std::string out;
    std::string category;
    int sections = 0;
    for (const NodeDef* def : registry.all()) {
        if (def->categoryName() != category) {
            category = def->categoryName();
            out += "\n## N" + std::to_string(++sections) + ". " + category + "\n\n| Node (`id`) | Parameter | Type " + kMiddot + " range " + kMiddot + " default | Tooltip |\n|---|---|---|---|\n";
        }
        bool first = true;
        const auto head = [&] { return first ? def->displayName() + " (`" + def->id() + "`)" : std::string(); };
        for (const InputDef& in : def->inputs()) {
            if (in.kind == InputKind::Hidden) continue;
            out += "| " + head() + " | " + in.name + " | " + typeCell(in) + " | " + tipCell(in) + " |\n";
            first = false;
        }
        if (first) out += "| " + head() + " | \xE2\x80\x94 | \xE2\x80\x94 | " + def->descriptionText() + " |\n";   // em dashes: no empty cells
    }
    return out;
}

}  // namespace gh::graph
