#include "graph/TipCoverage.h"

#include <algorithm>
#include <functional>

namespace gh::graph {

bool hasExample(const std::string& tooltip) {
    return tooltip.find("Example:") != std::string::npos || tooltip.find("e.g.") != std::string::npos;
}

std::vector<std::string> tooltipProblems(const Registry& registry) {
    std::vector<std::string> problems;
    for (const NodeDef* def : registry.all()) {
        for (const InputDef& in : def->inputs()) {
            const std::string where = def->id() + "." + in.name + ": ";
            if (in.kind == InputKind::Hidden) continue;
            if (in.tip.empty()) {
                problems.push_back(where + "no tooltip");
                continue;
            }
            if (in.widget && !hasExample(in.tip)) problems.push_back(where + "the tooltip has no example");
            for (const auto& [option, tip] : in.optionTips)
                if (!hasExample(tip)) problems.push_back(where + "the tooltip of option '" + option + "' has no example");
            for (const auto& [option, tip] : in.optionTips)
                if (std::find(in.options.begin(), in.options.end(), option) == in.options.end())
                    problems.push_back(where + "a tooltip for '" + option + "', which is not an option");
        }
    }
    return problems;
}

std::vector<std::string> settingKeys(const nlohmann::json& settings) {
    std::vector<std::string> keys;
    const std::function<void(const nlohmann::json&, const std::string&)> walk = [&](const nlohmann::json& j, const std::string& prefix) {
        for (const auto& [key, value] : j.items()) {
            const std::string path = prefix.empty() ? key : prefix + "." + key;
            if (value.is_object()) walk(value, path);
            else keys.push_back(path);
        }
    };
    walk(settings, "");
    return keys;
}

}  // namespace gh::graph
