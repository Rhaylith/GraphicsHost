#include "graph/Tips.h"

namespace gh::graph::tips {

const char* common(std::string_view key) {
    for (const CommonTip& tip : allCommon)
        if (key == tip.key) return tip.text;
    return nullptr;
}

const OptionTable* optionTable(std::string_view name) {
    for (const OptionTable& table : allOptionTables)
        if (name == table.name) return &table;
    return nullptr;
}

Param withOptions(Param param, std::span<const OptionTip> table) {
    std::vector<std::string> names;
    std::map<std::string, std::string> tips;
    for (const OptionTip& o : table) {
        names.push_back(o.option);
        tips[o.option] = o.tip;
    }
    return param.options(std::move(names)).optionTips(std::move(tips));
}

nlohmann::json toJson() {
    nlohmann::json common = nlohmann::json::object(), options = nlohmann::json::object();
    for (const CommonTip& tip : allCommon) common[tip.key] = tip.text;
    for (const OptionTable& table : allOptionTables) {
        nlohmann::json one = nlohmann::json::object();
        for (const OptionTip& o : table.options) one[o.option] = o.tip;
        options[table.name] = std::move(one);
    }
    return {{"common", common}, {"options", options}};
}

std::string common(std::string_view key, std::string_view addition) {
    const char* base = common(key);
    std::string text = base ? base : "";
    if (!addition.empty()) {
        if (!text.empty()) text += ' ';
        text += addition;
    }
    return text;
}

}  // namespace gh::graph::tips
