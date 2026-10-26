#pragma once

#include "graph/NodeDef.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

// Tooltips (workflow plan, section 12 and docs/workflow-node-reference.md).
//
// The shared texts (reference section C) and the per-option texts (section O) are constants in graph/TipsData.h, generated
// from the reference document by tools/gen_tips.awk. Each is wrapped in QT_TRANSLATE_NOOP("tips", ...), so lupdate finds
// it and the editor translates it with QCoreApplication::translate("tips", text). Node-specific tooltips are written
// where the node is defined, through the same macro.
namespace gh::graph::tips {

struct CommonTip {
    const char* key;
    const char* text;
};

struct OptionTip {
    const char* option;
    const char* tip;
};

struct SettingTip {
    const char* key;   // "server.port", "memory.mode" ...
    const char* text;
};

struct OptionTable {
    const char* name;
    std::span<const OptionTip> options;
};

}  // namespace gh::graph::tips

#include "graph/TipsData.h"

namespace gh::graph::tips {

// The shared tooltip with this key ("seed", "steps", ...), or nullptr.
const char* common(std::string_view key);

// The tips of one option list ("sampler", "scheduler", "upscaleMethod", "blendMode", "predictionType"), or nullptr.
const OptionTable* optionTable(std::string_view name);

// The input with its choices and their tooltips taken from an option table.
Param withOptions(Param param, std::span<const OptionTip> table);

// The tooltip of a setting by its dotted key ("server.port"), or nullptr.
const char* setting(std::string_view key);

// Every shared tip, option tip and setting tip as JSON: {"common": {key: text}, "options": {table: {option: text}},
// "settings": {key: text}} (the editor and the API serve it next to the node catalogue).
nlohmann::json toJson();

// The shared tooltip followed by a node-specific sentence (the reference shows this as "→ C.key + sentence").
std::string common(std::string_view key, std::string_view addition);

}  // namespace gh::graph::tips
