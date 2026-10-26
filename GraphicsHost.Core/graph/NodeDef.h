#pragma once

#include "graph/NodeApi.h"

#include <nlohmann/json.hpp>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Marks text for the translation tools (lupdate) without needing Qt in this library; the editor translates it with
// QCoreApplication::translate("tips", text). Tooltips, node names and descriptions are written through these.
#ifndef QT_TRANSLATE_NOOP
#define QT_TRANSLATE_NOOP(context, text) text
#endif

namespace gh::graph {

enum class InputKind { Required, Optional, Hidden };

// One input of a node: a socket (only links), a widget (a value the user edits, which can also be linked), or both.
struct InputDef {
    std::string name;
    ValueType type = ValueType::Any;
    InputKind kind = InputKind::Required;
    bool widget = false;
    nlohmann::json def;                           // default value of a widget
    std::optional<double> min, max, step, round;  // numbers
    std::string display = "number";               // Float/Int: "number" or "slider"
    bool seed = false;                            // an Int that has "control after run"
    bool multiline = false;                       // String
    bool dynamicPrompts = false;                  // String: {a|b} choices are resolved
    std::vector<std::string> options;             // Combo
    std::map<std::string, std::string> optionTips;
    std::string modelFolder;                      // Combo that lists a model folder live, e.g. "loras"
    bool lazy = false;
    std::string tip;                              // the tooltip
    bool isWidget() const { return widget; }
};

struct OutputDef {
    std::string name;
    ValueType type = ValueType::Any;
    std::string tip;
    bool isList = false;
};

// ---- fluent builders for inputs and outputs ---------------------------------------------------------------------------
class Param {
public:
    Param(std::string name, ValueType type, bool widget) {
        def_.name = std::move(name);
        def_.type = type;
        def_.widget = widget;
    }
    Param& tip(std::string text) { def_.tip = std::move(text); return *this; }
    Param& def(nlohmann::json value) { def_.def = std::move(value); return *this; }
    Param& range(double min, double max) { def_.min = min; def_.max = max; return *this; }
    Param& step(double s) { def_.step = s; return *this; }
    Param& round(double r) { def_.round = r; return *this; }
    Param& slider() { def_.display = "slider"; return *this; }
    Param& seedControl() { def_.seed = true; return *this; }
    Param& multiline() { def_.multiline = true; return *this; }
    Param& dynamicPrompts() { def_.dynamicPrompts = true; return *this; }
    Param& options(std::vector<std::string> list) { def_.options = std::move(list); return *this; }
    Param& optionTips(std::map<std::string, std::string> tips) { def_.optionTips = std::move(tips); return *this; }
    Param& modelFolder(std::string folder) { def_.modelFolder = std::move(folder); return *this; }
    Param& optional() { def_.kind = InputKind::Optional; return *this; }
    Param& hidden() { def_.kind = InputKind::Hidden; return *this; }
    Param& lazy() { def_.lazy = true; return *this; }
    operator const InputDef&() const { return def_; }
    const InputDef& get() const { return def_; }

private:
    InputDef def_;
};

// sockets
inline Param Socket(std::string name, ValueType type) { return Param(std::move(name), type, false); }
inline Param Model(std::string name) { return Socket(std::move(name), ValueType::Model); }
inline Param TextEncoder(std::string name) { return Socket(std::move(name), ValueType::TextEncoder); }
inline Param Vae(std::string name) { return Socket(std::move(name), ValueType::Vae); }
inline Param Conditioning(std::string name) { return Socket(std::move(name), ValueType::Conditioning); }
inline Param Latent(std::string name) { return Socket(std::move(name), ValueType::Latent); }
inline Param ImageIn(std::string name) { return Socket(std::move(name), ValueType::Image); }
inline Param MaskIn(std::string name) { return Socket(std::move(name), ValueType::Mask); }
inline Param AnyIn(std::string name) { return Socket(std::move(name), ValueType::Any); }
// widgets
inline Param Int(std::string name) { return Param(std::move(name), ValueType::Int, true); }
inline Param Float(std::string name) { return Param(std::move(name), ValueType::Float, true); }
inline Param String(std::string name) { return Param(std::move(name), ValueType::String, true); }
inline Param Bool(std::string name) { return Param(std::move(name), ValueType::Boolean, true); }
inline Param Combo(std::string name) { return Param(std::move(name), ValueType::Combo, true); }

class Out {
public:
    Out(std::string name, ValueType type) { def_.name = std::move(name); def_.type = type; }
    Out& tip(std::string text) { def_.tip = std::move(text); return *this; }
    Out& list() { def_.isList = true; return *this; }
    operator const OutputDef&() const { return def_; }

private:
    OutputDef def_;
};

// ---- the node definition ----------------------------------------------------------------------------------------------------
class NodeDef {
public:
    using ValidateFn = std::function<std::vector<std::string>(const std::map<std::string, nlohmann::json>& literals)>;
    // An extra part of the cache key (e.g. a file's modification time); returning "*" means "always run again".
    using FingerprintFn = std::function<std::string(const std::map<std::string, nlohmann::json>& literals)>;

    explicit NodeDef(std::string id) : id_(std::move(id)) {}

    NodeDef& display(std::string v) { display_ = std::move(v); return *this; }
    NodeDef& category(std::string v) { category_ = std::move(v); return *this; }
    NodeDef& description(std::string v) { description_ = std::move(v); return *this; }
    NodeDef& input(const InputDef& in) { inputs_.push_back(in); return *this; }
    NodeDef& output(const OutputDef& out) { outputs_.push_back(out); return *this; }
    NodeDef& run(NodeFn fn) { run_ = std::move(fn); return *this; }
    NodeDef& isOutput() { isOutput_ = true; return *this; }
    NodeDef& acceptsLists() { acceptsLists_ = true; return *this; }
    NodeDef& expands() { expands_ = true; return *this; }
    NodeDef& deprecated() { deprecated_ = true; return *this; }
    NodeDef& experimental() { experimental_ = true; return *this; }
    NodeDef& searchAliases(std::vector<std::string> words) { aliases_ = std::move(words); return *this; }
    NodeDef& validate(ValidateFn fn) { validate_ = std::move(fn); return *this; }
    NodeDef& fingerprint(FingerprintFn fn) { fingerprint_ = std::move(fn); return *this; }

    const std::string& id() const { return id_; }
    const std::string& displayName() const { return display_; }
    const std::string& categoryName() const { return category_; }
    const std::string& descriptionText() const { return description_; }
    const std::vector<InputDef>& inputs() const { return inputs_; }
    const std::vector<OutputDef>& outputs() const { return outputs_; }
    const NodeFn& runFn() const { return run_; }
    bool outputNode() const { return isOutput_; }
    bool takesLists() const { return acceptsLists_; }
    bool canExpand() const { return expands_; }
    bool isDeprecated() const { return deprecated_; }
    bool isExperimental() const { return experimental_; }
    const std::vector<std::string>& aliases() const { return aliases_; }
    const ValidateFn& validateFn() const { return validate_; }
    const FingerprintFn& fingerprintFn() const { return fingerprint_; }
    const InputDef* findInput(const std::string& name) const;

private:
    std::string id_, display_, category_, description_;
    std::vector<InputDef> inputs_;
    std::vector<OutputDef> outputs_;
    NodeFn run_;
    bool isOutput_ = false, acceptsLists_ = false, expands_ = false, deprecated_ = false, experimental_ = false;
    std::vector<std::string> aliases_;
    ValidateFn validate_;
    FingerprintFn fingerprint_;
};

// All node definitions known to this process: the built-in ones and, later, those of plugins.
class Registry {
public:
    // Throws std::invalid_argument for a duplicate id, an input or output without a name, or duplicate input names.
    void add(NodeDef def);
    const NodeDef* find(const std::string& id) const;
    std::vector<const NodeDef*> all() const;   // sorted by category, then display name
    size_t size() const { return nodes_.size(); }

    // The node catalogue the API and the editor read: every node with its inputs, outputs, ranges, defaults, options and
    // tooltips, and the type colours.
    nlohmann::json catalogue() const;

    // The registry of the built-in nodes (filled by nodes/Builtin.cpp).
    static const Registry& builtin();

private:
    std::map<std::string, NodeDef> nodes_;
};

}  // namespace gh::graph
