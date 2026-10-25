#include "nodes/Builtin.h"
#include "nodes/MathExpression.h"

#include <cmath>
#include <algorithm>
#include <regex>

namespace gh::nodes {

using namespace graph;

namespace {

constexpr double kBig = 1e9;            // far beyond anything a person types
constexpr double kBigInt = 9007199254740991.0;   // 2^53 - 1: the largest whole number a double holds exactly

NodeResult one(Value v) { return NodeResult{{std::move(v)}}; }

// The comparison used by util.compare: numbers compare as numbers, anything else as text.
int compareValues(const Value& a, const Value& b) {
    const auto numeric = [](const Value& v) { return std::holds_alternative<int64_t>(v.data) || std::holds_alternative<double>(v.data) || std::holds_alternative<bool>(v.data); };
    if (numeric(a) && numeric(b)) {
        const double x = a.asFloat(), y = b.asFloat();
        return x < y ? -1 : x > y ? 1 : 0;
    }
    const std::string x = a.asString(), y = b.asString();
    return x < y ? -1 : x > y ? 1 : 0;
}

void registerConstants(Registry& r) {
    r.add(NodeDef("util.int")
              .display("Integer")
              .category("utility")
              .description("A whole number that can feed several inputs at once.")
              .searchAliases({"constant", "number", "primitive"})
              .input(Int("value").range(-kBigInt, kBigInt).def(0).tip("A whole number that can feed several inputs at once. Example: one Integer feeding the width of three nodes."))
              .output(Out("value", ValueType::Int).tip("The number."))
              .run([](NodeContext& c) { return one(Value::makeInt(c.get("value").asInt())); }));
    r.add(NodeDef("util.float")
              .display("Float")
              .category("utility")
              .description("A decimal number that can feed several inputs at once.")
              .searchAliases({"constant", "number", "primitive"})
              .input(Float("value").range(-kBig, kBig).step(0.01).def(0.0).tip("A decimal number that can feed several inputs at once. Example: one Float feeding the cfg of two samplers."))
              .output(Out("value", ValueType::Float).tip("The number."))
              .run([](NodeContext& c) { return one(Value::makeFloat(c.get("value").asFloat())); }));
    r.add(NodeDef("util.string")
              .display("Text")
              .category("utility")
              .description("Text that can feed several inputs at once.")
              .searchAliases({"constant", "string", "primitive", "prompt"})
              .input(String("value").multiline().def("").tip("Text that can feed several inputs at once. Example: one prompt feeding two Text Encode nodes."))
              .output(Out("value", ValueType::String).tip("The text."))
              .run([](NodeContext& c) { return one(Value::makeString(c.get("value").asString())); }));
    r.add(NodeDef("util.bool")
              .display("Boolean")
              .category("utility")
              .description("An on/off switch that can feed several inputs at once.")
              .searchAliases({"constant", "toggle", "switch", "primitive"})
              .input(Bool("value").def(false).tip("An on/off switch that can feed several inputs at once. Example: one toggle that turns on both the hi-res pass and the upscaler."))
              .output(Out("value", ValueType::Boolean).tip("The switch position."))
              .run([](NodeContext& c) { return one(Value::makeBool(c.get("value").asBool())); }));
    r.add(NodeDef("util.primitive")
              .display("Constant")
              .category("utility")
              .description("A value that takes the type of whatever it is connected to.")
              .searchAliases({"constant", "primitive", "value"})
              .input(Param("value", ValueType::Any, true)
                         .tip("A value node that copies the widget of the first input it is linked to, including its range and options, and supports *control after run*. Example: one seed shared by two samplers."))
              .output(Out("value", ValueType::Any).tip("The value."))
              .run([](NodeContext& c) { return one(c.get("value")); }));
    r.add(NodeDef("util.reroute")
              .display("Reroute")
              .category("utility")
              .description("A pass-through point to tidy links.")
              .input(AnyIn("value").tip("A pass-through point to tidy links."))
              .output(Out("value", ValueType::Any).tip("The input, unchanged."))
              .run([](NodeContext& c) { return one(c.get("value")); }));
}

void registerNotes(Registry& r) {
    const auto note = [&](const char* id, const char* display, const char* what) {
        r.add(NodeDef(id)
                  .display(display)
                  .category("utility")
                  .description(what)
                  .input(String("text").multiline().def("").tip("Text for people reading the graph; not executed. Example: \"Set denoise to 0.4 for subtle changes.\""))
                  .run([](NodeContext&) { return NodeResult{}; }));
    };
    note("util.note", "Note", "A comment on the canvas.");
    note("util.markdown", "Markdown Note", "A comment on the canvas, formatted with Markdown.");
}

void registerText(Registry& r) {
    r.add(NodeDef("util.textJoin")
              .display("Join Text")
              .category("utility")
              .description("Joins two texts.")
              .searchAliases({"concatenate", "append"})
              .input(String("a").multiline().def("").tip("The first text. Example: \"a red fox\"."))
              .input(String("b").multiline().def("").tip("The second text. Example: \"in the snow\"."))
              .input(String("separator").def(", ").tip("Placed between the two texts. Example: \", \" turns \"fox\" and \"snow\" into \"fox, snow\"."))
              .output(Out("text", ValueType::String).tip("The joined text."))
              .run([](NodeContext& c) { return one(Value::makeString(c.get("a").asString() + c.get("separator").asString() + c.get("b").asString())); }));

    r.add(NodeDef("util.textReplace")
              .display("Replace Text")
              .category("utility")
              .description("Replaces every occurrence of a text, or of a pattern.")
              .searchAliases({"substitute", "find"})
              .input(String("text").multiline().def("").tip("The text to change. Example: \"a cat on a sofa\"."))
              .input(String("find").def("").tip("What to look for. Example: \"cat\"."))
              .input(String("replace").def("").tip("What to put instead; leave empty to delete the match. Example: \"dog\"."))
              .input(Bool("regex").def(false).tip("Treat the search text as a regular expression; $1 in the replacement stands for the first group. Example: find \"(black|white) cat\" and replace \"$1 dog\" turns \"black cat\" into \"black dog\"."))
              .output(Out("text", ValueType::String).tip("The changed text."))
              .run([](NodeContext& c) {
                  const std::string text = c.get("text").asString(), find = c.get("find").asString(), replace = c.get("replace").asString();
                  if (find.empty()) return one(Value::makeString(text));
                  if (c.get("regex").asBool()) {
                      try {
                          return one(Value::makeString(std::regex_replace(text, std::regex(find), replace)));
                      } catch (const std::regex_error& e) {
                          throw NodeError(std::string("the pattern is not valid: ") + e.what(), "find");
                      }
                  }
                  std::string out;
                  size_t at = 0;
                  for (;;) {
                      const size_t hit = text.find(find, at);
                      if (hit == std::string::npos) break;
                      out.append(text, at, hit - at).append(replace);
                      at = hit + find.size();
                  }
                  out.append(text, at, std::string::npos);
                  return one(Value::makeString(out));
              }));

    r.add(NodeDef("util.textSlice")
              .display("Substring")
              .category("utility")
              .description("Takes the characters between two positions.")
              .searchAliases({"cut", "trim", "truncate"})
              .input(String("text").multiline().def("").tip("The text to cut from. Example: a long prompt."))
              .input(Int("start").range(-100000, 100000).def(0).tip("Position of the first character to keep; negative counts from the end. Example: 0 keeps the beginning."))
              .input(Int("end").range(-100000, 100000).def(-1).tip("Position after the last character to keep; -1 means the end. Example: 75 keeps the first 75 characters."))
              .output(Out("text", ValueType::String).tip("The selected part."))
              .run([](NodeContext& c) {
                  const std::string text = c.get("text").asString();
                  const int64_t n = static_cast<int64_t>(text.size());
                  const auto position = [&](int64_t p, bool isEnd) {
                      if (p < 0) p = isEnd && p == -1 ? n : n + p;
                      return std::clamp<int64_t>(p, 0, n);
                  };
                  const int64_t from = position(c.get("start").asInt(), false), to = position(c.get("end").asInt(), true);
                  return one(Value::makeString(to > from ? text.substr(static_cast<size_t>(from), static_cast<size_t>(to - from)) : std::string()));
              }));
}

void registerLogic(Registry& r) {
    r.add(NodeDef("util.compare")
              .display("Compare")
              .category("utility")
              .description("Compares two values and outputs true or false.")
              .searchAliases({"equal", "greater", "less", "if"})
              .input(Param("a", ValueType::Any, true).def(0).tip("The left value; numbers compare as numbers, anything else as text. Example: the width from Image Size."))
              .input(Param("b", ValueType::Any, true).def(0).tip("The right value. Example: 1024."))
              .input(Combo("operation").options({"=", "\xE2\x89\xA0", "<", "\xE2\x89\xA4", ">", "\xE2\x89\xA5"}).def("=").tip("How to compare. Example: > is true when a is larger than b."))
              .output(Out("result", ValueType::Boolean).tip("True when the comparison holds."))
              .run([](NodeContext& c) {
                  const int order = compareValues(c.get("a"), c.get("b"));
                  const std::string op = c.get("operation").asString();
                  bool result = false;
                  if (op == "=") result = order == 0;
                  else if (op == "\xE2\x89\xA0") result = order != 0;
                  else if (op == "<") result = order < 0;
                  else if (op == "\xE2\x89\xA4") result = order <= 0;
                  else if (op == ">") result = order > 0;
                  else if (op == "\xE2\x89\xA5") result = order >= 0;
                  else throw NodeError("unknown comparison '" + op + "'", "operation");
                  return one(Value::makeBool(result));
              }));

    r.add(NodeDef("util.switch")
              .display("Switch")
              .category("utility")
              .description("Passes one of two inputs; the other branch is not run.")
              .searchAliases({"if", "branch", "select", "toggle"})
              .input(Bool("condition").def(false).tip("Which input to pass on: on passes onTrue, off passes onFalse. Example: a Boolean that turns the hi-res pass on and off."))
              .input(AnyIn("onTrue").lazy().optional().tip("The value passed when the condition is on; its branch only runs then. Example: the image after the hi-res pass."))
              .input(AnyIn("onFalse").lazy().optional().tip("The value passed when the condition is off; its branch only runs then. Example: the image straight from the first decode."))
              .output(Out("value", ValueType::Any).tip("The chosen value."))
              .run([](NodeContext& c) {
                  const char* wanted = c.get("condition").asBool() ? "onTrue" : "onFalse";
                  if (!c.has(wanted)) {
                      if (c.links.count(wanted)) return NodeResult{{}, {wanted}, std::nullopt};
                      throw NodeError(std::string(wanted) + " is not connected", wanted);
                  }
                  return one(c.get(wanted));
              }));

    r.add(NodeDef("util.math")
              .display("Math")
              .category("utility")
              .description("Calculates an arithmetic expression.")
              .searchAliases({"calculate", "expression", "formula"})
              .input(String("expression").def("a + b").tip("The calculation, using a, b and c and + - * / % ^ min max round floor ceil abs sqrt. Example: a * 1.5 to compute a hi-res width."))
              .input(Float("a").range(-kBig, kBig).def(0.0).tip("The value called a in the expression. Example: 1024."))
              .input(Float("b").range(-kBig, kBig).def(0.0).tip("The value called b in the expression. Example: 8."))
              .input(Float("c").range(-kBig, kBig).def(0.0).tip("The value called c in the expression. Example: 64."))
              .output(Out("float", ValueType::Float).tip("The result as a decimal number."))
              .output(Out("int", ValueType::Int).tip("The result rounded to a whole number."))
              .run([](NodeContext& c) {
                  double v = 0;
                  try {
                      v = evaluateMath(c.get("expression").asString(), {{'a', c.get("a").asFloat()}, {'b', c.get("b").asFloat()}, {'c', c.get("c").asFloat()}});
                  } catch (const MathError& e) {
                      throw NodeError(e.what(), "expression");
                  }
                  if (!std::isfinite(v)) throw NodeError("the result is not a finite number", "expression");
                  return NodeResult{{Value::makeFloat(v), Value::makeInt(static_cast<int64_t>(std::llround(v)))}};
              }));
}

void registerInspect(Registry& r) {
    r.add(NodeDef("util.previewAny")
              .display("Preview Any")
              .category("utility")
              .description("Shows a value as text.")
              .searchAliases({"inspect", "show", "debug", "display"})
              .isOutput()
              .input(AnyIn("value").tip("The value to show. Example: the seed actually used by a sampler."))
              .output(Out("text", ValueType::String).tip("A description of the value."))
              .run([](NodeContext& c) {
                  const Value& v = c.get("value");
                  const bool scalar = std::holds_alternative<int64_t>(v.data) || std::holds_alternative<double>(v.data) || std::holds_alternative<bool>(v.data) || std::holds_alternative<std::string>(v.data);
                  return one(Value::makeString(scalar ? v.asString() : v.summary()));
              }));

    r.add(NodeDef("util.imageSize")
              .display("Image Size")
              .category("utility")
              .description("Reads the size of an image.")
              .searchAliases({"dimensions", "width", "height", "resolution"})
              .input(ImageIn("image").tip("The image to measure."))
              .output(Out("width", ValueType::Int).tip("Width in pixels."))
              .output(Out("height", ValueType::Int).tip("Height in pixels."))
              .output(Out("batch", ValueType::Int).tip("How many images the batch holds."))
              .run([](NodeContext& c) {
                  const auto& shape = c.get("image").asTensor().shape();
                  if (shape.size() != 4) throw NodeError("an image has the shape [batch, height, width, channels]", "image");
                  return NodeResult{{Value::makeInt(shape[2]), Value::makeInt(shape[1]), Value::makeInt(shape[0])}};
              }));
}

}  // namespace

void registerUtilityNodes(Registry& r) {
    registerConstants(r);
    registerNotes(r);
    registerText(r);
    registerLogic(r);
    registerInspect(r);
}

}  // namespace gh::nodes
