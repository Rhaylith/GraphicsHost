#include "graph/Value.h"

#include <cmath>
#include <stdexcept>

namespace gh::graph {

namespace {
struct TypeInfo {
    ValueType type;
    const char* id;
    const char* color;
};
const TypeInfo kTypes[] = {
    {ValueType::Any, "*", "#9aa0a6"},
    {ValueType::Model, "MODEL", "#b39ddb"},
    {ValueType::TextEncoder, "TEXT_ENCODER", "#ffd54f"},
    {ValueType::Vae, "VAE", "#ff7043"},
    {ValueType::Conditioning, "CONDITIONING", "#ffa726"},
    {ValueType::Latent, "LATENT", "#ff80ab"},
    {ValueType::Image, "IMAGE", "#64b5f6"},
    {ValueType::Mask, "MASK", "#81c784"},
    {ValueType::Noise, "NOISE", "#b0bec5"},
    {ValueType::Guider, "GUIDER", "#80cbc4"},
    {ValueType::Sampler, "SAMPLER", "#ffab91"},
    {ValueType::Sigmas, "SIGMAS", "#fff176"},
    {ValueType::ControlNet, "CONTROL_NET", "#4db6ac"},
    {ValueType::StyleModel, "STYLE_MODEL", "#c5e1a5"},
    {ValueType::ClipVision, "CLIP_VISION", "#a5d6a7"},
    {ValueType::ClipVisionOutput, "CLIP_VISION_OUTPUT", "#90caf9"},
    {ValueType::UpscaleModel, "UPSCALE_MODEL", "#ce93d8"},
    {ValueType::Gligen, "GLIGEN", "#f48fb1"},
    {ValueType::PhotoMaker, "PHOTOMAKER", "#ef9a9a"},
    {ValueType::Hooks, "HOOKS", "#bcaaa4"},
    {ValueType::HookKeyframes, "HOOK_KEYFRAMES", "#d7ccc8"},
    {ValueType::TimestepKeyframes, "TIMESTEP_KEYFRAMES", "#cfd8dc"},
    {ValueType::Audio, "AUDIO", "#aed581"},
    {ValueType::Video, "VIDEO", "#4fc3f7"},
    {ValueType::Mesh, "MESH", "#dce775"},
    {ValueType::Voxel, "VOXEL", "#ffe082"},
    {ValueType::Load3dCamera, "LOAD3D_CAMERA", "#80deea"},
    {ValueType::Int, "INT", "#29b6f6"},
    {ValueType::Float, "FLOAT", "#26c6da"},
    {ValueType::String, "STRING", "#9ccc65"},
    {ValueType::Boolean, "BOOLEAN", "#ec407a"},
    {ValueType::Combo, "COMBO", "#bdbdbd"},
};

const TypeInfo& info(ValueType type) {
    for (const auto& t : kTypes)
        if (t.type == type) return t;
    return kTypes[0];
}
}  // namespace

const char* typeId(ValueType type) { return info(type).id; }
const char* typeColor(ValueType type) { return info(type).color; }

bool parseTypeId(const std::string& id, ValueType& out) {
    for (const auto& t : kTypes) {
        if (id == t.id) {
            out = t.type;
            return true;
        }
    }
    return false;
}

const std::vector<ValueType>& allValueTypes() {
    static const std::vector<ValueType> all = [] {
        std::vector<ValueType> v;
        for (const auto& t : kTypes) v.push_back(t.type);
        return v;
    }();
    return all;
}

bool linkCompatible(ValueType from, ValueType to) {
    if (from == ValueType::Any || to == ValueType::Any || from == to) return true;
    if (from == ValueType::Int && to == ValueType::Float) return true;
    if (from == ValueType::Combo && to == ValueType::String) return true;
    if (from == ValueType::String && to == ValueType::Combo) return true;
    return false;
}

int64_t Value::asInt() const {
    if (const auto* i = std::get_if<int64_t>(&data)) return *i;
    if (const auto* d = std::get_if<double>(&data)) return static_cast<int64_t>(std::llround(*d));
    if (const auto* b = std::get_if<bool>(&data)) return *b ? 1 : 0;
    if (const auto* s = std::get_if<std::string>(&data)) {
        try {
            size_t used = 0;
            const int64_t v = std::stoll(*s, &used);
            if (used == s->size()) return v;
        } catch (const std::exception&) {
        }
    }
    throw std::runtime_error(std::string("the value is not a whole number (") + typeId(type) + ")");
}

double Value::asFloat() const {
    if (const auto* d = std::get_if<double>(&data)) return *d;
    if (const auto* i = std::get_if<int64_t>(&data)) return static_cast<double>(*i);
    if (const auto* b = std::get_if<bool>(&data)) return *b ? 1.0 : 0.0;
    if (const auto* s = std::get_if<std::string>(&data)) {
        try {
            size_t used = 0;
            const double v = std::stod(*s, &used);
            if (used == s->size()) return v;
        } catch (const std::exception&) {
        }
    }
    throw std::runtime_error(std::string("the value is not a number (") + typeId(type) + ")");
}

bool Value::asBool() const {
    if (const auto* b = std::get_if<bool>(&data)) return *b;
    if (const auto* i = std::get_if<int64_t>(&data)) return *i != 0;
    if (const auto* d = std::get_if<double>(&data)) return *d != 0.0;
    if (const auto* s = std::get_if<std::string>(&data)) {
        if (*s == "true" || *s == "1" || *s == "on") return true;
        if (*s == "false" || *s == "0" || *s == "off" || s->empty()) return false;
    }
    throw std::runtime_error(std::string("the value is not true or false (") + typeId(type) + ")");
}

std::string Value::asString() const {
    if (const auto* s = std::get_if<std::string>(&data)) return *s;
    if (const auto* i = std::get_if<int64_t>(&data)) return std::to_string(*i);
    if (const auto* b = std::get_if<bool>(&data)) return *b ? "true" : "false";
    if (const auto* d = std::get_if<double>(&data)) {
        std::string s = std::to_string(*d);
        s.erase(s.find_last_not_of('0') + 1);   // 1.500000 -> 1.5
        if (!s.empty() && s.back() == '.') s.pop_back();
        return s;
    }
    throw std::runtime_error(std::string("the value is not text (") + typeId(type) + ")");
}

const tensor::Tensor& Value::asTensor() const {
    const auto* t = std::get_if<std::shared_ptr<const tensor::Tensor>>(&data);
    if (!t || !*t) throw std::runtime_error(std::string("the value is not a tensor (") + typeId(type) + ")");
    return **t;
}

std::string Value::summary() const {
    std::string out = typeId(type);
    if (isList()) return out + " list of " + std::to_string(list().size());
    if (std::holds_alternative<std::monostate>(data)) return out + " (empty)";
    if (const auto* t = std::get_if<std::shared_ptr<const tensor::Tensor>>(&data)) {
        out += " ";
        for (int i = 0; i < (*t)->ndim(); ++i) out += (i ? "x" : "") + std::to_string((*t)->dim(i));
        return out;
    }
    if (std::holds_alternative<std::shared_ptr<const void>>(data)) return out;
    std::string text = asString();
    if (text.size() > 60) text = text.substr(0, 57) + "...";
    return out + " " + text;
}

size_t Value::approxBytes() const {
    if (const auto* t = std::get_if<std::shared_ptr<const tensor::Tensor>>(&data)) return static_cast<size_t>((*t)->numel()) * sizeof(float);
    if (const auto* s = std::get_if<std::string>(&data)) return s->size();
    if (isList()) {
        size_t n = 0;
        for (const auto& v : list()) n += v.approxBytes();
        return n;
    }
    return 16;
}

}  // namespace gh::graph
