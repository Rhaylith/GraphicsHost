#pragma once

#include "tensor/Tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace gh::graph {

// The types that travel on links (workflow plan, section 4.1). Each has an id used in files and the API, and a colour used
// by the editor.
enum class ValueType {
    Any,
    Model, TextEncoder, Vae, Conditioning, Latent, Image, Mask,
    Noise, Guider, Sampler, Sigmas,
    ControlNet, StyleModel, ClipVision, ClipVisionOutput, UpscaleModel, Gligen, PhotoMaker,
    Hooks, HookKeyframes, TimestepKeyframes,
    Audio, Video, Mesh, Voxel, Load3dCamera,
    Int, Float, String, Boolean, Combo,
};

const char* typeId(ValueType type);                       // "MODEL", "INT", "*" ...
bool parseTypeId(const std::string& id, ValueType& out);
const char* typeColor(ValueType type);                    // "#rrggbb"
const std::vector<ValueType>& allValueTypes();

// Can an output of type `from` feed an input of type `to`? Any matches everything; an INT may feed a FLOAT; widget types
// otherwise only match themselves (a COMBO value is a string).
bool linkCompatible(ValueType from, ValueType to);

struct Value;
using ValueList = std::vector<Value>;

// One value flowing through a graph. Small types are stored inline; images, masks and latents as tensors; every other
// type (models, conditioning, samplers ...) as an opaque shared object that the nodes of that type agree on. A list is a
// value of its own (the node mapping rules of the executor decide what happens to it).
struct Value {
    ValueType type = ValueType::Any;
    std::variant<std::monostate, int64_t, double, bool, std::string, std::shared_ptr<const tensor::Tensor>,
                 std::shared_ptr<const void>, std::shared_ptr<const ValueList>>
        data;

    bool empty() const { return std::holds_alternative<std::monostate>(data); }
    bool isList() const { return std::holds_alternative<std::shared_ptr<const ValueList>>(data); }
    const ValueList& list() const { return *std::get<std::shared_ptr<const ValueList>>(data); }

    static Value makeInt(int64_t v) { return {ValueType::Int, v}; }
    static Value makeFloat(double v) { return {ValueType::Float, v}; }
    static Value makeBool(bool v) { return {ValueType::Boolean, v}; }
    static Value makeString(std::string v) { return {ValueType::String, std::move(v)}; }
    static Value makeTensor(ValueType type, tensor::Tensor t) { return {type, std::make_shared<const tensor::Tensor>(std::move(t))}; }
    static Value makeOpaque(ValueType type, std::shared_ptr<const void> p) { return {type, std::move(p)}; }
    static Value makeList(ValueType elementType, ValueList items) { return {elementType, std::make_shared<const ValueList>(std::move(items))}; }

    // Coercions between the scalar types (an INT reads as a FLOAT, a number as text); throw std::runtime_error otherwise.
    int64_t asInt() const;
    double asFloat() const;
    bool asBool() const;
    std::string asString() const;
    const tensor::Tensor& asTensor() const;
    template <typename T>
    std::shared_ptr<const T> asObject() const {
        return std::static_pointer_cast<const T>(std::get<std::shared_ptr<const void>>(data));
    }

    // A short description for events and logs: "INT 42", "IMAGE 1x512x512x3", "MODEL".
    std::string summary() const;
    // Rough size in bytes, for the output cache.
    size_t approxBytes() const;
};

}  // namespace gh::graph
