#include "models/CheckpointHeader.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace gh::models {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

constexpr uint64_t kMaxSafetensorsHeader = 256ull * 1024 * 1024;
constexpr uint64_t kMaxGgufCount = 10'000'000;

void setError(std::string* error, std::string message) {
    if (error) *error = std::move(message);
}

class Reader {
public:
    explicit Reader(std::ifstream& in) : in_(in) {}
    bool read(void* out, size_t n) {
        in_.read(static_cast<char*>(out), static_cast<std::streamsize>(n));
        return static_cast<size_t>(in_.gcount()) == n;
    }
    template <typename T>
    bool value(T& out) {
        return read(&out, sizeof(T));
    }
    bool string(std::string& out) {
        uint64_t length = 0;
        if (!value(length) || length > (64u << 20)) return false;
        out.resize(static_cast<size_t>(length));
        return length == 0 || read(out.data(), out.size());
    }

private:
    std::ifstream& in_;
};


// GGUF metadata value types
enum GgufType : uint32_t { U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 };

bool readGgufValue(Reader& r, uint32_t type, Json& out, int depth = 0) {
    switch (type) {
        case U8: { uint8_t v; if (!r.value(v)) return false; out = v; return true; }
        case I8: { int8_t v; if (!r.value(v)) return false; out = v; return true; }
        case U16: { uint16_t v; if (!r.value(v)) return false; out = v; return true; }
        case I16: { int16_t v; if (!r.value(v)) return false; out = v; return true; }
        case U32: { uint32_t v; if (!r.value(v)) return false; out = v; return true; }
        case I32: { int32_t v; if (!r.value(v)) return false; out = v; return true; }
        case F32: { float v; if (!r.value(v)) return false; out = v; return true; }
        case BOOL: { uint8_t v; if (!r.value(v)) return false; out = v != 0; return true; }
        case STR: { std::string v; if (!r.string(v)) return false; out = std::move(v); return true; }
        case U64: { uint64_t v; if (!r.value(v)) return false; out = v; return true; }
        case I64: { int64_t v; if (!r.value(v)) return false; out = v; return true; }
        case F64: { double v; if (!r.value(v)) return false; out = v; return true; }
        case ARR: {
            if (depth > 2) return false;
            uint32_t elementType = 0;
            uint64_t count = 0;
            if (!r.value(elementType) || !r.value(count) || count > kMaxGgufCount) return false;
            out = Json::array();
            for (uint64_t i = 0; i < count; ++i) {
                Json element;
                if (!readGgufValue(r, elementType, element, depth + 1)) return false;
                if (i < 16) out.push_back(std::move(element));  // keep only a sample: vocabularies are huge
            }
            return true;
        }
        default: return false;
    }
}

}  // namespace

const TensorInfo* CheckpointHeader::find(const std::string& name) const {
    const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const TensorInfo& t) { return t.name == name; });
    return it == tensors.end() ? nullptr : &*it;
}

bool CheckpointHeader::hasPrefix(const std::string& prefix) const {
    return std::any_of(tensors.begin(), tensors.end(), [&](const TensorInfo& t) { return t.name.rfind(prefix, 0) == 0; });
}

bool CheckpointHeader::hasSubstring(const std::string& part) const {
    return std::any_of(tensors.begin(), tensors.end(), [&](const TensorInfo& t) { return t.name.find(part) != std::string::npos; });
}

std::optional<CheckpointHeader> readSafetensorsHeader(const fs::path& file, std::string* error) {
    std::error_code ec;
    const uint64_t size = fs::file_size(file, ec);
    if (ec) {
        setError(error, "cannot read " + file.string());
        return std::nullopt;
    }
    std::ifstream in(file, std::ios::binary);
    uint64_t headerLength = 0;
    Reader reader(in);
    if (size < 10 || !reader.value(headerLength)) {
        setError(error, "file is too short to be a safetensors file");
        return std::nullopt;
    }
    if (headerLength == 0 || headerLength > kMaxSafetensorsHeader || headerLength > size - 8) {
        setError(error, "invalid safetensors header length");
        return std::nullopt;
    }
    std::string text(static_cast<size_t>(headerLength), '\0');
    if (!reader.read(text.data(), text.size())) {
        setError(error, "truncated safetensors header");
        return std::nullopt;
    }
    const Json doc = Json::parse(text, nullptr, false);
    if (!doc.is_object()) {
        setError(error, "safetensors header is not valid JSON");
        return std::nullopt;
    }
    CheckpointHeader header;
    header.format = CheckpointHeader::Format::Safetensors;
    header.fileSize = size;
    const uint64_t dataSize = size - 8 - headerLength;
    for (const auto& [key, value] : doc.items()) {
        if (key == "__metadata__") {
            if (value.is_object()) header.metadata = value;
            continue;
        }
        if (!value.is_object() || !value.contains("dtype") || !value.contains("shape") || !value.contains("data_offsets")) {
            setError(error, "tensor '" + key + "' is incomplete");
            return std::nullopt;
        }
        const auto& offsets = value["data_offsets"];
        if (!offsets.is_array() || offsets.size() != 2 || !offsets[0].is_number_unsigned() || !offsets[1].is_number_unsigned()) {
            setError(error, "tensor '" + key + "' has invalid offsets");
            return std::nullopt;
        }
        const uint64_t begin = offsets[0].get<uint64_t>(), end = offsets[1].get<uint64_t>();
        if (end < begin || end > dataSize) {
            setError(error, "tensor '" + key + "' lies outside the file");
            return std::nullopt;
        }
        TensorInfo t;
        t.name = key;
        t.dtype = value["dtype"].get<std::string>();
        t.shape = value["shape"].get<std::vector<int64_t>>();
        t.bytes = end - begin;
        t.offset = 8 + headerLength + begin;
        header.tensors.push_back(std::move(t));
    }
    return header;
}

std::optional<CheckpointHeader> readGgufHeader(const fs::path& file, std::string* error) {
    std::error_code ec;
    const uint64_t size = fs::file_size(file, ec);
    if (ec) {
        setError(error, "cannot read " + file.string());
        return std::nullopt;
    }
    std::ifstream in(file, std::ios::binary);
    Reader r(in);
    char magic[4] = {};
    uint32_t version = 0;
    uint64_t tensorCount = 0, kvCount = 0;
    if (!r.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        setError(error, "not a GGUF file");
        return std::nullopt;
    }
    if (!r.value(version) || (version != 2 && version != 3)) {
        setError(error, "unsupported GGUF version");
        return std::nullopt;
    }
    if (!r.value(tensorCount) || !r.value(kvCount) || tensorCount > kMaxGgufCount || kvCount > kMaxGgufCount) {
        setError(error, "invalid GGUF header");
        return std::nullopt;
    }
    CheckpointHeader header;
    header.format = CheckpointHeader::Format::Gguf;
    header.fileSize = size;
    for (uint64_t i = 0; i < kvCount; ++i) {
        std::string key;
        uint32_t type = 0;
        Json value;
        if (!r.string(key) || !r.value(type) || !readGgufValue(r, type, value)) {
            setError(error, "invalid GGUF metadata");
            return std::nullopt;
        }
        header.metadata[key] = std::move(value);
    }
    for (uint64_t i = 0; i < tensorCount; ++i) {
        TensorInfo t;
        uint32_t dims = 0, type = 0;
        uint64_t offset = 0;
        if (!r.string(t.name) || !r.value(dims) || dims > 8) {
            setError(error, "invalid GGUF tensor info");
            return std::nullopt;
        }
        t.shape.resize(dims);
        for (uint32_t d = 0; d < dims; ++d) {
            uint64_t dim = 0;
            if (!r.value(dim)) {
                setError(error, "invalid GGUF tensor info");
                return std::nullopt;
            }
            t.shape[d] = static_cast<int64_t>(dim);
        }
        if (!r.value(type) || !r.value(offset)) {
            setError(error, "invalid GGUF tensor info");
            return std::nullopt;
        }
        t.dtype = "GGML_" + std::to_string(type);
        header.tensors.push_back(std::move(t));
    }
    // sizes are not stored per tensor in GGUF; the weights are about the file size
    const uint64_t perTensor = header.tensors.empty() ? 0 : size / header.tensors.size();
    for (auto& t : header.tensors) t.bytes = perTensor;
    return header;
}

std::optional<CheckpointHeader> readCheckpointHeader(const fs::path& file, std::string* error) {
    const std::string ext = file.extension().string();
    if (ext == ".safetensors" || ext == ".sft") return readSafetensorsHeader(file, error);
    if (ext == ".gguf") return readGgufHeader(file, error);
    setError(error, "headers of " + ext + " files are not inspected");
    return std::nullopt;
}

}  // namespace gh::models
