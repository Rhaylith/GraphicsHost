#include "graph/OutputCache.h"

#include "core/Sha256.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace gh::graph {

using Json = nlohmann::json;

namespace {

// A literal as the node will see it: numbers take the input's type so 5 and 5.0 give one key.
Json normalised(const InputDef& in, const Json& v) {
    if (v.is_number() && in.type == ValueType::Int) return static_cast<int64_t>(std::llround(v.get<double>()));
    if (v.is_number() && in.type == ValueType::Float) return v.get<double>();
    return v;
}

}  // namespace

std::string nodeCacheKey(const JobNode& node, const Registry& registry, const std::map<std::string, Upstream>& upstream) {
    const NodeDef* def = registry.find(node.type);
    if (!def) return "";
    std::map<std::string, Json> literals;
    Json doc = Json::object();
    doc["type"] = node.type;
    Json lit = Json::object();
    for (const InputDef& in : def->inputs()) {
        const auto given = node.inputs.find(in.name);
        if (given != node.inputs.end()) {
            if (const Json* v = std::get_if<Json>(&given->second)) {
                lit[in.name] = normalised(in, *v);
                literals[in.name] = lit[in.name];
            }
        } else if (in.widget && !in.def.is_null() && !upstream.count(in.name)) {
            lit[in.name] = normalised(in, in.def);
            literals[in.name] = lit[in.name];
        }
    }
    for (const auto& [name, v] : node.inputs)
        if (!def->findInput(name)) lit[name] = std::get_if<Json>(&v) ? *std::get_if<Json>(&v) : Json();   // validator reports it; keep the key distinct
    doc["literals"] = lit;
    Json up = Json::object();
    for (const auto& [name, u] : upstream) up[name] = Json::array({u.key, u.output});
    doc["upstream"] = up;
    if (def->fingerprintFn()) {
        const std::string extra = def->fingerprintFn()(literals);
        if (extra == "*") return "";
        doc["fingerprint"] = extra;
    }
    return core::sha256Hex(doc.dump());
}

std::optional<CachePolicy> CachePolicy::parse(const std::string& text) {
    CachePolicy p;
    if (text == "last") return p;
    if (text == "none") {
        p.kind = Kind::None;
        return p;
    }
    const auto colon = text.find(':');
    if (colon == std::string::npos) return std::nullopt;
    const std::string kind = text.substr(0, colon);
    char* end = nullptr;
    const std::string number = text.substr(colon + 1);
    const double n = std::strtod(number.c_str(), &end);
    if (number.empty() || *end != '\0' || n <= 0) return std::nullopt;
    if (kind == "lru") {
        p.kind = Kind::Lru;
        p.entries = static_cast<int>(n);
        return p;
    }
    if (kind == "ram") {
        p.kind = Kind::Ram;
        p.freeRamGb = n;
        return p;
    }
    return std::nullopt;
}

std::string CachePolicy::toString() const {
    char buf[32];
    switch (kind) {
        case Kind::Lru: return "lru:" + std::to_string(entries);
        case Kind::Ram:
            std::snprintf(buf, sizeof buf, "ram:%g", freeRamGb);
            return buf;
        case Kind::None: return "none";
        default: return "last";
    }
}

uint64_t OutputCache::systemFreeRamBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    if (GlobalMemoryStatusEx(&status)) return status.ullAvailPhys;
#endif
    return UINT64_MAX;
}

OutputCache::OutputCache(CachePolicy policy, FreeRamFn freeRam) : policy_(policy), freeRam_(freeRam ? std::move(freeRam) : FreeRamFn(&OutputCache::systemFreeRamBytes)) {}

void OutputCache::setPolicy(CachePolicy policy) {
    std::lock_guard lock(mutex_);
    policy_ = policy;
    if (policy_.kind == CachePolicy::Kind::None) entries_.clear();
}

CachePolicy OutputCache::policy() const {
    std::lock_guard lock(mutex_);
    return policy_;
}

void OutputCache::beginJob() {
    std::lock_guard lock(mutex_);
    ++job_;
}

std::shared_ptr<const NodeOutputs> OutputCache::find(const std::string& key) {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(key);
    if (it == entries_.end()) return nullptr;
    it->second.lastUse = ++clock_;
    it->second.usedInJob = job_;
    return it->second.outputs;
}

void OutputCache::put(const std::string& key, NodeOutputs outputs) {
    std::lock_guard lock(mutex_);
    if (policy_.kind == CachePolicy::Kind::None || key.empty()) return;
    Entry e;
    e.bytes = 0;
    for (const Value& v : outputs) e.bytes += v.approxBytes();
    e.outputs = std::make_shared<const NodeOutputs>(std::move(outputs));
    e.lastUse = ++clock_;
    e.usedInJob = job_;
    entries_[key] = std::move(e);
}

void OutputCache::endJob() {
    std::lock_guard lock(mutex_);
    evict();
}

void OutputCache::evict() {
    using Kind = CachePolicy::Kind;
    switch (policy_.kind) {
        case Kind::None: entries_.clear(); return;
        case Kind::Last:
            for (auto it = entries_.begin(); it != entries_.end();) it = it->second.usedInJob == job_ ? std::next(it) : entries_.erase(it);
            return;
        case Kind::Lru:
        case Kind::Ram: break;
    }
    const auto oldest = [&] {
        return std::min_element(entries_.begin(), entries_.end(), [](const auto& a, const auto& b) { return a.second.lastUse < b.second.lastUse; });
    };
    if (policy_.kind == Kind::Lru) {
        while (static_cast<int>(entries_.size()) > policy_.entries) entries_.erase(oldest());
    } else {
        const uint64_t threshold = static_cast<uint64_t>(policy_.freeRamGb * 1024.0 * 1024.0 * 1024.0);
        while (!entries_.empty() && freeRam_() < threshold) entries_.erase(oldest());
    }
}

void OutputCache::clear() {
    std::lock_guard lock(mutex_);
    entries_.clear();
}

size_t OutputCache::size() const {
    std::lock_guard lock(mutex_);
    return entries_.size();
}

size_t OutputCache::bytes() const {
    std::lock_guard lock(mutex_);
    size_t total = 0;
    for (const auto& [key, e] : entries_) total += e.bytes;
    return total;
}

}  // namespace gh::graph
