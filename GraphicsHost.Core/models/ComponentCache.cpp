#include "models/ComponentCache.h"

#include "core/Log.h"

#include <algorithm>

namespace gh::models {

ComponentCache::Lease& ComponentCache::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        release();
        cache_ = other.cache_;
        id_ = std::move(other.id_);
        object_ = std::move(other.object_);
        other.cache_ = nullptr;
    }
    return *this;
}

void ComponentCache::Lease::release() {
    if (cache_) cache_->release(id_);
    cache_ = nullptr;
    object_.reset();
}

ComponentCache::ComponentCache(MemoryManager& memory, Clock clock) : memory_(memory), clock_(std::move(clock)) {}

std::chrono::steady_clock::time_point ComponentCache::now() const {
    return clock_ ? clock_() : std::chrono::steady_clock::now();
}

void ComponentCache::release(const std::string& id) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = entries_.find(id);
    if (it == entries_.end()) return;
    if (it->second.users > 0) --it->second.users;
    it->second.lastUse = now();
}

// Carries out what the memory plan asks (caller holds the lock).
void ComponentCache::apply(const MemoryPlan& plan) {
    for (const MemoryAction& action : plan.actions) {
        const auto it = entries_.find(action.id);
        if (it == entries_.end()) continue;
        Entry& e = it->second;
        if (action.kind == MemoryAction::Kind::Offload && e.object) {
            log::debug("component " + e.key.id() + " moves to RAM");
            e.object->moveToRam();
            e.residency = Residency::Ram;
        } else {
            log::debug("component " + e.key.id() + " is closed");
            entries_.erase(it);
        }
    }
}

ComponentCache::Lease ComponentCache::acquire(const ComponentKey& key, const Use& use, const Loader& load) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const std::string id = key.id();
    auto it = entries_.find(id);
    const bool open = it != entries_.end();
    const bool inVram = open && it->second.residency == Residency::Vram;

    MemoryRequest request;
    request.target = id;
    request.weightsBytes = inVram ? 0 : (open ? it->second.bytes : use.expectedBytes);
    request.activationBytes = use.activationBytes;
    request.targetSupportsPartial = use.canSplit || (open && it->second.object && it->second.object->supportsPartial());
    for (const auto& [eid, e] : entries_) {
        ComponentInfo info;
        info.id = eid;
        info.bytes = e.bytes;
        info.residency = e.residency;
        info.lastUse = e.lastUse;
        info.pinned = e.users > 0 || e.external;
        request.components.push_back(info);
    }
    const MemoryPlan plan = memory_.plan(request);
    if (!plan.fits) throw MemoryError(plan.note.empty() ? "not enough memory for " + key.file : plan.note + " (" + key.file + ")");
    apply(plan);

    it = entries_.find(id);   // the plan never touches the target, but apply() erased others
    if (it == entries_.end()) {
        std::shared_ptr<ComponentObject> object = load();
        if (!object) throw MemoryError("could not open " + key.file);
        Entry e;
        e.key = key;
        e.object = std::move(object);
        e.bytes = e.object->weightsBytes();
        e.residency = Residency::Vram;
        it = entries_.emplace(id, std::move(e)).first;
    } else if (it->second.residency != Residency::Vram) {
        it->second.object->moveToVram();
        it->second.residency = Residency::Vram;
    }
    Entry& e = it->second;
    if (plan.residentFraction < 1.0 && e.object && e.object->supportsPartial()) {
        e.object->setResidentFraction(plan.residentFraction);
        e.fraction = plan.residentFraction;
    } else if (e.fraction < 1.0 && e.object && e.object->supportsPartial()) {
        e.object->setResidentFraction(1.0);
        e.fraction = 1.0;
    }
    e.idleMinutes = use.idleMinutes;
    e.lastUse = now();
    ++e.users;
    return Lease(this, id, e.object);
}

void ComponentCache::adopt(const ComponentKey& key, uint64_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    Entry e;
    e.key = key;
    e.bytes = bytes;
    e.external = true;
    e.lastUse = now();
    entries_[key.id()] = std::move(e);
}

void ComponentCache::forget(const ComponentKey& key) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = entries_.find(key.id());
    if (it != entries_.end() && it->second.external) entries_.erase(it);
}

size_t ComponentCache::closeFile(const std::string& file) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    size_t closed = 0;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.key.file == file && it->second.users == 0 && !it->second.external) {
            it = entries_.erase(it);
            ++closed;
        } else {
            ++it;
        }
    }
    return closed;
}

bool ComponentCache::holdsFile(const std::string& file) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return std::any_of(entries_.begin(), entries_.end(), [&](const auto& kv) { return kv.second.key.file == file; });
}

size_t ComponentCache::closeIdle() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    size_t closed = 0;
    const auto t = now();
    for (auto it = entries_.begin(); it != entries_.end();) {
        const Entry& e = it->second;
        if (!e.external && e.users == 0 && e.idleMinutes > 0 && t - e.lastUse >= std::chrono::minutes(e.idleMinutes)) {
            log::info("component " + e.key.id() + " closed after " + std::to_string(e.idleMinutes) + " idle minutes");
            it = entries_.erase(it);
            ++closed;
        } else {
            ++it;
        }
    }
    return closed;
}

size_t ComponentCache::offloadAll() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    size_t moved = 0;
    for (auto& [id, e] : entries_) {
        if (e.external || e.users > 0 || e.residency != Residency::Vram || !e.object) continue;
        e.object->moveToRam();
        e.residency = Residency::Ram;
        ++moved;
    }
    return moved;
}

size_t ComponentCache::closeAll() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    size_t closed = 0;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (!it->second.external && it->second.users == 0) {
            it = entries_.erase(it);
            ++closed;
        } else {
            ++it;
        }
    }
    return closed;
}

std::vector<ComponentStats> ComponentCache::stats() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<ComponentStats> out;
    for (const auto& [id, e] : entries_) out.push_back({e.key, e.bytes, e.residency, e.fraction, e.users, e.external, e.lastUse});
    return out;
}

uint64_t ComponentCache::bytesIn(Residency residency) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& [id, e] : entries_)
        if (e.residency == residency) total += e.bytes;
    return total;
}

}  // namespace gh::models
