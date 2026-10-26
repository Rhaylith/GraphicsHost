#pragma once

#include "models/MemoryManager.h"

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace gh::models {

// A part of a model that has been opened: its weights live in video memory or in RAM. Implemented by the wrapper around the
// runtime's component handles (and by test doubles).
class ComponentObject {
public:
    virtual ~ComponentObject() = default;
    virtual uint64_t weightsBytes() const = 0;
    virtual void moveToRam() = 0;                       // weights to system RAM; the VRAM copy is released
    virtual void moveToVram() = 0;                      // weights back to the GPU (all of them, or `fraction` of them)
    virtual bool supportsPartial() const { return false; }
    virtual void setResidentFraction(double fraction) { (void)fraction; }
};

// What identifies a component: the file, the part of it (diffusion, text-encoder, vae, ...) and the options that change the
// weights in memory (precision). Patches are not part of the identity: the same component serves any patch list.
struct ComponentKey {
    std::string file;
    std::string part;
    std::string options;
    std::string id() const { return file + "|" + part + "|" + options; }
    bool operator<(const ComponentKey& o) const { return id() < o.id(); }
    bool operator==(const ComponentKey& o) const { return id() == o.id(); }
};

class MemoryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ComponentStats {
    ComponentKey key;
    uint64_t bytes = 0;
    Residency residency = Residency::Vram;
    double residentFraction = 1.0;
    int users = 0;                    // leases held
    bool external = false;            // registered by someone else (ModelManager): counted, never moved
    std::chrono::steady_clock::time_point lastUse{};
};

// The components that are open, shared by every node that loads the same file with the same options (workflow plan,
// section 6.3). Before a component is opened or brought back to the GPU the MemoryManager says what has to leave; the cache
// does that: least recently used components move to RAM or are closed. A component that a node is using (a lease) is never
// moved. Idle components close after their time to live.
class ComponentCache {
public:
    using Clock = std::function<std::chrono::steady_clock::time_point()>;
    using Loader = std::function<std::shared_ptr<ComponentObject>()>;

    explicit ComponentCache(MemoryManager& memory, Clock clock = {});

    // Keeps a component in VRAM while it exists.
    class Lease {
    public:
        Lease() = default;
        Lease(Lease&& other) noexcept : cache_(other.cache_), id_(std::move(other.id_)), object_(std::move(other.object_)) { other.cache_ = nullptr; }
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() { release(); }
        ComponentObject& object() const { return *object_; }
        explicit operator bool() const { return object_ != nullptr; }
        void release();

    private:
        friend class ComponentCache;
        Lease(ComponentCache* cache, std::string id, std::shared_ptr<ComponentObject> object) : cache_(cache), id_(std::move(id)), object_(std::move(object)) {}
        ComponentCache* cache_ = nullptr;
        std::string id_;
        std::shared_ptr<ComponentObject> object_;
    };

    struct Use {
        uint64_t expectedBytes = 0;        // weights, when the component is not open yet (from the file header)
        uint64_t activationBytes = 0;      // what the call that follows needs
        int idleMinutes = 10;              // close after this long without use; 0 = never
        bool canSplit = false;              // the component may keep only part of its weights in VRAM (needed before it is open)
    };

    // Opens the component (calling `load` if it is not open) and brings it into VRAM, making room first.
    // Throws MemoryError when it cannot be made to fit.
    Lease acquire(const ComponentKey& key, const Use& use, const Loader& load);

    // Registers a component somebody else owns (the model the Generate path loaded) so the memory plan counts it. Never moved.
    void adopt(const ComponentKey& key, uint64_t bytes);
    void forget(const ComponentKey& key);
    // Closes the unused components of a file (ModelManager asks before it opens that file itself, so a file is never held twice).
    size_t closeFile(const std::string& file);
    bool holdsFile(const std::string& file) const;

    // Closes components that have been idle longer than their time to live. Returns how many.
    size_t closeIdle();
    // The "free memory" actions: weights to RAM / close everything that is not in use.
    size_t offloadAll();
    size_t closeAll();

    std::vector<ComponentStats> stats() const;
    uint64_t bytesIn(Residency residency) const;

    MemoryManager& memory() { return memory_; }

private:
    struct Entry {
        ComponentKey key;
        std::shared_ptr<ComponentObject> object;   // null for adopted components
        uint64_t bytes = 0;
        Residency residency = Residency::Vram;
        double fraction = 1.0;
        int users = 0;
        int idleMinutes = 10;
        bool external = false;
        std::chrono::steady_clock::time_point lastUse{};
    };
    void release(const std::string& id);
    void apply(const MemoryPlan& plan);
    std::chrono::steady_clock::time_point now() const;

    MemoryManager& memory_;
    Clock clock_;
    mutable std::recursive_mutex mutex_;   // recursive: the memory budget the plan asks for may look at the cache
    std::map<std::string, Entry> entries_;
};

}  // namespace gh::models
