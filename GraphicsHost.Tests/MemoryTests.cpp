#include "TestUtil.h"
#include "core/Settings.h"
#include "engine/ModelManager.h"
#include "models/ComponentCache.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::models;
using namespace std::chrono_literals;
namespace chrono = std::chrono;

namespace {
constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kGiB = 1ull << 30;

chrono::steady_clock::time_point t(int seconds) { return chrono::steady_clock::time_point(chrono::seconds(seconds)); }

ComponentInfo info(const char* id, uint64_t bytes, int lastUse, Residency r = Residency::Vram, bool pinned = false) {
    return ComponentInfo{id, bytes, r, t(lastUse), pinned};
}

MemoryManager managerWith(MemoryMode mode, uint64_t vramFree, uint64_t ramFree = 64 * kGiB, uint64_t reserve = 512 * kMiB) {
    MemoryManager::Config c;
    c.mode = mode;
    c.reserveVramBytes = reserve;
    return MemoryManager(c, [=] { return MemoryBudget{24 * kGiB, vramFree, ramFree}; });
}

std::vector<std::string> actions(const MemoryPlan& p) {
    std::vector<std::string> out;
    for (const auto& a : p.actions) out.push_back((a.kind == MemoryAction::Kind::Offload ? "offload " : "close ") + a.id);
    return out;
}

// A component whose moves are recorded.
class Fake final : public ComponentObject {
public:
    Fake(uint64_t bytes, bool partial = false, std::vector<std::string>* log = nullptr, std::string name = "") : bytes_(bytes), partial_(partial), log_(log), name_(std::move(name)) {}
    uint64_t weightsBytes() const override { return bytes_; }
    void moveToRam() override { note("ram"); }
    void moveToVram() override { note("vram"); }
    bool supportsPartial() const override { return partial_; }
    void setResidentFraction(double f) override { fraction = f; note("fraction"); }
    double fraction = 1.0;

private:
    void note(const char* what) {
        if (log_) log_->push_back(name_ + " " + what);
    }
    uint64_t bytes_;
    bool partial_;
    std::vector<std::string>* log_;
    std::string name_;
};
}  // namespace

TEST_CASE("memory: mode names", "[unit][memory]") {
    for (const char* name : {"gpu-only", "high", "normal", "low", "stream"}) {
        const auto mode = parseMemoryMode(name);
        REQUIRE(mode.has_value());
        CHECK(std::string(toString(*mode)) == name);
    }
    CHECK_FALSE(parseMemoryMode("turbo").has_value());
    CHECK(std::string(toString(Residency::Ram)) == "ram");
}

TEST_CASE("memory plan: nothing moves while everything fits", "[unit][memory]") {
    const auto mm = managerWith(MemoryMode::Normal, 10 * kGiB);
    MemoryRequest r;
    r.target = "unet";
    r.weightsBytes = 3 * kGiB;
    r.activationBytes = 2 * kGiB;
    r.components = {info("vae", kGiB, 1)};
    const auto plan = mm.plan(r);
    CHECK(plan.fits);
    CHECK(plan.actions.empty());
    CHECK(plan.residentFraction == 1.0);
    CHECK(plan.neededBytes == 5 * kGiB);
    CHECK(plan.availableBytes == 10 * kGiB - 512 * kMiB);   // the reserve is kept free
}

TEST_CASE("memory plan: least recently used components leave first, to RAM when there is room", "[unit][memory]") {
    const auto mm = managerWith(MemoryMode::Normal, 4 * kGiB);
    MemoryRequest r;
    r.target = "unet";
    r.weightsBytes = 5 * kGiB;
    r.activationBytes = kGiB;
    r.components = {info("clip", kGiB, 30), info("vae", kGiB, 10), info("old-unet", 2 * kGiB, 5), info("busy", 8 * kGiB, 1, Residency::Vram, true), info("parked", 2 * kGiB, 2, Residency::Ram)};
    // needs 6 GiB, has 3.5: old-unet (2) and vae (1) leave, then it fits (6.5); the pinned and the parked ones are left alone
    const auto plan = mm.plan(r);
    CHECK(plan.fits);
    CHECK(actions(plan) == std::vector<std::string>{"offload old-unet", "offload vae"});
}

TEST_CASE("memory plan: gpu-only closes, and a short RAM makes normal close too", "[unit][memory]") {
    MemoryRequest r;
    r.target = "unet";
    r.weightsBytes = 5 * kGiB;
    r.activationBytes = kGiB;
    r.components = {info("clip", 2 * kGiB, 1), info("vae", kGiB, 2)};

    auto plan = managerWith(MemoryMode::GpuOnly, 4 * kGiB).plan(r);
    CHECK(plan.fits);
    CHECK(actions(plan) == std::vector<std::string>{"close clip", "close vae"});   // 3.5 + 2 = 5.5 < 6, so vae too

    plan = managerWith(MemoryMode::Normal, 4 * kGiB, 4 * kGiB + 512 * kMiB).plan(r);   // 4.5 GiB of RAM minus the 4 GiB margin: nothing fits there
    CHECK(actions(plan) == std::vector<std::string>{"close clip", "close vae"});

    plan = managerWith(MemoryMode::High, 4 * kGiB, 5 * kGiB).plan(r);     // high keeps only a 512 MiB margin
    CHECK(actions(plan) == std::vector<std::string>{"offload clip", "offload vae"});
}

TEST_CASE("memory plan: what does not fit is reported with the numbers", "[unit][memory]") {
    MemoryRequest r;
    r.target = "unet";
    r.weightsBytes = 12 * kGiB;
    r.activationBytes = 2 * kGiB;
    const auto plan = managerWith(MemoryMode::Normal, 6 * kGiB).plan(r);
    CHECK_FALSE(plan.fits);
    CHECK(plan.note.find("needs 14336 MiB") != std::string::npos);
    CHECK(plan.note.find("5632 MiB available") != std::string::npos);
    CHECK(plan.note.find("low or stream") != std::string::npos);
}

TEST_CASE("memory plan: low mode empties the device and may keep part of the weights in RAM", "[unit][memory]") {
    MemoryRequest r;
    r.target = "unet";
    r.weightsBytes = 10 * kGiB;
    r.activationBytes = 2 * kGiB;
    r.targetSupportsPartial = true;
    r.components = {info("clip", kGiB, 1), info("vae", kGiB, 2)};
    // 8 GiB free: low moves the others out even though they would not have to leave, then 10 + 2 > 9.5 → partial
    const auto plan = managerWith(MemoryMode::Low, 8 * kGiB).plan(r);
    CHECK(actions(plan) == std::vector<std::string>{"offload clip", "offload vae"});
    CHECK(plan.fits);
    CHECK(plan.residentFraction > 0.74);
    CHECK(plan.residentFraction < 0.76);   // (9.5 - 2) / 10
    CHECK(plan.note.find("stream") != std::string::npos);

    r.targetSupportsPartial = false;   // a component that cannot be split does not fit
    CHECK_FALSE(managerWith(MemoryMode::Low, 8 * kGiB).plan(r).fits);

    r.activationBytes = 20 * kGiB;     // the activations alone are too big
    r.targetSupportsPartial = true;
    CHECK_FALSE(managerWith(MemoryMode::Low, 8 * kGiB).plan(r).fits);
}

TEST_CASE("memory plan: stream mode keeps every weight in RAM", "[unit][memory]") {
    MemoryRequest r;
    r.target = "unet";
    r.weightsBytes = 20 * kGiB;
    r.activationBytes = 3 * kGiB;
    r.components = {info("clip", kGiB, 1)};
    const auto plan = managerWith(MemoryMode::Stream, 6 * kGiB).plan(r);
    CHECK(plan.fits);
    CHECK(plan.residentFraction == 0.0);
    CHECK(plan.neededBytes == 3 * kGiB);
    CHECK(actions(plan) == std::vector<std::string>{"offload clip"});
    r.activationBytes = 9 * kGiB;
    CHECK_FALSE(managerWith(MemoryMode::Stream, 6 * kGiB).plan(r).fits);
}

TEST_CASE("memory: activation estimates come from the measured constants", "[unit][memory]") {
    const uint64_t sd = MemoryManager::activationBytes("SD 1.x", 512, 512);
    const uint64_t xl = MemoryManager::activationBytes("SDXL", 1024, 1024);
    CHECK(sd > 1000 * kMiB);
    CHECK(sd < 3 * kGiB);
    CHECK(xl > 3 * kGiB);
    CHECK(MemoryManager::activationBytes("SD 1.x", 512, 512, 4) == 4 * sd);
}

namespace {
struct CacheRig {
    uint64_t totalVram = 10 * kGiB;
    chrono::steady_clock::time_point time = t(1000);
    ComponentCache* cache = nullptr;
    MemoryManager mm;
    ComponentCache components;
    std::vector<std::string> log;

    explicit CacheRig(MemoryMode mode = MemoryMode::Normal)
        : mm(
              [mode] {
                  MemoryManager::Config c;
                  c.mode = mode;
                  c.reserveVramBytes = 0;
                  return c;
              }(),
              [this] {
                  uint64_t used = cache ? cache->bytesIn(Residency::Vram) : 0;
                  return MemoryBudget{totalVram, totalVram > used ? totalVram - used : 0, 64 * kGiB};
              }),
          components(mm, [this] { return time; }) {
        cache = &components;
    }

    ComponentCache::Lease take(const char* file, uint64_t bytes, uint64_t activation = 0, bool partial = false, int idle = 10) {
        ComponentCache::Use use;
        use.expectedBytes = bytes;
        use.activationBytes = activation;
        use.idleMinutes = idle;
        use.canSplit = partial;
        return components.acquire({file, "diffusion", ""}, use, [&, bytes, partial, name = std::string(file)] { return std::make_shared<Fake>(bytes, partial, &log, name); });
    }
};
}  // namespace

TEST_CASE("component cache: a component is opened once and shared", "[unit][memory][components]") {
    CacheRig rig;
    int loads = 0;
    ComponentCache::Use use;
    use.expectedBytes = 2 * kGiB;
    const auto load = [&] {
        ++loads;
        return std::make_shared<Fake>(2 * kGiB);
    };
    auto a = rig.components.acquire({"a.safetensors", "unet", "fp16"}, use, load);
    auto b = rig.components.acquire({"a.safetensors", "unet", "fp16"}, use, load);
    CHECK(loads == 1);
    CHECK(&a.object() == &b.object());
    auto other = rig.components.acquire({"a.safetensors", "unet", "fp8"}, use, load);   // other options: another component
    CHECK(loads == 2);

    const auto stats = rig.components.stats();
    REQUIRE(stats.size() == 2);
    int users = 0;
    for (const auto& s : stats) users += s.users;
    CHECK(users == 3);
    a.release();
    b.release();
    other.release();
    for (const auto& s : rig.components.stats()) CHECK(s.users == 0);
    CHECK(rig.components.holdsFile("a.safetensors"));
    CHECK_FALSE(rig.components.holdsFile("b.safetensors"));
    CHECK(rig.components.bytesIn(Residency::Vram) == 4 * kGiB);
}

TEST_CASE("component cache: making room moves the least recently used component to RAM and brings it back", "[unit][memory][components]") {
    CacheRig rig;   // 10 GiB of VRAM
    { auto a = rig.take("a", 4 * kGiB); }
    rig.time += 10s;
    { auto b = rig.take("b", 4 * kGiB); }
    rig.time += 10s;
    CHECK(rig.log.empty());
    { auto c = rig.take("c", 4 * kGiB); }   // 12 GiB do not fit: a, the oldest, goes to RAM
    CHECK(rig.log == std::vector<std::string>{"a ram"});
    CHECK(rig.components.bytesIn(Residency::Ram) == 4 * kGiB);
    rig.time += 10s;
    { auto a = rig.take("a", 4 * kGiB); }   // back to the GPU: b is the oldest now
    CHECK(rig.log == std::vector<std::string>{"a ram", "b ram", "a vram"});
    std::map<std::string, Residency> where;
    for (const auto& s : rig.components.stats()) where[s.key.file] = s.residency;
    CHECK(where["a"] == Residency::Vram);
    CHECK(where["b"] == Residency::Ram);
    CHECK(where["c"] == Residency::Vram);
}

TEST_CASE("component cache: a component in use is never moved, and what cannot fit is refused", "[unit][memory][components]") {
    CacheRig rig;
    auto held = rig.take("held", 8 * kGiB);
    CHECK_THROWS_AS(rig.take("other", 4 * kGiB), MemoryError);   // 12 > 10 and nothing may leave
    CHECK(rig.log.empty());
    held.release();
    CHECK_NOTHROW(rig.take("other", 4 * kGiB));
    CHECK(rig.log == std::vector<std::string>{"held ram"});

    CacheRig gpuOnly(MemoryMode::GpuOnly);   // gpu-only never goes through RAM: the old component is closed
    { auto a = gpuOnly.take("a", 8 * kGiB); }
    { auto b = gpuOnly.take("b", 4 * kGiB); }
    CHECK(gpuOnly.log.empty());
    CHECK_FALSE(gpuOnly.components.holdsFile("a"));
    CHECK(gpuOnly.components.holdsFile("b"));
}

TEST_CASE("component cache: low mode keeps part of a big component in RAM", "[unit][memory][components]") {
    CacheRig rig(MemoryMode::Low);
    auto big = rig.take("big", 12 * kGiB, 2 * kGiB, true);   // 10 GiB of VRAM: (10 - 2) / 12 of it stays resident
    auto* fake = dynamic_cast<Fake*>(&big.object());
    REQUIRE(fake != nullptr);
    CHECK(fake->fraction > 0.66);
    CHECK(fake->fraction < 0.67);
    const auto stats = rig.components.stats();
    REQUIRE(stats.size() == 1);
    CHECK(stats[0].residentFraction == fake->fraction);
}

TEST_CASE("component cache: idle components close after their time to live", "[unit][memory][components]") {
    CacheRig rig;
    { auto a = rig.take("a", kGiB, 0, false, 5); }
    { auto b = rig.take("b", kGiB, 0, false, 0); }   // 0 = never
    auto busy = rig.take("busy", kGiB, 0, false, 1);
    rig.time += 4min;
    CHECK(rig.components.closeIdle() == 0);
    rig.time += 2min;
    CHECK(rig.components.closeIdle() == 1);   // a: idle 6 min; busy is in use; b never closes
    CHECK_FALSE(rig.components.holdsFile("a"));
    CHECK(rig.components.holdsFile("b"));
    CHECK(rig.components.holdsFile("busy"));
    rig.time += 3h;
    CHECK(rig.components.closeIdle() == 0);   // busy still in use
    busy.release();
    rig.time += 2min;
    CHECK(rig.components.closeIdle() == 1);
}

TEST_CASE("component cache: free memory actions leave components in use alone", "[unit][memory][components]") {
    CacheRig rig;
    { auto a = rig.take("a", kGiB); }
    { auto b = rig.take("b", kGiB); }
    auto busy = rig.take("busy", kGiB);
    CHECK(rig.components.offloadAll() == 2);
    CHECK(rig.components.bytesIn(Residency::Ram) == 2 * kGiB);
    CHECK(rig.components.bytesIn(Residency::Vram) == kGiB);
    CHECK(rig.components.closeAll() == 2);
    CHECK(rig.components.stats().size() == 1);
    busy.release();
    CHECK(rig.components.closeAll() == 1);
    CHECK(rig.components.stats().empty());
}

TEST_CASE("component cache: the Generate path's model is counted and a file is never held twice", "[unit][memory][components]") {
    CacheRig rig;
    rig.components.adopt({"model.safetensors", "model", ""}, 6 * kGiB);
    CHECK(rig.components.holdsFile("model.safetensors"));
    CHECK(rig.components.bytesIn(Residency::Vram) == 6 * kGiB);
    CHECK_THROWS_AS(rig.take("x", 6 * kGiB), MemoryError);   // 4 GiB are left, and the model is not the cache's to move
    CHECK(rig.components.closeFile("model.safetensors") == 0);   // external: not closed by the cache
    rig.components.forget({"model.safetensors", "model", ""});
    CHECK_FALSE(rig.components.holdsFile("model.safetensors"));

    { auto a = rig.take("graph.safetensors", kGiB); }
    CHECK(rig.components.closeFile("graph.safetensors") == 1);   // what ModelManager does before it opens that file itself
    CHECK_FALSE(rig.components.holdsFile("graph.safetensors"));
}

TEST_CASE("model manager: the loaded model is registered in the cache and replaced files are cleared", "[unit][memory][components]") {
    using namespace gh::engine;
    class StubModel final : public ILoadedModel {
    public:
        explicit StubModel(std::string path) { info_.modelPath = std::move(path); info_.family = "SD 1.x"; }
        const ModelInfo& info() const override { return info_; }
        GenerationResult generate(const GenerationRequest&, const ProgressSink&, std::stop_token) override { return {}; }
        ModelInfo info_;
    };
    class StubEngine final : public IImageEngine {
    public:
        std::string runtimeId() const override { return "stub"; }
        std::unique_ptr<ILoadedModel> load(const LoadOptions& o) override { return std::make_unique<StubModel>(o.modelPath); }
    };
    gh::test::TempDir dir;
    const auto file = dir.path() / "m.safetensors";
    std::ofstream(file, std::ios::binary) << std::string(4096, 'x');

    CacheRig rig;
    ModelManager manager([] { return std::make_shared<StubEngine>(); }, {}, std::chrono::seconds(0));
    manager.setComponentCache(&rig.components);
    rig.take(file.string().c_str(), kGiB);   // the graph side holds the same file (released at once)
    CHECK(rig.components.holdsFile(file.string()));

    LoadOptions o;
    o.modelPath = file.string();
    manager.load(o);
    bool external = false;
    for (const auto& s : rig.components.stats()) external = external || (s.external && s.key.file == file.string());
    CHECK(external);                                          // the Generate path's model is counted
    CHECK(rig.components.stats().size() == 1);                // and the unused graph copy of the same file was closed

    manager.unload();
    CHECK(rig.components.stats().empty());
}

TEST_CASE("settings: the memory section round trips and is validated", "[unit][memory][settings]") {
    gh::core::Settings s;
    CHECK(s.memory.mode == "normal");
    CHECK(s.memory.reserveVramMb == 512);
    CHECK(s.memory.asyncOffload);
    s.memory.mode = "low";
    s.memory.reserveVramMb = 1024;
    s.memory.asyncOffload = false;
    const auto back = gh::core::settingsFromJson(gh::core::toJson(s));
    CHECK(back.memory.mode == "low");
    CHECK(back.memory.reserveVramMb == 1024);
    CHECK_FALSE(back.memory.asyncOffload);
    CHECK(gh::core::validate(back).empty());

    s.memory.mode = "turbo";
    s.memory.reserveVramMb = -1;
    const auto errors = gh::core::validate(s);
    REQUIRE(errors.size() == 2);
    CHECK(errors[0].field == "memory.mode");
    CHECK(errors[1].field == "memory.reserveVramMb");
}
