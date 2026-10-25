#include "SafetensorsFixture.h"
#include "TestUtil.h"

#include "models/ModelLibrary.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <thread>

using namespace gh::models;
using namespace gh::test::st;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

std::vector<FakeTensor> vaeTensors() {
    return {{"encoder.conv_in.weight", {128, 3, 3, 3}}, {"decoder.conv_in.weight", {512, 4, 3, 3}}};
}

std::vector<FakeTensor> sdxlTensors() {
    auto t = sd1Tensors();
    t.push_back({"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight", {64, 64}});
    return t;
}

const ModelCard* byFile(const std::vector<ModelCard>& cards, const std::string& file) {
    for (const auto& c : cards)
        if (c.file == file) return &c;
    return nullptr;
}

}  // namespace

TEST_CASE("library: scan finds every kind in a nested layout", "[unit][library]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "stabilityai" / "sdxl" / "base.safetensors", sdxlTensors());
    writeSafetensors(dir.path() / "runwayml" / "sd15" / "v1-5.safetensors", sd1Tensors());
    writeSafetensors(dir.path() / "vae" / "ft-mse.safetensors", vaeTensors());
    gh::test::writeText(dir.path() / "old.ckpt", "pickle");
    gh::test::writeText(dir.path() / "readme.txt", "x");
    gh::test::writeText(dir.path() / "notes.json", "{}");

    ModelLibrary library({dir.path()});
    const auto cards = library.scan();
    REQUIRE(cards.size() == 4);
    CHECK(byFile(cards, "base.safetensors")->familyName == "SDXL");
    CHECK(byFile(cards, "v1-5.safetensors")->familyName == "SD 1.x");
    CHECK(byFile(cards, "ft-mse.safetensors")->kind == Kind::Vae);
    CHECK(byFile(cards, "old.ckpt")->kind == Kind::Checkpoint);
    CHECK(library.ofKind(Kind::Checkpoint).size() == 3);
    CHECK(library.ofKind(Kind::Vae).size() == 1);
    // sorted by path
    for (size_t i = 1; i < cards.size(); ++i) CHECK(cards[i - 1].path < cards[i].path);
}

TEST_CASE("library: missing folders and empty folders are not errors", "[unit][library]") {
    gh::test::TempDir dir;
    ModelLibrary library({dir.path() / "nope", dir.path()});
    CHECK(library.scan().empty());
    ModelLibrary none;
    CHECK(none.scan().empty());
}

TEST_CASE("library: unchanged files come from the cache, changed files are read again", "[unit][library]") {
    gh::test::TempDir dir;
    const auto file = dir.path() / "m.safetensors";
    writeSafetensors(file, sd1Tensors());
    ModelLibrary library({dir.path()});
    CHECK(library.scan()[0].familyName == "SD 1.x");

    // overwrite with an SDXL checkpoint of another size: the cache entry is stale and must not be used
    writeSafetensors(file, sdxlTensors());
    CHECK(library.scan()[0].familyName == "SDXL");
    // repeated scans stay consistent
    CHECK(library.scan()[0].familyName == "SDXL");
}

TEST_CASE("library: a VAE next to a checkpoint is linked to it", "[unit][library][companions]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "dream.safetensors", sd1Tensors());
    writeSafetensors(dir.path() / "dream.vae.safetensors", vaeTensors());
    writeSafetensors(dir.path() / "other.safetensors", sd1Tensors());
    writeSafetensors(dir.path() / "unrelated-vae.safetensors", vaeTensors());

    ModelLibrary library({dir.path()});
    const auto cards = library.scan();
    CHECK(byFile(cards, "dream.safetensors")->vae == "dream.vae.safetensors");
    CHECK(byFile(cards, "dream.safetensors")->companions == std::vector<std::string>{"dream.vae.safetensors"});
    CHECK(byFile(cards, "other.safetensors")->vae.empty());  // "unrelated-vae" does not start with "other"
}

TEST_CASE("library: companions recorded by a download are linked, and the source is shown", "[unit][library][companions]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "sdxl-base.safetensors", sdxlTensors());
    writeSafetensors(dir.path() / "sdxl_vae.safetensors", vaeTensors());
    gh::test::writeText(sourceSidecarFor(dir.path() / "sdxl-base.safetensors"),
                        R"({"repo":"stabilityai/stable-diffusion-xl-base-1.0","file":"sd_xl_base_1.0.safetensors","revision":"abc123",
                            "companions":[{"file":"sdxl_vae.safetensors","role":"vae"},{"file":"missing.safetensors","role":"vae"}]})");
    ModelLibrary library({dir.path()});
    const auto cards = library.scan();
    const auto* card = byFile(cards, "sdxl-base.safetensors");
    REQUIRE(card != nullptr);
    CHECK(card->vae == "sdxl_vae.safetensors");
    CHECK(card->companions == std::vector<std::string>{"sdxl_vae.safetensors"});  // the missing file is ignored
    CHECK(card->source["revision"] == "abc123");
    CHECK(toJson(*card)["source"]["repo"] == "stabilityai/stable-diffusion-xl-base-1.0");
    CHECK(byFile(cards, "sdxl_vae.safetensors")->source.is_null());
}

TEST_CASE("library: names resolve to checkpoints", "[unit][library][resolve]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "a" / "dreamshaper_8.safetensors", sd1Tensors());
    writeSafetensors(dir.path() / "dreamshaper_xl.safetensors", sdxlTensors());
    writeSafetensors(dir.path() / "realistic.safetensors", sd1Tensors());
    writeSafetensors(dir.path() / "realistic.vae.safetensors", vaeTensors());

    ModelLibrary library({dir.path()});
    using S = ModelLibrary::Resolution::Status;
    CHECK(library.resolve("realistic").status == S::Found);
    CHECK(library.resolve("realistic").card->vae == "realistic.vae.safetensors");  // the library's card, VAE included
    CHECK(library.resolve("realistic.safetensors").status == S::Found);
    CHECK(library.resolve("dreamshaper_8").card->file == "dreamshaper_8.safetensors");
    CHECK(library.resolve("dreamshaper_x").card->file == "dreamshaper_xl.safetensors");  // unique prefix
    CHECK(library.resolve("dream").status == S::Ambiguous);
    CHECK(library.resolve("dream").problem.find("matches 2") != std::string::npos);
    CHECK(library.resolve("realistic.vae").status == S::NotFound);   // a VAE is not a checkpoint
    CHECK(library.resolve("nothing").status == S::NotFound);
    CHECK(library.resolve("nothing").problem.find(dir.path().string()) != std::string::npos);
    CHECK(library.resolve("").status == S::NotFound);

    // a path works even outside the folders
    gh::test::TempDir elsewhere;
    writeSafetensors(elsewhere.path() / "x.safetensors", sdxlTensors());
    const auto byPath = library.resolve((elsewhere.path() / "x.safetensors").string());
    REQUIRE(byPath.status == S::Found);
    CHECK(byPath.card->familyName == "SDXL");
}

TEST_CASE("library: the folder watcher reports new, changed and removed files", "[unit][library][watch]") {
    gh::test::TempDir dir;
    ModelLibrary library({dir.path()});
    std::atomic<int> calls{0};
    library.startWatching([&] { ++calls; });
    std::this_thread::sleep_for(200ms);  // let the watcher arm

    auto waitFor = [&](int n) {
        for (int i = 0; i < 50 && calls.load() < n; ++i) std::this_thread::sleep_for(100ms);
        return calls.load() >= n;
    };
    writeSafetensors(dir.path() / "new.safetensors", sd1Tensors());
    CHECK(waitFor(1));
    CHECK(library.scan().size() == 1);

    fs::remove(dir.path() / "new.safetensors");
    CHECK(waitFor(2));
    CHECK(library.scan().empty());

    library.stopWatching();
    const int before = calls.load();
    writeSafetensors(dir.path() / "later.safetensors", sd1Tensors());
    std::this_thread::sleep_for(800ms);
    CHECK(calls.load() == before);  // stopped watchers stay quiet
    library.stopWatching();         // harmless twice
}
