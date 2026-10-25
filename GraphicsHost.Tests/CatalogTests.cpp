#include "TestUtil.h"

#include "core/Paths.h"
#include "models/Catalog.h"

#include <catch2/catch_test_macros.hpp>

#include <set>

using namespace gh::models;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

json entry(const std::string& key, json overrides = json::object()) {
    json e = {{"key", key},
              {"name", "Model " + key},
              {"kind", "checkpoint"},
              {"family", "SD 1.x"},
              {"repo", "someone/some-repo"},
              {"file", key + ".safetensors"},
              {"sizeBytes", 1000},
              {"sha256", std::string(64, 'a')},
              {"tags", {"general"}}};
    for (const auto& [k, v] : overrides.items()) e[k] = v;
    return e;
}

json doc(std::vector<json> entries) {
    return {{"schemaVersion", 1}, {"entries", entries}};
}

}  // namespace

TEST_CASE("catalog: the shipped catalog loads without a single warning", "[unit][catalog]") {
    const auto file = gh::core::AppPaths::executableDir() / "data" / "catalog.json";
    REQUIRE(fs::exists(file));
    const Catalog catalog = Catalog::load({file});
    for (const auto& w : catalog.warnings()) WARN(w);
    CHECK(catalog.warnings().empty());
    CHECK(catalog.entries().size() >= 6);

    std::set<std::string> keys;
    for (const auto& e : catalog.entries()) {
        CHECK(keys.insert(e.key).second);               // keys are unique
        CHECK(e.sha256.size() == 64);
        CHECK(e.sizeBytes > 1000000);
        CHECK(e.url().rfind("https://huggingface.co/", 0) == 0);
        CHECK(e.url().find(e.revision) != std::string::npos);   // pinned to the commit the hash was read from
        CHECK_FALSE(e.license.empty());
        CHECK_FALSE(e.family.empty());
        for (const auto& c : e.companions) CHECK(catalog.find(c.key) != nullptr);  // companions exist
    }
    CHECK(catalog.find("sd15") != nullptr);
    CHECK(catalog.find("sdxl") != nullptr);
}

TEST_CASE("catalog: entry fields and derived values", "[unit][catalog]") {
    const auto c = Catalog::fromJson(doc({entry("a", {{"revision", "abc123"}, {"gated", true}, {"minVramGb", 8}, {"defaults", {{"request", {{"steps", 4}}}}}})}));
    REQUIRE(c.entries().size() == 1);
    const auto& e = c.entries()[0];
    CHECK(e.url() == "https://huggingface.co/someone/some-repo/resolve/abc123/a.safetensors");
    CHECK(e.relativePath() == fs::path("someone") / "some-repo" / "a.safetensors");
    CHECK(e.gated);
    CHECK(e.minVramGb == 8);
    CHECK(e.defaults["request"]["steps"] == 4);
    const auto noRevision = Catalog::fromJson(doc({entry("b")}));
    CHECK(noRevision.entries()[0].url().find("/resolve/main/") != std::string::npos);
}

TEST_CASE("catalog: bad entries are skipped with a reason, good ones stay", "[unit][catalog]") {
    std::vector<std::string> warnings;
    const auto c = Catalog::fromJson(doc({entry("good"),
                                          entry("nohash", {{"sha256", "short"}}),
                                          entry("nosize", {{"sizeBytes", 0}}),
                                          entry("badrepo", {{"repo", "no-slash"}}),
                                          entry("traversal", {{"file", "../../evil.safetensors"}}),
                                          entry("badkind", {{"kind", "model"}}),
                                          json{{"name", "no key"}},
                                          json("not an object")}),
                                     &warnings);
    REQUIRE(c.entries().size() == 1);
    CHECK(c.entries()[0].key == "good");
    CHECK(warnings.size() == 7);
    CHECK(warnings[0].find("sha256") != std::string::npos);
    CHECK(Catalog::fromJson(json::object()).warnings().size() == 1);
}

TEST_CASE("catalog: a later file adds entries and replaces those with the same key", "[unit][catalog]") {
    gh::test::TempDir dir;
    gh::test::writeText(dir.path() / "shipped.json", doc({entry("a", {{"name", "Shipped A"}}), entry("b")}).dump());
    gh::test::writeText(dir.path() / "user.json", doc({entry("a", {{"name", "My A"}}), entry("c")}).dump());
    const auto c = Catalog::load({dir.path() / "shipped.json", dir.path() / "user.json", dir.path() / "absent.json"});
    CHECK(c.warnings().empty());
    REQUIRE(c.entries().size() == 3);
    CHECK(c.find("a")->name == "My A");
    CHECK(c.find("b") != nullptr);
    CHECK(c.find("c") != nullptr);
    CHECK(c.find("d") == nullptr);

    gh::test::writeText(dir.path() / "broken.json", "{ nope");
    const auto withBroken = Catalog::load({dir.path() / "shipped.json", dir.path() / "broken.json"});
    CHECK(withBroken.entries().size() == 2);
    CHECK(withBroken.warnings().size() == 1);
}

TEST_CASE("catalog: search by text, family, kind and recommendation", "[unit][catalog]") {
    const auto c = Catalog::fromJson(doc({entry("sd15", {{"recommended", true}, {"tags", {"general", "small"}}}),
                                          entry("sdxl", {{"family", "SDXL"}, {"name", "Stable Diffusion XL"}, {"recommended", true}}),
                                          entry("turbo", {{"family", "SDXL"}, {"tags", {"fast"}}}),
                                          entry("vae1", {{"kind", "vae"}, {"tags", {"vae"}}})}));
    CHECK(c.search({}).size() == 4);
    CHECK(c.search({"", "SDXL", "", false}).size() == 2);
    CHECK(c.search({"", "sdxl", "", false}).size() == 2);        // family is case-insensitive
    CHECK(c.search({"fast", "", "", false}).size() == 1);
    CHECK(c.search({"STABLE diffusion xl", "", "", false}).size() == 1);
    CHECK(c.search({"", "", "vae", false}).size() == 1);
    CHECK(c.search({"", "", "", true}).size() == 2);
    CHECK(c.search({"small", "SDXL", "", false}).empty());
}

TEST_CASE("catalog: companions come along when marked auto", "[unit][catalog]") {
    const auto c = Catalog::fromJson(doc({entry("xl", {{"family", "SDXL"}, {"companions", {{{"role", "vae"}, {"key", "xlvae"}, {"auto", true}}, {{"role", "vae"}, {"key", "optional"}}, {{"role", "vae"}, {"key", "ghost"}, {"auto", true}}}}}),
                                          entry("xlvae", {{"kind", "vae"}}),
                                          entry("optional", {{"kind", "vae"}})}));
    const auto* xl = c.find("xl");
    REQUIRE(xl != nullptr);
    const auto automatic = c.withCompanions(*xl);
    REQUIRE(automatic.size() == 2);   // itself + the auto companion that exists; "ghost" is ignored
    CHECK(automatic[0]->key == "xl");
    CHECK(automatic[1]->key == "xlvae");
    CHECK(c.withCompanions(*xl, false).size() == 3);
}
