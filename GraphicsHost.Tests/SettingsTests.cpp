#include "TestUtil.h"

#include "core/AtomicFile.h"
#include "core/Paths.h"
#include "core/Settings.h"

#include <catch2/catch_test_macros.hpp>

using gh::core::AppPaths;
using gh::core::Settings;
using gh::core::SettingsStore;
using gh::test::TempDir;
using nlohmann::json;

TEST_CASE("default settings validate", "[unit][settings]") {
    CHECK(gh::core::validate(Settings{}).empty());
    CHECK(Settings{}.safety.enabled == false);  // the safety filter is off by default
    CHECK(Settings{}.backend == "auto");
}

TEST_CASE("settings survive a JSON round trip and keep unknown keys", "[unit][settings]") {
    Settings s;
    s.backend = "cpu";
    s.threads = 12;
    s.server.port = 4321;
    s.server.lan = true;
    s.safety.enabled = true;
    s.unknown["futureKey"] = {{"a", 1}};

    const Settings back = gh::core::settingsFromJson(gh::core::toJson(s));
    CHECK(back.backend == "cpu");
    CHECK(back.threads == 12);
    CHECK(back.server.port == 4321);
    CHECK(back.server.lan);
    CHECK(back.safety.enabled);
    CHECK(back.unknown["futureKey"]["a"] == 1);
}

TEST_CASE("missing keys keep their defaults", "[unit][settings]") {
    const Settings s = gh::core::settingsFromJson(json::parse(R"({"backend":"vulkan"})"));
    CHECK(s.backend == "vulkan");
    CHECK(s.server.port == 1234);
    CHECK(s.logging.level == "info");
}

TEST_CASE("a wrong type for a known key is an error", "[unit][settings]") {
    CHECK_THROWS(gh::core::settingsFromJson(json::parse(R"({"threads":"many"})")));
}

TEST_CASE("validation reports every bad field", "[unit][settings]") {
    Settings s;
    s.backend = "cuda";
    s.threads = -1;
    s.server.port = 70000;
    s.logging.level = "loud";
    s.interfaceMode = "expert";
    s.gpuDevice = -2;
    const auto errors = gh::core::validate(s);
    REQUIRE(errors.size() == 6);
    CHECK(errors[0].field == "backend");
    CHECK(errors[1].field == "gpuDevice");
    CHECK(errors[2].field == "threads");
    CHECK(errors[3].field == "interfaceMode");
    CHECK(errors[4].field == "server.port");
    CHECK(errors[5].field == "logging.level");
}

TEST_CASE("migration 0 to 1 moves the server options", "[unit][settings][migration]") {
    json doc = json::parse(R"({"backend":"cpu","port":8080,"lan":true})");
    const int applied = gh::core::migrate(doc);
    CHECK(applied == 1);
    CHECK(doc["schemaVersion"] == gh::core::kSettingsSchemaVersion);
    CHECK(doc["server"]["port"] == 8080);
    CHECK(doc["server"]["lan"] == true);
    CHECK_FALSE(doc.contains("port"));
    CHECK(doc["backend"] == "cpu");
}

TEST_CASE("migrating a current document changes nothing", "[unit][settings][migration]") {
    json doc = gh::core::toJson(Settings{});
    const json before = doc;
    CHECK(gh::core::migrate(doc) == 0);
    CHECK(doc == before);
}

TEST_CASE("a document from a newer version is refused", "[unit][settings][migration]") {
    json doc = {{"schemaVersion", gh::core::kSettingsSchemaVersion + 1}};
    CHECK_THROWS_AS(gh::core::migrate(doc), std::runtime_error);
}

TEST_CASE("store: no file means defaults", "[unit][settings][store]") {
    TempDir dir;
    const auto result = SettingsStore(AppPaths(dir.path())).load();
    CHECK_FALSE(result.migrated);
    CHECK_FALSE(result.warning.has_value());
    CHECK(result.settings.backend == "auto");
}

TEST_CASE("store: saved settings load again", "[unit][settings][store]") {
    TempDir dir;
    SettingsStore store{AppPaths(dir.path())};
    Settings s;
    s.backend = "cpu";
    s.threads = 6;
    store.save(s);
    const auto loaded = store.load();
    CHECK(loaded.settings.backend == "cpu");
    CHECK(loaded.settings.threads == 6);
    CHECK_FALSE(loaded.migrated);
}

TEST_CASE("store: saving invalid settings throws and leaves the file alone", "[unit][settings][store]") {
    TempDir dir;
    SettingsStore store{AppPaths(dir.path())};
    Settings s;
    store.save(s);
    const auto before = gh::test::readText(AppPaths(dir.path()).settingsFile());

    s.server.port = 0;
    CHECK_THROWS_AS(store.save(s), std::invalid_argument);
    CHECK(gh::test::readText(AppPaths(dir.path()).settingsFile()) == before);
}

TEST_CASE("store: an old file is migrated and a backup kept", "[unit][settings][store][migration]") {
    TempDir dir;
    AppPaths paths(dir.path());
    gh::test::writeText(paths.settingsFile(), R"({"backend":"cpu","port":9000})");

    const auto result = SettingsStore(paths).load();
    CHECK(result.migrated);
    CHECK(result.settings.server.port == 9000);
    CHECK(std::filesystem::exists(dir.path() / "settings.json.bak-v0"));
    // the upgraded file was written back
    CHECK(json::parse(gh::test::readText(paths.settingsFile()))["schemaVersion"] == gh::core::kSettingsSchemaVersion);
}

TEST_CASE("store: a corrupt file is set aside and defaults are used", "[unit][settings][store]") {
    TempDir dir;
    AppPaths paths(dir.path());
    gh::test::writeText(paths.settingsFile(), "{ this is not json");

    const auto result = SettingsStore(paths).load();
    CHECK(result.warning.has_value());
    CHECK(result.settings.backend == "auto");
    CHECK(std::filesystem::exists(dir.path() / "settings.json.invalid"));
    CHECK_FALSE(std::filesystem::exists(paths.settingsFile()));
}

TEST_CASE("atomic write replaces the content and leaves no temp file", "[unit][files]") {
    TempDir dir;
    const auto file = dir.path() / "sub" / "a.txt";
    gh::core::writeFileAtomic(file, "first");
    gh::core::writeFileAtomic(file, "second");
    CHECK(gh::core::readFile(file).value() == "second");
    CHECK_FALSE(std::filesystem::exists(dir.path() / "sub" / "a.txt.tmp"));
    CHECK_FALSE(gh::core::readFile(dir.path() / "missing.txt").has_value());
}

TEST_CASE("paths: ensureDirectories creates the whole layout", "[unit][paths]") {
    TempDir dir;
    AppPaths paths(dir.path() / "home");
    paths.ensureDirectories();
    for (const auto& p : {paths.root(), paths.presetsDir(), paths.runtimesDir(), paths.logsDir(), paths.outputsDir(),
                          paths.defaultModelsDir()}) {
        CHECK(std::filesystem::is_directory(p));
    }
}

TEST_CASE("settings: theme and accent are validated and round-trip", "[unit][settings]") {
    gh::core::Settings s;
    CHECK(s.theme == "system");
    CHECK(s.accent.empty());
    CHECK(gh::core::validate(s).empty());
    s.theme = "dark";
    s.accent = "#2B6cb0";
    CHECK(gh::core::validate(s).empty());
    const auto back = gh::core::settingsFromJson(gh::core::toJson(s));
    CHECK(back.theme == "dark");
    CHECK(back.accent == "#2B6cb0");
    s.theme = "neon";
    s.accent = "blue";
    const auto errors = gh::core::validate(s);
    CHECK(errors.size() == 2);
}
