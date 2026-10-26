#include "TestUtil.h"

#include "core/AtomicFile.h"
#include "gallery/GalleryStore.h"

#include <catch2/catch_test_macros.hpp>

#include <sqlite3.h>

#include <regex>
#include <set>
#include <thread>

using namespace gh::engine;
using namespace gh::gallery;

namespace {

Image tinyImage(uint8_t shade = 100) {
    Image image;
    image.width = 8;
    image.height = 8;
    image.channels = 3;
    image.data.assign(8 * 8 * 3, shade);
    return image;
}

GenerationRequest request(const std::string& prompt, const std::string& format = "png") {
    GenerationRequest r;
    r.prompt = prompt;
    r.width = r.height = 8;
    r.steps = 10;
    r.cfgScale = 7.5f;
    r.format = format;
    return r;
}

ModelInfo model() {
    ModelInfo m;
    m.modelPath = "D:/models/v1-5-pruned-emaonly.safetensors";
    m.family = "SD 1.x";
    m.runtimeId = "cpu-avx2";
    return m;
}

}  // namespace

TEST_CASE("gallery: add writes the image, a sidecar and an index row", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    const auto entry = store.add(tinyImage(), request("a lighthouse"), 1234, model(), 1.5);

    CHECK(entry.id > 0);
    CHECK(std::regex_match(entry.path, std::regex(R"(\d{4}/\d{2}/\d{2}/\d{6}-1234-1\.png)")));
    CHECK(entry.model == "v1-5-pruned-emaonly.safetensors");
    CHECK(entry.seed == 1234);
    CHECK(entry.width == 8);

    const auto image = store.absolutePath(entry);
    REQUIRE(std::filesystem::exists(image));
    auto sidecar = image;
    sidecar.replace_extension(".json");
    REQUIRE(std::filesystem::exists(sidecar));

    // the sidecar and the metadata inside the PNG describe the same generation
    const auto fromSidecar = nlohmann::json::parse(gh::test::readText(sidecar));
    const auto bytes = gh::core::readFile(image).value();
    const auto embedded = nlohmann::json::parse(
        readImageMetadata(std::vector<uint8_t>(bytes.begin(), bytes.end())).value());
    CHECK(fromSidecar == embedded);
    CHECK(fromSidecar["request"]["prompt"] == "a lighthouse");
    CHECK(fromSidecar["request"]["seed"] == 1234);  // the seed of this image
    CHECK(fromSidecar["family"] == "SD 1.x");
    CHECK(fromSidecar["seconds"] == 1.5);
}

TEST_CASE("gallery: the recorded request reproduces the image", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    auto r = request("reproduce me");
    r.sampler = "dpm++2m";
    r.negativePrompt = "blurry";
    const auto entry = store.add(tinyImage(), r, 99, model(), 0.1);
    const auto back = requestFromJson(entry.parameters["request"]);
    CHECK(toJson(back) == toJson([&] { auto c = r; c.seed = 99; return c; }()));
}

TEST_CASE("gallery: list is newest first and paginates", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    for (int i = 1; i <= 5; ++i) {
        store.add(tinyImage(), request("image " + std::to_string(i)), i, model(), 0.1);
    }
    CHECK(store.count() == 5);
    const auto all = store.list();
    REQUIRE(all.size() == 5);
    CHECK(all.front().prompt == "image 5");
    CHECK(all.back().prompt == "image 1");

    const auto page = store.list(GalleryQuery{"", 2, 1});
    REQUIRE(page.size() == 2);
    CHECK(page[0].prompt == "image 4");
    CHECK(page[1].prompt == "image 3");
    CHECK(store.list(GalleryQuery{"", 10, 99}).empty());
}

TEST_CASE("gallery: search matches the prompt, case-insensitively, and treats wildcards literally", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    store.add(tinyImage(), request("A red Lighthouse at dusk"), 1, model(), 0.1);
    store.add(tinyImage(), request("a green forest"), 2, model(), 0.1);
    store.add(tinyImage(), request("100% pure_water"), 3, model(), 0.1);

    CHECK(store.list(GalleryQuery{"lighthouse", 50, 0}).size() == 1);
    CHECK(store.count("LIGHTHOUSE") == 1);
    CHECK(store.count("a ") == 2);
    CHECK(store.list(GalleryQuery{"100%", 50, 0}).size() == 1);
    CHECK(store.count("%") == 1);       // only the prompt that really contains a percent sign
    CHECK(store.count("pure_") == 1);
    CHECK(store.count("pure-") == 0);   // underscore is not a wildcard
    CHECK(store.count("nothing like this") == 0);
}

TEST_CASE("gallery: get and remove", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    const auto entry = store.add(tinyImage(), request("to delete"), 5, model(), 0.1);
    REQUIRE(store.get(entry.id).has_value());
    CHECK(store.get(entry.id)->prompt == "to delete");
    CHECK_FALSE(store.get(entry.id + 100).has_value());

    const auto image = store.absolutePath(entry);
    CHECK(store.remove(entry.id));
    CHECK_FALSE(std::filesystem::exists(image));
    CHECK_FALSE(store.get(entry.id).has_value());
    CHECK_FALSE(store.remove(entry.id));
    CHECK(store.count() == 0);
}

TEST_CASE("gallery: names never collide, even for the same second and seed", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    std::set<std::string> paths;
    for (int i = 0; i < 6; ++i) {
        paths.insert(store.add(tinyImage(), request("same"), 7, model(), 0.1).path);
    }
    CHECK(paths.size() == 6);
}

TEST_CASE("gallery: all formats, metadata can be switched off", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    for (const char* format : {"png", "jpeg", "webp"}) {
        const auto entry = store.add(tinyImage(), request("fmt", format), 1, model(), 0.1);
        CHECK(entry.format == format);
        const auto bytes = gh::core::readFile(store.absolutePath(entry)).value();
        const std::vector<uint8_t> v(bytes.begin(), bytes.end());
        CHECK(readImageMetadata(v).has_value());
        CHECK(decodeImage(v).width == 8);
    }
    auto r = request("private");
    r.embedMetadata = false;
    const auto entry = store.add(tinyImage(), r, 1, model(), 0.1);
    const auto bytes = gh::core::readFile(store.absolutePath(entry)).value();
    CHECK_FALSE(readImageMetadata(std::vector<uint8_t>(bytes.begin(), bytes.end())).has_value());
    auto sidecar = store.absolutePath(entry);
    sidecar.replace_extension(".json");
    CHECK(std::filesystem::exists(sidecar));  // the sidecar is always written
}

TEST_CASE("gallery: addResult stores every image with its own seed", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    GenerationResult result;
    result.request = request("batch");
    result.request.seed = 10;
    result.images = {tinyImage(10), tinyImage(20), tinyImage(30)};
    result.seeds = {10, 11, 12};
    result.seconds = 3.0;
    const auto entries = store.addResult(result, model());
    REQUIRE(entries.size() == 3);
    CHECK(entries[0].seed == 10);
    CHECK(entries[2].seed == 12);
    CHECK(entries[1].parameters["request"]["seed"] == 11);
    CHECK(entries[1].parameters["seconds"] == 1.0);  // the batch time is shared
}

TEST_CASE("gallery: the index survives a restart and can be rebuilt from the sidecars", "[unit][gallery]") {
    gh::test::TempDir dir;
    {
        GalleryStore store(dir.path());
        store.add(tinyImage(), request("first"), 1, model(), 0.1);
        store.add(tinyImage(), request("second", "webp"), 2, model(), 0.1);
    }
    {
        GalleryStore store(dir.path());  // reopened
        CHECK(store.count() == 2);
    }
    for (const char* name : {"gallery.db", "gallery.db-wal", "gallery.db-shm"}) {
        std::filesystem::remove(dir.path() / name);
    }
    GalleryStore store(dir.path());  // a fresh, empty index
    CHECK(store.count() == 0);
    CHECK(store.rescan() == 2);
    CHECK(store.count() == 2);
    CHECK(store.count("second") == 1);
    CHECK(store.rescan() == 0);  // nothing more to add

    // an image deleted behind our back disappears from the index
    const auto entry = store.list(GalleryQuery{"first", 10, 0}).front();
    std::filesystem::remove(store.absolutePath(entry));
    store.rescan();
    CHECK(store.count() == 1);
}

TEST_CASE("gallery: several threads can add at once", "[unit][gallery]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 5; ++i) {
                store.add(tinyImage(), request("thread " + std::to_string(t)), t * 100 + i, model(), 0.1);
            }
        });
    }
    for (auto& t : threads) t.join();
    CHECK(store.count() == 20);
}

// ---- history of graph jobs (workflow plan W2.5) -------------------------------------------------------------------------

namespace {
HistoryEntry historyEntry(int64_t number, const std::string& status = "done") {
    HistoryEntry h;
    h.queueNumber = number;
    h.clientId = "editor";
    h.label = "run " + std::to_string(number);
    h.status = status;
    h.seconds = 1.25;
    h.job = {{"format", "graphicshost.job"}, {"nodes", nlohmann::json::object()}};
    h.workflow = {{"format", "graphicshost.workflow"}, {"id", "wf"}};
    h.outputs = nlohmann::json::array({{{"node", "9"}, {"gallery", 12}, {"path", "2026/10/25/a.png"}}});
    h.timings = {{"4", 120.5}};
    if (status == "failed") {
        h.error = "out of memory";
        h.errorNode = "4";
    }
    return h;
}
}  // namespace

TEST_CASE("gallery: graph history stores, lists and prunes jobs", "[unit][gallery][history]") {
    gh::test::TempDir dir;
    GalleryStore store(dir.path());
    CHECK(store.historyCount() == 0);
    const int64_t first = store.addHistory(historyEntry(1));
    store.addHistory(historyEntry(2, "failed"));
    store.addHistory(historyEntry(3));
    CHECK(store.historyCount() == 3);

    const auto list = store.history(2, 0);   // newest first, the large fields left out
    REQUIRE(list.size() == 2);
    CHECK(list[0].queueNumber == 3);
    CHECK(list[1].status == "failed");
    CHECK(list[1].error == "out of memory");
    CHECK(list[1].errorNode == "4");
    CHECK(list[0].job.is_null());   // not loaded in a listing
    CHECK(store.history(2, 2).size() == 1);

    const auto item = store.historyItem(first);
    REQUIRE(item.has_value());
    CHECK(item->label == "run 1");
    CHECK(item->clientId == "editor");
    CHECK(item->seconds == 1.25);
    CHECK(item->job["format"] == "graphicshost.job");
    CHECK(item->workflow["id"] == "wf");
    CHECK(item->outputs[0]["gallery"] == 12);
    CHECK(item->timings["4"] == 120.5);
    CHECK(!item->created.empty());
    CHECK_FALSE(store.historyItem(9999).has_value());

    CHECK(store.pruneHistory(0) == 0);   // 0 means unlimited
    CHECK(store.pruneHistory(2) == 1);   // the oldest goes
    CHECK(store.historyCount() == 2);
    CHECK_FALSE(store.historyItem(first).has_value());
    CHECK(store.removeHistory(store.history(1, 0)[0].id));
    CHECK_FALSE(store.removeHistory(first));
    CHECK(store.historyCount() == 1);
}

TEST_CASE("gallery: a version 1 database gains the history table and keeps its images", "[unit][gallery][history]") {
    gh::test::TempDir dir;
    {
        GalleryStore store(dir.path());
        store.add(tinyImage(), request("old image"), 1, model(), 1.0);
    }
    // downgrade the index by hand: drop the history table and set the old version
    {
        sqlite3* db = nullptr;
        REQUIRE(sqlite3_open((dir.path() / "gallery.db").string().c_str(), &db) == SQLITE_OK);
        REQUIRE(sqlite3_exec(db, "DROP TABLE graph_history; PRAGMA user_version=1;", nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(db);
    }
    GalleryStore reopened(dir.path());
    CHECK(reopened.count() == 1);
    CHECK(reopened.addHistory(historyEntry(1)) > 0);
    CHECK(reopened.historyCount() == 1);
    {
        sqlite3* db = nullptr;
        REQUIRE(sqlite3_open((dir.path() / "gallery.db").string().c_str(), &db) == SQLITE_OK);
        sqlite3_stmt* s = nullptr;
        REQUIRE(sqlite3_prepare_v2(db, "PRAGMA user_version", -1, &s, nullptr) == SQLITE_OK);
        REQUIRE(sqlite3_step(s) == SQLITE_ROW);
        CHECK(sqlite3_column_int(s, 0) == 2);
        sqlite3_finalize(s);
        sqlite3_close(db);
    }
}
