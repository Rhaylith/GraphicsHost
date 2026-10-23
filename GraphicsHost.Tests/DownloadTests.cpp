#include "TestHttpServer.h"
#include "TestUtil.h"

#include "core/Sha256.h"
#include "models/DownloadManager.h"
#include "models/ModelLibrary.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::models;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

DownloadRequest request(gh::test::TestHttpServer& server, const fs::path& dest, const std::string& path, uint64_t size) {
    DownloadRequest r;
    r.url = server.url(path + std::to_string(size));
    r.destination = dest;
    return r;
}

std::string expectedHash(uint64_t size) {
    return gh::core::sha256Hex(gh::test::blobBytes(0, size));
}

bool waitUntil(const std::function<bool()>& condition, std::chrono::milliseconds limit = 10s) {
    const auto end = std::chrono::steady_clock::now() + limit;
    while (!condition() && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(10ms);
    return condition();
}

}  // namespace

TEST_CASE("sha256: known answers", "[unit][download]") {
    CHECK(gh::core::sha256Hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(gh::core::sha256Hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    gh::test::TempDir dir;
    gh::test::writeText(dir.path() / "f.txt", "abc");
    CHECK(gh::core::sha256File(dir.path() / "f.txt").value() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK_FALSE(gh::core::sha256File(dir.path() / "missing").has_value());
    CHECK_FALSE(gh::core::sha256File(dir.path() / "f.txt", [](uint64_t, uint64_t) { return false; }).has_value());  // stopped
}

TEST_CASE("download: a file arrives complete, verified, with a source record", "[unit][download]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    DownloadManager manager;
    const uint64_t size = 300 * 1000 + 17;
    auto r = request(server, dir.path() / "sub" / "model.safetensors", "/blob/", size);
    r.sha256 = expectedHash(size);
    r.source = {{"repo", "someone/model"}, {"revision", "abc123"}, {"companions", json::array()}};
    const auto d = manager.enqueue(r);
    REQUIRE(d->waitFor(20s));
    INFO(d->message());
    CHECK(d->state() == DownloadState::Completed);
    CHECK(d->bytesDone() == size);
    CHECK(d->bytesTotal() == size);
    CHECK(d->sha256() == expectedHash(size));
    CHECK(gh::test::readText(r.destination) == gh::test::blobBytes(0, size));
    CHECK_FALSE(fs::exists(fs::path(r.destination.string() + ".part")));

    const auto record = json::parse(gh::test::readText(sourceSidecarFor(r.destination)));
    CHECK(record["repo"] == "someone/model");
    CHECK(record["revision"] == "abc123");
    CHECK(record["sha256"] == expectedHash(size));
    CHECK(record["sizeBytes"] == size);
    CHECK(record["url"].get<std::string>().find("/blob/") != std::string::npos);
    CHECK_FALSE(record["downloadedAt"].get<std::string>().empty());
}

TEST_CASE("download: a wrong checksum fails and removes the file", "[unit][download]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    DownloadManager manager;
    auto r = request(server, dir.path() / "m.safetensors", "/blob/", 50000);
    r.sha256 = std::string(64, '0');
    const auto d = manager.enqueue(r);
    REQUIRE(d->waitFor(20s));
    CHECK(d->state() == DownloadState::Failed);
    CHECK(d->error() == DownloadError::Checksum);
    CHECK(d->message().find("checksum mismatch") != std::string::npos);
    CHECK_FALSE(fs::exists(r.destination));
    CHECK_FALSE(fs::exists(fs::path(r.destination.string() + ".part")));
}

TEST_CASE("download: a partial file from an earlier run is continued with a Range request", "[unit][download][resume]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    DownloadManager manager;
    const uint64_t size = 400000, have = 123456;
    const auto dest = dir.path() / "m.safetensors";
    gh::test::writeText(fs::path(dest.string() + ".part"), gh::test::blobBytes(0, have));  // what a killed run leaves

    auto r = request(server, dest, "/blob/", size);
    r.sha256 = expectedHash(size);
    const auto d = manager.enqueue(r);
    REQUIRE(d->waitFor(20s));
    INFO(d->message());
    CHECK(d->state() == DownloadState::Completed);
    CHECK(server.range() == "bytes=" + std::to_string(have) + "-");
    CHECK(gh::test::readText(dest) == gh::test::blobBytes(0, size));
}

TEST_CASE("download: a server that ignores Range makes it start over, not corrupt the file", "[unit][download][resume]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    DownloadManager manager;
    const uint64_t size = 200000;
    const auto dest = dir.path() / "m.safetensors";
    gh::test::writeText(fs::path(dest.string() + ".part"), gh::test::blobBytes(0, 50000));
    const auto d = manager.enqueue(request(server, dest, "/norange/", size));
    REQUIRE(d->waitFor(20s));
    INFO(d->message());
    CHECK(d->state() == DownloadState::Completed);
    CHECK(gh::test::readText(dest) == gh::test::blobBytes(0, size));
}

TEST_CASE("download: a part that is already complete is just verified", "[unit][download][resume]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    DownloadManager manager;
    const uint64_t size = 90000;
    const auto dest = dir.path() / "m.safetensors";
    gh::test::writeText(fs::path(dest.string() + ".part"), gh::test::blobBytes(0, size));
    auto r = request(server, dest, "/blob/", size);
    r.sha256 = expectedHash(size);
    const auto d = manager.enqueue(r);
    REQUIRE(d->waitFor(20s));
    CHECK(d->state() == DownloadState::Completed);
    CHECK(gh::test::readText(dest) == gh::test::blobBytes(0, size));
}

TEST_CASE("download: HTTP problems are reported with their meaning", "[unit][download]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    DownloadManager manager;

    const auto missing = manager.enqueue(request(server, dir.path() / "a.bin", "/nothing/", 10));
    REQUIRE(missing->waitFor(20s));
    CHECK(missing->state() == DownloadState::Failed);
    CHECK(missing->error() == DownloadError::Http);
    CHECK(missing->httpStatus() == 404);

    const auto gated = manager.enqueue(request(server, dir.path() / "b.bin", "/gated/", 10000));
    REQUIRE(gated->waitFor(20s));
    CHECK(gated->state() == DownloadState::Failed);
    CHECK(gated->error() == DownloadError::Unauthorized);
    CHECK(gated->message().find("token") != std::string::npos);
    CHECK(gated->httpStatus() == 401);

    auto withToken = request(server, dir.path() / "c.bin", "/gated/", 10000);
    withToken.bearerToken = "secret";
    const auto ok = manager.enqueue(withToken);
    REQUIRE(ok->waitFor(20s));
    INFO(ok->message());
    CHECK(ok->state() == DownloadState::Completed);
    CHECK(server.authorization() == "Bearer secret");

    // nothing is listening there
    DownloadRequest dead;
    dead.url = "http://127.0.0.1:1/x";
    dead.destination = dir.path() / "d.bin";
    const auto unreachable = manager.enqueue(dead);
    REQUIRE(unreachable->waitFor(30s));
    CHECK(unreachable->state() == DownloadState::Failed);
    CHECK(unreachable->error() == DownloadError::Network);
}

TEST_CASE("download: not enough disk space is refused before downloading", "[unit][download]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    DownloadManager manager;
    manager.setFreeSpaceFunction([](const fs::path&) { return uint64_t{1000}; });
    const auto d = manager.enqueue(request(server, dir.path() / "m.bin", "/blob/", 5000000));
    REQUIRE(d->waitFor(20s));
    CHECK(d->state() == DownloadState::Failed);
    CHECK(d->error() == DownloadError::DiskSpace);
    CHECK(d->message().find("not enough disk space") != std::string::npos);
    CHECK_FALSE(fs::exists(fs::path((dir.path() / "m.bin").string() + ".part")));  // nothing was written
}

TEST_CASE("download: pause keeps the partial file, resume finishes the job", "[unit][download][resume]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    server.slowMs = 5;  // 16 KiB per 5 ms
    DownloadManager manager;
    const uint64_t size = 3 * 1000 * 1000;
    auto r = request(server, dir.path() / "m.safetensors", "/blob/", size);
    r.sha256 = expectedHash(size);
    const auto d = manager.enqueue(r);

    REQUIRE(waitUntil([&] { return d->bytesDone() > 200000; }));
    manager.pause(d->id());
    REQUIRE(waitUntil([&] { return d->state() == DownloadState::Paused; }));
    const uint64_t paused = d->bytesDone();
    CHECK(paused < size);
    const fs::path part = r.destination.string() + ".part";
    REQUIRE(fs::exists(part));
    CHECK(fs::file_size(part) >= paused - 1);

    server.slowMs = 0;
    manager.resume(d->id());
    REQUIRE(d->waitFor(20s));
    INFO(d->message());
    CHECK(d->state() == DownloadState::Completed);
    CHECK(server.range().rfind("bytes=", 0) == 0);  // it continued, it did not start over
    CHECK(gh::test::readText(r.destination) == gh::test::blobBytes(0, size));
}

TEST_CASE("download: cancel removes the partial file", "[unit][download]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    server.slowMs = 5;
    DownloadManager manager;
    auto r = request(server, dir.path() / "m.safetensors", "/blob/", 5 * 1000 * 1000);
    const auto d = manager.enqueue(r);
    REQUIRE(waitUntil([&] { return d->bytesDone() > 100000; }));
    manager.cancel(d->id());
    REQUIRE(d->waitFor(10s));
    CHECK(d->state() == DownloadState::Cancelled);
    CHECK_FALSE(fs::exists(r.destination));
    CHECK_FALSE(fs::exists(fs::path(r.destination.string() + ".part")));
}

TEST_CASE("download: a failed transfer can be resumed once the problem is gone", "[unit][download][resume]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    DownloadManager manager;
    bool plentyOfSpace = false;
    manager.setFreeSpaceFunction([&](const fs::path&) { return plentyOfSpace ? UINT64_MAX : uint64_t{10}; });
    const auto d = manager.enqueue(request(server, dir.path() / "m.bin", "/blob/", 100000));
    REQUIRE(d->waitFor(20s));
    CHECK(d->state() == DownloadState::Failed);
    plentyOfSpace = true;
    manager.resume(d->id());
    REQUIRE(waitUntil([&] { return d->state() == DownloadState::Completed; }));
    CHECK(d->error() == DownloadError::None);
    CHECK(d->message().empty());
}

TEST_CASE("download: the queue runs one download after the other and reports progress", "[unit][download]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    server.slowMs = 2;
    DownloadManager manager(1);
    std::mutex m;
    std::vector<std::string> events;
    manager.setObserver([&](const Download& d) {
        std::lock_guard<std::mutex> lock(m);
        events.push_back(std::to_string(d.id()) + ":" + toString(d.state()));
    });
    const auto first = manager.enqueue(request(server, dir.path() / "a.bin", "/blob/", 200000));
    const auto second = manager.enqueue(request(server, dir.path() / "b.bin", "/blob/", 200000));
    CHECK(manager.list().size() == 2);
    CHECK(manager.find(second->id()) == second);
    CHECK(manager.find(99) == nullptr);
    REQUIRE(second->waitFor(30s));
    CHECK(first->state() == DownloadState::Completed);
    CHECK(second->state() == DownloadState::Completed);
    std::lock_guard<std::mutex> lock(m);
    const auto at = [&](const std::string& e) { return std::find(events.begin(), events.end(), e) - events.begin(); };
    CHECK(at("1:completed") < at("2:running"));  // strictly one after the other
}

TEST_CASE("download: destroying the manager mid-transfer keeps the partial file for next time", "[unit][download][resume]") {
    gh::test::TempDir dir;
    gh::test::TestHttpServer server;
    server.slowMs = 5;
    const fs::path dest = dir.path() / "m.safetensors";
    {
        DownloadManager manager;
        const auto d = manager.enqueue(request(server, dest, "/blob/", 4 * 1000 * 1000));
        REQUIRE(waitUntil([&] { return d->bytesDone() > 150000; }));
    }  // the application quits
    const fs::path part = dest.string() + ".part";
    REQUIRE(fs::exists(part));
    CHECK(fs::file_size(part) > 100000);
    CHECK_FALSE(fs::exists(dest));

    server.slowMs = 0;
    DownloadManager again;  // the next start of the application
    const auto d = again.enqueue(request(server, dest, "/blob/", 4 * 1000 * 1000));
    REQUIRE(d->waitFor(30s));
    CHECK(d->state() == DownloadState::Completed);
    CHECK(gh::test::readText(dest) == gh::test::blobBytes(0, 4 * 1000 * 1000));
}
