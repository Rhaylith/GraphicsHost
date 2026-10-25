#include "TestHttpServer.h"

#include "models/ModelUpdates.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

using namespace gh::models;
using nlohmann::json;

namespace {

ModelCard card(const std::string& file, json source) {
    ModelCard c;
    c.file = file;
    c.name = file;
    c.source = std::move(source);
    return c;
}

std::string sha(char c) { return std::string(64, c); }

}  // namespace

TEST_CASE("updates: up to date, changed, missing and not checkable", "[unit][updates]") {
    gh::test::TestHttpServer server;
    std::atomic<int> requests{0};
    server.server().Get(R"(/api/models/([^/]+)/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
        ++requests;
        const std::string repo = req.matches[1].str() + "/" + req.matches[2].str();
        if (repo == "gone/repo") { res.status = 404; return; }
        if (repo == "limit/repo") { res.status = 429; return; }
        json doc = {{"id", repo}, {"sha", "newcommit"}, {"siblings", json::array({
            {{"rfilename", "same.safetensors"}, {"size", 5}, {"lfs", {{"sha256", sha('a')}}}},
            {{"rfilename", "changed.safetensors"}, {"size", 5}, {"lfs", {{"sha256", sha('c')}}}},
            {{"rfilename", "nohash.safetensors"}, {"size", 5}}})}};
        res.set_content(doc.dump(), "application/json");
    });
    HuggingFace hub("", server.url(""));

    const std::vector<ModelCard> cards = {
        card("same.safetensors", {{"repo", "pub/repo"}, {"file", "same.safetensors"}, {"sha256", sha('a')}, {"revision", "oldcommit"}}),
        card("changed-local-name.safetensors", {{"repo", "pub/repo"}, {"file", "changed.safetensors"}, {"sha256", sha('b')}}),
        card("renamed.safetensors", {{"repo", "pub/repo"}, {"file", "was-renamed.safetensors"}, {"sha256", sha('d')}}),
        card("nohash.safetensors", {{"repo", "pub/repo"}, {"file", "nohash.safetensors"}, {"sha256", sha('e')}}),
        card("gone.safetensors", {{"repo", "gone/repo"}, {"file", "x.safetensors"}, {"sha256", sha('f')}}),
        card("limited.safetensors", {{"repo", "limit/repo"}, {"file", "y.safetensors"}, {"sha256", sha('f')}}),
        card("manual.safetensors", json(nullptr)),                                   // no record: not listed
        card("badrecord.safetensors", {{"repo", "no-slash"}, {"file", "a"}}),         // not a repository name: not listed
    };
    const auto updates = checkForUpdates(cards, hub);
    REQUIRE(updates.size() == 6);
    auto find = [&](const std::string& file) -> const UpdateInfo& {
        for (const auto& u : updates) if (u.file == file) return u;
        FAIL("not listed: " << file);
        return updates[0];
    };
    CHECK(find("same.safetensors").status == UpdateStatus::UpToDate);
    CHECK(find("same.safetensors").latestCommit == "newcommit");   // other files changed, this one did not
    CHECK(find("changed-local-name.safetensors").status == UpdateStatus::UpdateAvailable);
    CHECK(find("changed-local-name.safetensors").latestSha256 == sha('c'));
    CHECK(find("changed-local-name.safetensors").message.find("download it again") != std::string::npos);
    CHECK(find("renamed.safetensors").status == UpdateStatus::Missing);
    CHECK(find("nohash.safetensors").status == UpdateStatus::Unknown);
    CHECK(find("gone.safetensors").status == UpdateStatus::Unknown);
    CHECK(find("gone.safetensors").message.find("404") != std::string::npos);
    CHECK(find("limited.safetensors").status == UpdateStatus::Unknown);
    CHECK(requests == 3);   // one request per repository, not per file
    CHECK(std::string(toString(UpdateStatus::UpdateAvailable)) == "update-available");
}

TEST_CASE("updates: offline is reported per model, not thrown", "[unit][updates]") {
    HuggingFace dead("", "http://127.0.0.1:1");
    const auto updates = checkForUpdates({card("a.safetensors", {{"repo", "pub/repo"}, {"file", "a.safetensors"}, {"sha256", sha('a')}})}, dead);
    REQUIRE(updates.size() == 1);
    CHECK(updates[0].status == UpdateStatus::Unknown);
    CHECK_FALSE(updates[0].message.empty());
    CHECK(checkForUpdates({}, dead).empty());
}
