#include "GenApiTestUtil.h"

#include "engine/SafetyFilter.h"

#include <fstream>

#include <catch2/catch_test_macros.hpp>

using namespace gh;
using gh::test::GenApp;
using nlohmann::json;

TEST_CASE("safety: the blocklist matches whole words and phrases, any case", "[unit][safety]") {
    const engine::BlocklistFilter f({"nsfw", "blood bath", "Gore"});
    CHECK(f.checkPrompt("a cat on a sofa").allowed);
    CHECK_FALSE(f.checkPrompt("A NSFW picture").allowed);
    CHECK_FALSE(f.checkPrompt("gore, horror").allowed);
    CHECK_FALSE(f.checkPrompt("a blood-bath scene").allowed);   // punctuation does not hide a phrase
    CHECK(f.checkPrompt("a blood sample in a bath").allowed);   // the words are not adjacent
    CHECK(f.checkPrompt("gorestone and nsfwish").allowed);      // whole words only
    CHECK(f.checkPrompt("").allowed);
    CHECK(f.checkPrompt("x").reason.empty());
    CHECK(f.checkPrompt("nsfw").reason.find("nsfw") != std::string::npos);
    CHECK(engine::BlocklistFilter({}).checkPrompt("anything").allowed);
}

TEST_CASE("safety: terms are read from files, ignoring comments", "[unit][safety]") {
    test::TempDir dir;
    {
        std::ofstream out(dir.path() / "t.txt");
        out << "# a comment\n\n  alpha  \nbeta gamma\r\n";
    }
    const auto terms = engine::BlocklistFilter::readTerms({dir.path() / "t.txt", dir.path() / "missing.txt"});
    REQUIRE(terms.size() == 2);
    CHECK(terms[0] == "alpha");
    CHECK(terms[1] == "beta gamma");
}

TEST_CASE("safety: off by default, on by setting or by the request, never switched off by the request", "[integration][api][safety]") {
    GenApp g;
    const json bad = {{"model", "tiny"}, {"prompt", "an nsfw picture"}};
    REQUIRE(g.post("/v1/images/generations", bad)->status == 200);   // off: nothing is checked

    // the request turns it on for itself
    json asked = bad;
    asked["safety_filter"] = true;
    const auto refused = g.post("/v1/images/generations", asked);
    CHECK(refused->status == 400);
    CHECK(test::bodyJson(refused)["error"]["code"] == "content_policy_violation");
    CHECK(test::bodyJson(refused)["error"]["message"].get<std::string>().find("nsfw") != std::string::npos);
    CHECK(g.state->requests.size() == 1);   // the refused one never reached the engine

    // a harmless prompt passes with it on
    json fine = {{"model", "tiny"}, {"prompt", "a quiet lake"}, {"safety_filter", true}};
    CHECK(g.post("/v1/images/generations", fine)->status == 200);

    // on in the settings: every request is checked, and the request cannot turn it off
    auto settings = g.app.host->settings();
    settings.safety.enabled = true;
    g.app.host->updateSettings(settings);
    CHECK(g.post("/v1/images/generations", bad)->status == 400);
    json optOut = bad;
    optOut["safety_filter"] = false;
    CHECK(g.post("/v1/images/generations", optOut)->status == 400);
    CHECK(g.post("/api/v1/generate", bad)->status == 400);

    // the negative prompt is where people write what they do not want: it is not checked
    CHECK(g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "a quiet lake"}, {"negative_prompt", "nsfw, gore"}})->status == 200);

    // the user's own list is added to the shipped one
    {
        std::ofstream out(g.app.home.path() / "safety-terms.txt");
        out << "quiet lake\n";
    }
    CHECK(g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "a quiet lake"}})->status == 400);
}
