#include "GenApiTestUtil.h"

#include "engine/ImageCodec.h"
#include "server/Base64.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace gh;
using gh::test::GenApp;
using nlohmann::json;

namespace {

// A gradient picture of an awkward size, as a PNG.
std::vector<uint8_t> samplePng(uint32_t width = 100, uint32_t height = 60) {
    engine::Image image;
    image.width = width;
    image.height = height;
    image.data.resize(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < image.data.size(); ++i) image.data[i] = static_cast<uint8_t>(i * 3);
    return engine::encodeImage(image, engine::ImageFormat::Png);
}

std::string asBase64(const std::vector<uint8_t>& bytes) { return server::base64Encode(bytes); }

engine::GenerationRequest lastRequest(GenApp& g) {
    std::lock_guard<std::mutex> lock(g.state->mutex);
    REQUIRE_FALSE(g.state->requests.empty());
    return g.state->requests.back();
}

}  // namespace

TEST_CASE("img2img: /v1/images/edits takes a multipart image", "[integration][api][img2img]") {
    GenApp g;
    auto c = g.client();
    const auto png = samplePng();
    httplib::UploadFormDataItems form = {{"prompt", "make it a painting", "", ""},
                                         {"model", "tiny", "", ""},
                                         {"strength", "0.5", "", ""},
                                         {"n", "1", "", ""},
                                         {"image", std::string(png.begin(), png.end()), "in.png", "image/png"}};
    const auto r = c.Post("/v1/images/edits", form);
    REQUIRE(r);
    INFO(r->body);
    REQUIRE(r->status == 200);
    CHECK(test::bodyJson(r)["data"][0]["b64_json"].get<std::string>().rfind("iVBORw0KGgo", 0) == 0);

    const auto request = lastRequest(g);
    REQUIRE(request.initImage);
    CHECK(request.strength == 0.5f);
    CHECK(request.prompt == "make it a painting");
    // the result takes the picture's size (rounded to a multiple of 8); the picture is scaled to it
    CHECK(request.width == 104);
    CHECK(request.height == 64);
    CHECK(request.initImage->width == 104);
    CHECK(request.initImage->height == 64);
    CHECK(request.initImage->channels == 3);
}

TEST_CASE("img2img: edits and generations also take a base64 image or a gallery image", "[integration][api][img2img]") {
    GenApp g;
    const std::string b64 = asBase64(samplePng(128, 128));

    auto viaEdits = g.post("/v1/images/edits", {{"model", "tiny"}, {"prompt", "p"}, {"image", b64}});
    REQUIRE(viaEdits->status == 200);
    CHECK(lastRequest(g).initImage);

    // with a data URL prefix, and a size given: the picture is scaled to it
    auto viaGen = g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "p"}, {"init_image", "data:image/png;base64," + b64}, {"size", "192x128"}, {"strength", 0.3}});
    REQUIRE(viaGen->status == 200);
    const auto scaled = lastRequest(g);
    REQUIRE(scaled.initImage);
    CHECK(scaled.initImage->width == 192);
    CHECK(scaled.initImage->height == 128);
    CHECK(scaled.strength == 0.3f);

    // start from something that is already in the gallery
    const auto gallery = g.app.host->gallery().list();
    REQUIRE_FALSE(gallery.empty());
    auto viaGallery = g.post("/api/v1/generate", {{"model", "tiny"}, {"prompt", "again"}, {"init_image_gallery_id", gallery.front().id}});
    REQUIRE(viaGallery->status == 202);
    for (int i = 0; i < 100; ++i) {   // let the queued job run before the app goes away
        const auto job = test::bodyJson(g.app.client().Get("/api/v1/jobs/" + std::to_string(test::bodyJson(viaGallery)["job"]["id"].get<int64_t>())));
        if (job["state"] == "succeeded") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(lastRequest(g).initImage);

    // without an image, a generation is plain text to image
    REQUIRE(g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "plain"}})->status == 200);
    CHECK_FALSE(lastRequest(g).initImage);
}

TEST_CASE("img2img: bad pictures and strengths are refused", "[integration][api][img2img]") {
    GenApp g;
    auto expect = [&](const char* path, const json& body, int status, const std::string& code) {
        const auto r = g.post(path, body);
        REQUIRE(r);
        INFO(r->body);
        CHECK(r->status == status);
        CHECK(test::bodyJson(r)["error"]["code"] == code);
    };
    const std::string good = asBase64(samplePng());
    expect("/v1/images/edits", {{"model", "tiny"}, {"prompt", "p"}}, 400, "invalid_parameters");                                   // no image
    expect("/v1/images/edits", {{"model", "tiny"}, {"prompt", "p"}, {"image", "not base64!"}}, 400, "invalid_parameters");
    expect("/v1/images/edits", {{"model", "tiny"}, {"prompt", "p"}, {"image", asBase64({1, 2, 3, 4, 5, 6, 7, 8})}}, 400, "invalid_parameters");   // not a picture
    expect("/v1/images/edits", {{"model", "tiny"}, {"prompt", "p"}, {"image", 5}}, 400, "invalid_parameters");
    expect("/v1/images/edits", {{"model", "tiny"}, {"prompt", "p"}, {"image", good}, {"strength", 0}}, 400, "invalid_parameters");
    expect("/v1/images/edits", {{"model", "tiny"}, {"prompt", "p"}, {"image", good}, {"strength", 1.5}}, 400, "invalid_parameters");
    expect("/api/v1/generate", {{"model", "tiny"}, {"prompt", "p"}, {"init_image_gallery_id", 4242}}, 400, "invalid_parameters");

    auto c = g.client();
    const auto png = samplePng();
    httplib::UploadFormDataItems noImage = {{"prompt", "p", "", ""}};
    CHECK(c.Post("/v1/images/edits", noImage)->status == 400);
    httplib::UploadFormDataItems badNumber = {{"prompt", "p", "", ""}, {"strength", "lots", "", ""}, {"image", std::string(png.begin(), png.end()), "in.png", "image/png"}};
    CHECK(c.Post("/v1/images/edits", badNumber)->status == 400);
    CHECK(g.state->requests.empty());   // nothing reached the engine
}

TEST_CASE("img2img: the gallery remembers the strength", "[integration][api][img2img]") {
    GenApp g;
    REQUIRE(g.post("/v1/images/edits", {{"model", "tiny"}, {"prompt", "p"}, {"image", asBase64(samplePng())}, {"strength", 0.6}})->status == 200);
    const auto entries = g.app.host->gallery().list();
    REQUIRE(entries.size() == 1);
    CHECK(entries[0].parameters["request"]["strength"].get<double>() == Catch::Approx(0.6));
    CHECK(entries[0].parameters["request"]["hasInitImage"] == true);
}
