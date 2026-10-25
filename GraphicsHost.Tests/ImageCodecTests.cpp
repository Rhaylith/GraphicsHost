#include "engine/ImageCodec.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::engine;

namespace {

// A smooth colour gradient: compresses well and shows lossy errors clearly.
Image gradient(uint32_t w, uint32_t h, uint32_t channels = 3) {
    Image image;
    image.width = w;
    image.height = h;
    image.channels = channels;
    image.data.resize(static_cast<size_t>(w) * h * channels);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            uint8_t* p = &image.data[(static_cast<size_t>(y) * w + x) * channels];
            p[0] = static_cast<uint8_t>(x * 255 / (w - 1));
            p[1] = static_cast<uint8_t>(y * 255 / (h - 1));
            p[2] = static_cast<uint8_t>((x + y) * 255 / (w + h - 2));
            if (channels == 4) p[3] = 255;
        }
    }
    return image;
}

double meanAbsDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    REQUIRE(a.size() == b.size());
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += std::abs(int(a[i]) - int(b[i]));
    return sum / static_cast<double>(a.size());
}

const std::string kMeta = R"({"prompt":"a \"quoted\" lighthouse","seed":42,"umlaut":"häuser"})";

}  // namespace

TEST_CASE("codec: format names and extensions", "[unit][codec]") {
    CHECK(parseImageFormat("png") == ImageFormat::Png);
    CHECK(parseImageFormat("jpeg") == ImageFormat::Jpeg);
    CHECK(parseImageFormat("jpg") == ImageFormat::Jpeg);
    CHECK(parseImageFormat("webp") == ImageFormat::Webp);
    CHECK_FALSE(parseImageFormat("bmp").has_value());
    CHECK(std::string(extensionFor(ImageFormat::Jpeg)) == ".jpg");
    CHECK(std::string(mimeTypeFor(ImageFormat::Webp)) == "image/webp");
}

TEST_CASE("codec: PNG is lossless and carries the metadata", "[unit][codec]") {
    const Image original = gradient(64, 48);
    const auto bytes = encodeImage(original, ImageFormat::Png, kMeta);
    CHECK(detectImageFormat(bytes) == ImageFormat::Png);
    const Image back = decodeImage(bytes);
    CHECK(back.width == 64);
    CHECK(back.height == 48);
    CHECK(back.data == original.data);
    CHECK(readImageMetadata(bytes) == kMeta);
}

TEST_CASE("codec: PNG without metadata, with RGBA input, with long metadata", "[unit][codec]") {
    const auto plain = encodeImage(gradient(16, 16), ImageFormat::Png);
    CHECK_FALSE(readImageMetadata(plain).has_value());

    const Image rgba = gradient(16, 16, 4);
    const Image back = decodeImage(encodeImage(rgba, ImageFormat::Png));
    CHECK(back.channels == 3);
    CHECK(back.data == gradient(16, 16).data);  // alpha dropped, colours intact

    const std::string big(5000, 'x');  // above the compression threshold of the text chunk
    CHECK(readImageMetadata(encodeImage(gradient(8, 8), ImageFormat::Png, big)) == big);
}

TEST_CASE("codec: JPEG is close to the original and carries the metadata", "[unit][codec]") {
    const Image original = gradient(64, 48);
    const auto bytes = encodeImage(original, ImageFormat::Jpeg, kMeta, 95);
    CHECK(detectImageFormat(bytes) == ImageFormat::Jpeg);
    const Image back = decodeImage(bytes);
    CHECK(back.width == 64);
    CHECK(back.height == 48);
    CHECK(meanAbsDiff(original.data, back.data) < 3.0);
    CHECK(readImageMetadata(bytes) == kMeta);
    // lower quality means a smaller file
    CHECK(encodeImage(original, ImageFormat::Jpeg, {}, 20).size() < encodeImage(original, ImageFormat::Jpeg, {}, 95).size());
}

TEST_CASE("codec: WebP lossy and lossless, with metadata", "[unit][codec]") {
    const Image original = gradient(64, 48);
    const auto lossy = encodeImage(original, ImageFormat::Webp, kMeta, 90);
    CHECK(detectImageFormat(lossy) == ImageFormat::Webp);
    CHECK(meanAbsDiff(original.data, decodeImage(lossy).data) < 4.0);
    CHECK(readImageMetadata(lossy) == kMeta);

    const auto lossless = encodeImage(original, ImageFormat::Webp, kMeta, 100);
    CHECK(decodeImage(lossless).data == original.data);
    CHECK(readImageMetadata(lossless) == kMeta);

    CHECK_FALSE(readImageMetadata(encodeImage(original, ImageFormat::Webp, {}, 90)).has_value());
}

TEST_CASE("codec: bad input is rejected", "[unit][codec]") {
    Image broken = gradient(8, 8);
    broken.data.pop_back();
    CHECK_THROWS_AS(encodeImage(broken, ImageFormat::Png), std::runtime_error);
    CHECK_THROWS_AS(encodeImage(Image{}, ImageFormat::Jpeg), std::runtime_error);

    CHECK_FALSE(detectImageFormat({1, 2, 3}).has_value());
    CHECK_FALSE(readImageMetadata({1, 2, 3}).has_value());
    CHECK_THROWS_AS(decodeImage({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), std::runtime_error);

    auto truncated = encodeImage(gradient(32, 32), ImageFormat::Png);
    truncated.resize(truncated.size() / 2);
    CHECK_THROWS(decodeImage(truncated));
}

TEST_CASE("codec: a real-sized image round trips", "[unit][codec]") {
    const Image big = gradient(1024, 1024);
    const auto png = encodeImage(big, ImageFormat::Png, kMeta);
    CHECK(decodeImage(png).data == big.data);
}
