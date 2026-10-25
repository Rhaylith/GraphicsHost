#include "tensor/Tensor.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace gh::tensor;
using Catch::Approx;

namespace {
// A [1, H, W, C] image with the values 0, 1, 2, ... in memory order.
Tensor ramp(int64_t h, int64_t w, int64_t c = 1) {
    std::vector<float> v(static_cast<size_t>(h * w * c));
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
    return Tensor({1, h, w, c}, v);
}
}  // namespace

TEST_CASE("tensor: shape, indexing and construction errors", "[unit][tensor]") {
    Tensor t({2, 3}, std::vector<float>{0, 1, 2, 3, 4, 5});
    CHECK(t.ndim() == 2);
    CHECK(t.numel() == 6);
    CHECK(t.at({1, 2}) == 5.0f);
    CHECK(t.offset({1, 0}) == 3);
    CHECK_THROWS(t.at({2, 0}));
    CHECK_THROWS(t.at({0}));
    CHECK_THROWS(Tensor({2, 2}, std::vector<float>{1, 2, 3}));
    CHECK_THROWS(Tensor({1, 1, 1, 1, 1, 1}));
    CHECK(Tensor({0, 4}).empty());
}

TEST_CASE("tensor: element-wise operations", "[unit][tensor]") {
    const Tensor a({3}, std::vector<float>{1, 2, 3}), b({3}, std::vector<float>{10, 20, 30});
    CHECK(add(a, b).values() == std::vector<float>{11, 22, 33});
    CHECK(sub(b, a).values() == std::vector<float>{9, 18, 27});
    CHECK(mul(a, b).values() == std::vector<float>{10, 40, 90});
    CHECK(scale(a, 2).values() == std::vector<float>{2, 4, 6});
    CHECK(addScalar(a, 1).values() == std::vector<float>{2, 3, 4});
    CHECK(clamp(b, 15, 25).values() == std::vector<float>{15, 20, 25});
    CHECK(lerp(a, b, 0.5f).values() == std::vector<float>{5.5f, 11, 16.5f});
    CHECK(invert(Tensor({2}, std::vector<float>{0.25f, 1.0f})).values() == std::vector<float>{0.75f, 0.0f});
    CHECK_THROWS(add(a, Tensor({2})));
    // a single value broadcasts
    CHECK(add(a, Tensor({1}, 5.0f)).values() == std::vector<float>{6, 7, 8});
    // big tensors take the parallel path and give the same answer
    Tensor big({100000}, 2.0f);
    const Tensor r = mul(big, big);
    CHECK(r.values().front() == 4.0f);
    CHECK(r.values().back() == 4.0f);
}

TEST_CASE("tensor: slerp follows the angle, not the straight line", "[unit][tensor]") {
    const Tensor a({1, 2}, std::vector<float>{1, 0}), b({1, 2}, std::vector<float>{0, 1});
    const Tensor mid = slerp(a, b, 0.5f);
    CHECK(mid.values()[0] == Approx(0.70710678f).margin(1e-5));
    CHECK(mid.values()[1] == Approx(0.70710678f).margin(1e-5));
    CHECK(slerp(a, b, 0.0f).values()[0] == Approx(1.0f));
    CHECK(slerp(a, b, 1.0f).values()[1] == Approx(1.0f));
    // parallel inputs fall back to the straight line
    CHECK(slerp(a, a, 0.3f).values()[0] == Approx(1.0f));
}

TEST_CASE("tensor: channel statistics", "[unit][tensor]") {
    // two pixels, two channels: ch0 = {1, 3}, ch1 = {10, 10}
    const Tensor t({1, 1, 2, 2}, std::vector<float>{1, 10, 3, 10});
    const auto mean = channelMean(t), sd = channelStd(t);
    CHECK(mean[0] == Approx(2.0f));
    CHECK(mean[1] == Approx(10.0f));
    CHECK(sd[0] == Approx(1.0f));
    CHECK(sd[1] == Approx(0.0f));
}

TEST_CASE("tensor: concat and slice along any axis", "[unit][tensor]") {
    const Tensor a({2, 2}, std::vector<float>{1, 2, 3, 4}), b({2, 2}, std::vector<float>{5, 6, 7, 8});
    CHECK(concat({a, b}, 0).values() == std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    const Tensor along1 = concat({a, b}, 1);
    CHECK(along1.shape() == std::vector<int64_t>{2, 4});
    CHECK(along1.values() == std::vector<float>{1, 2, 5, 6, 3, 4, 7, 8});
    CHECK(slice(along1, 1, 1, 3).values() == std::vector<float>{2, 5, 4, 7});
    CHECK(slice(a, 0, 1, 2).values() == std::vector<float>{3, 4});
    CHECK_THROWS(concat({a, Tensor({3, 2})}, 1));
    CHECK_THROWS(slice(a, 0, 1, 5));
}

TEST_CASE("tensor: crop, pad, flip and rotate", "[unit][tensor]") {
    const Tensor img = ramp(2, 3);   // [[0 1 2] [3 4 5]]
    CHECK(crop(img, 1, 0, 2, 2).values() == std::vector<float>{1, 2, 4, 5});
    CHECK_THROWS(crop(img, 2, 0, 2, 1));

    CHECK(pad(img, 1, 0, 0, 1, PadMode::Constant, -1).values() == std::vector<float>{-1, 0, 1, 2, -1, 3, 4, 5, -1, -1, -1, -1});
    CHECK(pad(img, 1, 0, 0, 0, PadMode::Replicate).values() == std::vector<float>{0, 0, 1, 2, 3, 3, 4, 5});
    CHECK(pad(img, 2, 0, 0, 0, PadMode::Reflect).values() == std::vector<float>{2, 1, 0, 1, 2, 5, 4, 3, 4, 5});

    CHECK(flip(img, true).values() == std::vector<float>{2, 1, 0, 5, 4, 3});
    CHECK(flip(img, false).values() == std::vector<float>{3, 4, 5, 0, 1, 2});

    const Tensor cw = rotate90(img, 1);   // clockwise: [[3 0] [4 1] [5 2]]
    CHECK(cw.shape() == std::vector<int64_t>{1, 3, 2, 1});
    CHECK(cw.values() == std::vector<float>{3, 0, 4, 1, 5, 2});
    CHECK(rotate90(img, 2).values() == std::vector<float>{5, 4, 3, 2, 1, 0});
    CHECK(rotate90(img, -1).values() == rotate90(img, 3).values());
    CHECK(rotate90(rotate90(img, 1), 3).values() == img.values());
}

TEST_CASE("tensor: resize methods", "[unit][tensor]") {
    const Tensor img = ramp(2, 2);   // [[0 1] [2 3]]
    const Tensor nearest = resize(img, 4, 4, Resample::Nearest);
    CHECK(nearest.at({0, 0, 0, 0}) == 0.0f);
    CHECK(nearest.at({0, 1, 1, 0}) == 0.0f);
    CHECK(nearest.at({0, 0, 3, 0}) == 1.0f);
    CHECK(nearest.at({0, 3, 0, 0}) == 2.0f);
    CHECK(nearest.at({0, 3, 3, 0}) == 3.0f);

    // area averaging 4x4 -> 2x2 is the mean of each 2x2 block
    const Tensor four = ramp(4, 4);
    const Tensor area = resize(four, 2, 2, Resample::Area);
    CHECK(area.values() == std::vector<float>{2.5f, 4.5f, 10.5f, 12.5f});

    // every method keeps a constant image constant (the weights sum to one), up and down
    for (const Resample m : {Resample::Nearest, Resample::Bilinear, Resample::Bicubic, Resample::Area, Resample::Lanczos}) {
        const Tensor flat({1, 7, 5, 3}, 0.4f);
        for (const auto& size : {std::pair<int64_t, int64_t>{14, 10}, {3, 2}, {7, 5}, {20, 3}}) {
            const Tensor r = resize(flat, size.first, size.second, m);
            REQUIRE(r.shape() == std::vector<int64_t>{1, size.first, size.second, 3});
            for (const float v : r.values()) REQUIRE(v == Approx(0.4f).margin(1e-5));
        }
    }

    // bilinear upscaling of a horizontal ramp stays monotonic and inside the input range
    const Tensor line({1, 1, 4, 1}, std::vector<float>{0, 1, 2, 3});
    const Tensor up = resize(line, 1, 8, Resample::Bilinear);
    for (int64_t x = 1; x < 8; ++x) CHECK(up.at({0, 0, x, 0}) >= up.at({0, 0, x - 1, 0}));
    CHECK(up.at({0, 0, 0, 0}) == Approx(0.0f).margin(1e-5));
    CHECK(up.at({0, 0, 7, 0}) == Approx(3.0f).margin(1e-5));
    // bicubic and lanczos overshoot a little at most
    const Tensor sharp = resize(Tensor({1, 1, 4, 1}, std::vector<float>{0, 0, 1, 1}), 1, 16, Resample::Lanczos);
    for (const float v : sharp.values()) CHECK((v > -0.2f && v < 1.2f));
    // masks ([B, H, W]) work as well
    CHECK(resize(Tensor({1, 4, 4}, 1.0f), 2, 2, Resample::Bilinear).shape() == std::vector<int64_t>{1, 2, 2});
}

TEST_CASE("tensor: blur, sharpen and morphology", "[unit][tensor]") {
    Tensor flat({1, 6, 6, 1}, 0.5f);
    for (const float v : gaussianBlur(flat, 2, 1.0f).values()) CHECK(v == Approx(0.5f).margin(1e-5));

    Tensor dot({1, 9, 9, 1}, 0.0f);
    dot.at({0, 4, 4, 0}) = 1.0f;
    const Tensor blurred = gaussianBlur(dot, 3, 1.0f);
    float total = 0;
    for (const float v : blurred.values()) total += v;
    CHECK(total == Approx(1.0f).margin(1e-4));                                       // brightness is kept
    CHECK(blurred.at({0, 4, 4, 0}) > blurred.at({0, 4, 5, 0}));                        // peak in the middle
    CHECK(blurred.at({0, 4, 5, 0}) == Approx(blurred.at({0, 4, 3, 0})).margin(1e-6));  // symmetric
    CHECK(blurred.at({0, 3, 4, 0}) == Approx(blurred.at({0, 4, 5, 0})).margin(1e-6));

    const Tensor sharper = sharpen(blurred, 3, 1.0f, 1.0f);
    CHECK(sharper.at({0, 4, 4, 0}) > blurred.at({0, 4, 4, 0}));
    for (const float v : sharper.values()) CHECK((v >= 0.0f && v <= 1.0f));

    const Tensor grown = dilate(dot, 1);
    CHECK(grown.at({0, 3, 3, 0}) == 1.0f);
    CHECK(grown.at({0, 2, 4, 0}) == 0.0f);
    CHECK(erode(grown, 1).at({0, 4, 4, 0}) == 1.0f);
    CHECK(erode(grown, 1).at({0, 3, 3, 0}) == 0.0f);
}

TEST_CASE("tensor: conversion to and from the codec images", "[unit][tensor]") {
    gh::engine::Image rgb;
    rgb.width = 2;
    rgb.height = 1;
    rgb.channels = 3;
    rgb.data = {0, 128, 255, 255, 0, 51};
    const Tensor t = fromImage(rgb);
    CHECK(t.shape() == std::vector<int64_t>{1, 1, 2, 3});
    CHECK(t.at({0, 0, 0, 1}) == Approx(128 / 255.0f));
    CHECK(toImage(t).data == rgb.data);   // exact round trip

    gh::engine::Image rgba = rgb;
    rgba.channels = 4;
    rgba.data = {0, 128, 255, 10, 255, 0, 51, 255};
    CHECK(fromImage(rgba).at({0, 0, 1, 2}) == Approx(51 / 255.0f));
    const Tensor alpha = alphaMask(rgba);
    CHECK(alpha.shape() == std::vector<int64_t>{1, 1, 2});
    CHECK(alpha.at({0, 0, 0}) == Approx(10 / 255.0f));
    CHECK(alphaMask(rgb).at({0, 0, 1}) == 1.0f);
    // a one-channel mask writes out as grey
    const gh::engine::Image grey = toImage(Tensor({1, 1, 1, 1}, 0.5f));
    CHECK(grey.data == std::vector<uint8_t>{128, 128, 128});
    CHECK_THROWS(fromImage(gh::engine::Image{}));
}
