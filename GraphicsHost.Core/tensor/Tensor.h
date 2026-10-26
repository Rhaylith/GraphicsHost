#pragma once

#include "engine/GenerationRequest.h"   // engine::Image

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace gh::tensor {

// A small owned tensor: float32, row-major, up to 5 dimensions. Images are [B, H, W, C] (channel last) with values in
// 0..1; masks [B, H, W]; latents keep the layout the model uses. The operations below are the host-side maths of the
// workflow nodes (image, mask and latent nodes), not model inference. Loops over large tensors run in parallel and are
// compiled for AVX2 (/arch:AVX2 for all first-party code).
class Tensor {
public:
    static constexpr int kMaxDims = 5;

    Tensor() = default;
    explicit Tensor(std::vector<int64_t> shape, float fill = 0.0f);
    Tensor(std::vector<int64_t> shape, std::vector<float> data);

    bool empty() const { return data_.empty(); }
    int ndim() const { return static_cast<int>(shape_.size()); }
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(int axis) const { return shape_.at(static_cast<size_t>(axis)); }
    int64_t numel() const { return static_cast<int64_t>(data_.size()); }
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    const std::vector<float>& values() const { return data_; }

    // Row-major offset of an index (one entry per dimension).
    int64_t offset(std::initializer_list<int64_t> index) const;
    float at(std::initializer_list<int64_t> index) const { return data_[static_cast<size_t>(offset(index))]; }
    float& at(std::initializer_list<int64_t> index) { return data_[static_cast<size_t>(offset(index))]; }

    bool sameShape(const Tensor& other) const { return shape_ == other.shape_; }

    static Tensor zeros(std::vector<int64_t> shape) { return Tensor(std::move(shape), 0.0f); }
    static Tensor full(std::vector<int64_t> shape, float value) { return Tensor(std::move(shape), value); }

private:
    std::vector<int64_t> shape_;
    std::vector<float> data_;
};

// ---- element-wise (shapes must be equal, or one side a single value) ---------------------------------------------------
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& a, float factor);
Tensor addScalar(const Tensor& a, float value);
Tensor clamp(const Tensor& a, float low, float high);
Tensor lerp(const Tensor& a, const Tensor& b, float t);             // a + (b - a) * t
Tensor slerp(const Tensor& a, const Tensor& b, float t);            // along the angle between the two flattened tensors per batch item
Tensor invert(const Tensor& a);                                     // 1 - a

// ---- statistics over the last axis treated as channels: [.., C] -> per-channel values --------------------------------
std::vector<float> channelMean(const Tensor& a);
std::vector<float> channelStd(const Tensor& a);                     // population standard deviation

// ---- structure ----------------------------------------------------------------------------------------------------------
Tensor concat(const std::vector<Tensor>& parts, int axis);
Tensor slice(const Tensor& a, int axis, int64_t begin, int64_t end);       // [begin, end) along axis
// Image-like [B, H, W, C] (or [B, H, W]) operations:
Tensor crop(const Tensor& a, int64_t x, int64_t y, int64_t width, int64_t height);
enum class PadMode { Constant, Replicate, Reflect };
Tensor pad(const Tensor& a, int64_t left, int64_t top, int64_t right, int64_t bottom, PadMode mode = PadMode::Constant, float value = 0.0f);
Tensor flip(const Tensor& a, bool horizontal);                      // horizontal: mirror left-right, else top-bottom
Tensor rotate90(const Tensor& a, int quarterTurns);                 // clockwise, any integer

// ---- resampling and filters, on [B, H, W, C] or [B, H, W] ------------------------------------------------------------
enum class Resample { Nearest, Bilinear, Bicubic, Area, Lanczos };
Tensor resize(const Tensor& a, int64_t height, int64_t width, Resample method);
Tensor gaussianBlur(const Tensor& a, int radius, float sigma);      // separable, edges replicated
Tensor sharpen(const Tensor& a, int radius, float sigma, float amount);   // unsharp mask, result clamped to 0..1
Tensor dilate(const Tensor& a, int radius);                         // grey-scale maximum over a (2r+1) square
Tensor erode(const Tensor& a, int radius);

// ---- conversion with the codec's 8-bit images -----------------------------------------------------------------------------
// [1, H, W, 3] in 0..1 (an alpha channel, if the image has one, is dropped; see alphaMask).
Tensor fromImage(const engine::Image& image);
// [1, H, W] in 0..1 from the alpha channel of an RGBA image (all ones for RGB).
Tensor alphaMask(const engine::Image& image);
// Batch item `index` of an [B, H, W, 3|1] tensor as an 8-bit RGB image (values clamped and rounded).
engine::Image toImage(const Tensor& a, int64_t index = 0);

}  // namespace gh::tensor
