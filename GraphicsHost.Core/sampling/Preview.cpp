#include "sampling/Preview.h"

#include <algorithm>

namespace gh::sampling {

using tensor::Tensor;

std::vector<float> latentRgbFactors(const std::string& family) {
    if (family == "SDXL") {
        return {0.3651f, 0.4232f, 0.4341f,
                -0.2533f, -0.0042f, 0.1068f,
                0.1076f, 0.1111f, -0.0362f,
                -0.3165f, -0.2492f, -0.2188f};
    }
    if (family == "SD 1.x") {
        return {0.3512f, 0.2297f, 0.3227f,
                0.3250f, 0.4974f, 0.2350f,
                -0.2829f, 0.1762f, 0.2721f,
                -0.2120f, -0.2616f, -0.7177f};
    }
    return {};
}

Tensor latentToRgb(const Tensor& latent, const std::string& family) {
    if (latent.ndim() != 4) return {};
    const int64_t B = latent.dim(0), C = latent.dim(1), H = latent.dim(2), W = latent.dim(3);
    std::vector<float> factors = latentRgbFactors(family);
    const bool mapped = C == 4;
    if (mapped && factors.empty()) factors = latentRgbFactors("SD 1.x");
    Tensor out({B, H, W, 3});
    for (int64_t b = 0; b < B; ++b)
        for (int64_t y = 0; y < H; ++y)
            for (int64_t x = 0; x < W; ++x)
                for (int rgb = 0; rgb < 3; ++rgb) {
                    float v;
                    if (mapped) {
                        float sum = 0;
                        for (int64_t c = 0; c < C; ++c) sum += latent.data()[((b * C + c) * H + y) * W + x] * factors[static_cast<size_t>(c * 3 + rgb)];
                        v = (sum + 1.0f) * 0.5f;
                    } else {
                        const int64_t c = std::min<int64_t>(rgb, C - 1);
                        v = (latent.data()[((b * C + c) * H + y) * W + x] + 3.0f) / 6.0f;
                    }
                    out.data()[((b * H + y) * W + x) * 3 + rgb] = std::clamp(v, 0.0f, 1.0f);
                }
    return out;
}

Tensor previewImage(const Tensor& latent, const std::string& family, int maxSize) {
    Tensor rgb = latentToRgb(latent, family);
    if (rgb.empty()) return rgb;
    const int64_t H = rgb.dim(1), W = rgb.dim(2);
    const int64_t longest = std::max(H, W);
    if (maxSize <= 0 || longest <= maxSize) return rgb;
    const double scale = static_cast<double>(maxSize) / static_cast<double>(longest);
    return tensor::resize(rgb, std::max<int64_t>(1, static_cast<int64_t>(H * scale)), std::max<int64_t>(1, static_cast<int64_t>(W * scale)), tensor::Resample::Area);
}

}  // namespace gh::sampling
