#pragma once

#include "tensor/Tensor.h"

#include <string>

namespace gh::sampling {

// Live previews while sampling (workflow plan, section 6.4): the current estimate of the clean latent is turned into a small
// picture with a fixed 3 x C linear map per model family, which is fast and rough. (Preview decoders, small networks that
// give a better picture, are a later step.)

// A latent [batch, channels, height, width] to an image [batch, height, width, 3] in 0..1.
// Families: "SD 1.x", "SDXL"; any other 4-channel latent uses the SD 1.x map, other channel counts show the first three
// channels stretched to the range -3..3.
tensor::Tensor latentToRgb(const tensor::Tensor& latent, const std::string& family);

// The same, shrunk so the longer side is at most maxSize pixels (never enlarged).
tensor::Tensor previewImage(const tensor::Tensor& latent, const std::string& family, int maxSize);

// The 4 x 3 map of a family, row per latent channel (empty for families without one).
std::vector<float> latentRgbFactors(const std::string& family);

}  // namespace gh::sampling
