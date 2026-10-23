#pragma once

#include "engine/ImageEngine.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gh::engine {

enum class ImageFormat { Png, Jpeg, Webp };

// "png", "jpeg" (also "jpg") or "webp"; nullopt for anything else.
std::optional<ImageFormat> parseImageFormat(std::string_view name);
const char* extensionFor(ImageFormat format);  // ".png", ".jpg", ".webp"
const char* mimeTypeFor(ImageFormat format);

// Encodes RGB or RGBA pixels. metadata, when given, is embedded so the picture can be reproduced later:
//   PNG  : a tEXt chunk with keyword "graphicshost"
//   JPEG : a COM (comment) marker
//   WebP : an XMP chunk
// quality: 1..100 for JPEG and lossy WebP (100 = lossless WebP); ignored for PNG, which is always lossless.
// Throws std::runtime_error when the pixels are inconsistent or an encoder fails.
std::vector<uint8_t> encodeImage(const Image& image, ImageFormat format, const std::optional<std::string>& metadata = {},
                                 int quality = 92);

// Format by content (not by file name); nullopt if the bytes are none of the three.
std::optional<ImageFormat> detectImageFormat(const std::vector<uint8_t>& bytes);

// The metadata string embedded by encodeImage, if there is one.
std::optional<std::string> readImageMetadata(const std::vector<uint8_t>& bytes);

// Decodes to RGB (3 channels). Throws std::runtime_error on damaged or unknown data.
Image decodeImage(const std::vector<uint8_t>& bytes);

}  // namespace gh::engine
