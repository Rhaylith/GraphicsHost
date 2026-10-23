#pragma once

// Pictures that come with a generation request: the source image of image to image (and, later, a mask).

#include "engine/ImageCodec.h"
#include "host/AppHost.h"
#include "server/Base64.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace gh::server {

// A decoded picture (RGB). Throws std::invalid_argument when the bytes are none of PNG, JPEG or WebP.
inline std::shared_ptr<const engine::Image> imageFromBytes(const std::vector<uint8_t>& bytes, const std::string& what) {
    if (!engine::detectImageFormat(bytes)) throw std::invalid_argument(what + " is not a PNG, JPEG or WebP image");
    try {
        return std::make_shared<const engine::Image>(engine::decodeImage(bytes));
    } catch (const std::exception& e) {
        throw std::invalid_argument(what + " cannot be read: " + e.what());
    }
}

// A PNG, JPEG or WebP as base64 (a "data:image/png;base64," prefix is fine). Throws std::invalid_argument.
inline std::shared_ptr<const engine::Image> imageFromBase64(const std::string& text, const std::string& what) {
    std::string payload = text;
    if (payload.rfind("data:", 0) == 0) {
        const auto comma = payload.find(',');
        if (comma == std::string::npos) throw std::invalid_argument(what + " is not a valid data URL");
        payload.erase(0, comma + 1);
    }
    std::vector<uint8_t> bytes;
    if (!base64Decode(payload, bytes) || bytes.empty()) throw std::invalid_argument(what + " is not valid base64");
    return imageFromBytes(bytes, what);
}

inline std::shared_ptr<const engine::Image> imageFromGallery(host::AppHost& host, int64_t id, const std::string& what) {
    const auto entry = host.gallery().get(id);
    if (!entry) throw std::invalid_argument(what + ": there is no gallery image " + std::to_string(id));
    std::ifstream in(host.gallery().absolutePath(*entry), std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return imageFromBytes(bytes, what);
}

// The source image of a JSON request: "init_image" (also "initImage", "image") as base64, or "init_image_gallery_id" /
// "initImageGalleryId" to start from something already in the gallery. Throws std::invalid_argument.
inline void attachInputImages(host::AppHost& host, const nlohmann::json& body, host::GenerateCall& call) {
    for (const char* key : {"init_image", "initImage", "image"}) {
        if (body.contains(key) && !body[key].is_null()) {
            if (!body[key].is_string()) throw std::invalid_argument(std::string(key) + " must be a base64 string");
            call.initImage = imageFromBase64(body[key].get<std::string>(), key);
            return;
        }
    }
    for (const char* key : {"init_image_gallery_id", "initImageGalleryId"}) {
        if (body.contains(key) && !body[key].is_null()) {
            if (!body[key].is_number_integer()) throw std::invalid_argument(std::string(key) + " must be a whole number");
            call.initImage = imageFromGallery(host, body[key].get<int64_t>(), key);
            return;
        }
    }
}

}  // namespace gh::server
