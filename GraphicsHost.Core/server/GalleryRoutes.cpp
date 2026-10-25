#include "engine/ImageCodec.h"
#include "server/HttpCommon.h"
#include "server/RouteContext.h"

#include <algorithm>
#include <fstream>
#include <iterator>

namespace gh::server {

using Json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

Json entryJson(const gallery::GalleryEntry& e) {
    return {{"id", e.id},         {"path", e.path},       {"created", e.created}, {"prompt", e.prompt}, {"model", e.model},
            {"family", e.family}, {"runtime", e.runtime}, {"seed", e.seed},       {"width", e.width},   {"height", e.height},
            {"format", e.format}, {"parameters", e.parameters},
            {"url", "/outputs/" + e.path},
            {"thumbnailUrl", "/api/v1/gallery/" + std::to_string(e.id) + "/thumbnail"}};
}

bool readAll(const fs::path& path, std::vector<uint8_t>& bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

std::string mimeForPath(const fs::path& p) {
    const std::string ext = p.extension().string();
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".webp") return "image/webp";
    if (ext == ".json") return "application/json";
    return "application/octet-stream";
}

// Box-filter shrink so that the longer side is at most maxSide.
engine::Image shrink(const engine::Image& src, uint32_t maxSide) {
    const uint32_t longer = std::max(src.width, src.height);
    if (longer <= maxSide || src.width == 0 || src.height == 0) return src;
    const double scale = static_cast<double>(maxSide) / longer;
    engine::Image out;
    out.width = std::max<uint32_t>(1, static_cast<uint32_t>(src.width * scale));
    out.height = std::max<uint32_t>(1, static_cast<uint32_t>(src.height * scale));
    out.channels = 3;
    out.data.resize(static_cast<size_t>(out.width) * out.height * 3);
    for (uint32_t y = 0; y < out.height; ++y) {
        const uint32_t y0 = static_cast<uint32_t>(static_cast<uint64_t>(y) * src.height / out.height);
        const uint32_t y1 = std::max(y0 + 1, static_cast<uint32_t>(static_cast<uint64_t>(y + 1) * src.height / out.height));
        for (uint32_t x = 0; x < out.width; ++x) {
            const uint32_t x0 = static_cast<uint32_t>(static_cast<uint64_t>(x) * src.width / out.width);
            const uint32_t x1 = std::max(x0 + 1, static_cast<uint32_t>(static_cast<uint64_t>(x + 1) * src.width / out.width));
            uint64_t sum[3] = {0, 0, 0};
            uint32_t n = 0;
            for (uint32_t yy = y0; yy < std::min(y1, src.height); ++yy) {
                for (uint32_t xx = x0; xx < std::min(x1, src.width); ++xx) {
                    const size_t at = (static_cast<size_t>(yy) * src.width + xx) * src.channels;
                    for (int c = 0; c < 3; ++c) sum[c] += src.data[at + c];
                    ++n;
                }
            }
            for (int c = 0; c < 3; ++c) out.data[(static_cast<size_t>(y) * out.width + x) * 3 + c] = static_cast<uint8_t>(n ? sum[c] / n : 0);
        }
    }
    return out;
}

bool idFrom(const httplib::Request& req, int64_t& id) {
    try {
        id = std::stoll(req.matches[1]);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

void registerGalleryRoutes(httplib::Server& server, RouteContext& ctx) {
    server.Get("/api/v1/gallery", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        gallery::GalleryQuery q;
        q.text = req.get_param_value("text");
        try {
            if (req.has_param("limit")) q.limit = std::clamp(std::stoi(req.get_param_value("limit")), 1, 500);
            if (req.has_param("offset")) q.offset = std::max(0, std::stoi(req.get_param_value("offset")));
        } catch (const std::exception&) {
            sendError(res, 400, "invalid_request_error", "limit and offset must be numbers", "invalid_parameters");
            return;
        }
        Json items = Json::array();
        for (const auto& e : ctx.host.gallery().list(q)) items.push_back(entryJson(e));
        sendJson(res, {{"items", items}, {"total", ctx.host.gallery().count(q.text)}, {"limit", q.limit}, {"offset", q.offset}});
    });

    server.Get(R"(/api/v1/gallery/(\d+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        int64_t id = 0;
        const auto entry = idFrom(req, id) ? ctx.host.gallery().get(id) : std::nullopt;
        if (!entry) {
            sendError(res, 404, "not_found", "no such image", "image_not_found");
            return;
        }
        sendJson(res, entryJson(*entry));
    });

    server.Delete(R"(/api/v1/gallery/(\d+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        int64_t id = 0;
        if (!idFrom(req, id) || !ctx.host.gallery().remove(id)) {
            sendError(res, 404, "not_found", "no such image", "image_not_found");
            return;
        }
        res.status = 204;
    });

    server.Get(R"(/api/v1/gallery/(\d+)/(image|thumbnail))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        int64_t id = 0;
        const auto entry = idFrom(req, id) ? ctx.host.gallery().get(id) : std::nullopt;
        std::vector<uint8_t> bytes;
        if (!entry || !readAll(ctx.host.gallery().absolutePath(*entry), bytes)) {
            sendError(res, 404, "not_found", "no such image", "image_not_found");
            return;
        }
        std::string mime = mimeForPath(entry->path);
        if (req.matches[2] == "thumbnail") {
            int side = 256;
            try {
                if (req.has_param("size")) side = std::clamp(std::stoi(req.get_param_value("size")), 16, 1024);
                const engine::Image small = shrink(engine::decodeImage(bytes), static_cast<uint32_t>(side));
                bytes = engine::encodeImage(small, engine::ImageFormat::Jpeg, std::nullopt, 80);
                mime = "image/jpeg";
            } catch (const std::exception& e) {
                sendError(res, 500, "server_error", std::string("cannot make a thumbnail: ") + e.what(), "thumbnail_failed");
                return;
            }
        }
        res.set_header("Cache-Control", "private, max-age=3600");
        res.set_content(std::string(bytes.begin(), bytes.end()), mime);
    });

    // The image files themselves, as the "url" fields of the API point to them.
    server.Get(R"(/outputs/(.+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const std::string relative = req.matches[1];
        std::error_code ec;
        const fs::path root = fs::weakly_canonical(ctx.host.gallery().root(), ec);
        const fs::path file = fs::weakly_canonical(root / fs::path(std::u8string(relative.begin(), relative.end())), ec);
        const auto [rootEnd, fileAt] = std::mismatch(root.begin(), root.end(), file.begin(), file.end());
        std::vector<uint8_t> bytes;
        if (ec || rootEnd != root.end() || file == root || mimeForPath(file) == "application/octet-stream" || !fs::is_regular_file(file, ec) || !readAll(file, bytes)) {
            sendError(res, 404, "not_found", "no such file", "file_not_found");
            return;
        }
        res.set_header("Cache-Control", "private, max-age=3600");
        res.set_content(std::string(bytes.begin(), bytes.end()), mimeForPath(file));
    });
}

}  // namespace gh::server
