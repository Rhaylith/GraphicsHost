#include "models/HuggingFace.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>

namespace gh::models {

using Json = nlohmann::json;

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string text(const Json& j, const char* key) {
    const auto it = j.find(key);
    return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
}

int64_t number(const Json& j, const char* key) {
    const auto it = j.find(key);
    return it != j.end() && it->is_number() ? it->get<int64_t>() : 0;
}

HfModel parseModel(const Json& j) {
    HfModel m;
    m.id = text(j, "id");
    if (m.id.empty()) m.id = text(j, "modelId");
    m.downloads = number(j, "downloads");
    m.likes = number(j, "likes");
    m.pipeline = text(j, "pipeline_tag");
    m.library = text(j, "library_name");
    m.lastModified = text(j, "lastModified");
    m.commit = text(j, "sha");
    // "gated" is false, or the string "auto" / "manual"
    if (const auto it = j.find("gated"); it != j.end() && it->is_string()) m.gated = it->get<std::string>();
    if (const auto it = j.find("tags"); it != j.end() && it->is_array()) {
        for (const auto& t : *it) {
            if (!t.is_string()) continue;
            const std::string tag = t.get<std::string>();
            m.tags.push_back(tag);
            if (tag.rfind("license:", 0) == 0 && m.license.empty()) m.license = tag.substr(8);
        }
    }
    if (m.license.empty()) {
        if (const auto card = j.find("cardData"); card != j.end() && card->is_object()) m.license = text(*card, "license");
    }
    return m;
}

[[noreturn]] void raise(const net::HttpResponse& r, const std::string& what, const std::string& body) {
    if (!r.error.empty()) {
        throw HfError(HfErrorCode::Network, "cannot reach Hugging Face for " + what + ": " + r.error);
    }
    if (r.status == 401) {
        throw HfError(HfErrorCode::Unauthorized, what + ": Hugging Face did not accept the access token (HTTP 401); log in again", 401);
    }
    if (r.status == 403) {
        throw HfError(HfErrorCode::Gated, what + ": access is refused (HTTP 403); this model is gated: accept its terms on the model page, with the account the token belongs to", 403);
    }
    if (r.status == 404) {
        throw HfError(HfErrorCode::NotFound, what + ": not found (HTTP 404)", 404);
    }
    if (r.status == 429) {
        throw HfError(HfErrorCode::RateLimited, "Hugging Face is rate limiting requests (HTTP 429); try again in a minute", 429);
    }
    (void)body;
    throw HfError(HfErrorCode::Other, what + ": HTTP " + std::to_string(r.status), r.status);
}

}  // namespace

std::string guessFileKind(const std::string& path) {
    const std::string p = lower(path);
    if (p.find("lora") != std::string::npos) return "lora";
    if (p.find("vae") != std::string::npos) return "vae";
    const std::string ext = p.substr(p.find_last_of('.') == std::string::npos ? p.size() : p.find_last_of('.'));
    if (ext == ".safetensors" || ext == ".ckpt" || ext == ".gguf" || ext == ".sft") return "checkpoint";
    return "other";
}

std::vector<HfFile> usableFiles(const HfModelInfo& info) {
    static const char* componentFolders[] = {"unet/", "text_encoder/", "text_encoder_2/", "text_encoder_3/", "vae/",
                                             "safety_checker/", "image_encoder/", "transformer/", "controlnet/", "scheduler/", "tokenizer/"};
    std::vector<HfFile> out;
    for (const auto& f : info.files) {
        const std::string p = lower(f.path);
        const auto dot = p.find_last_of('.');
        const std::string ext = dot == std::string::npos ? "" : p.substr(dot);
        if (ext != ".safetensors" && ext != ".ckpt" && ext != ".gguf" && ext != ".sft") continue;
        if (f.sizeBytes < 20ull * 1024 * 1024) continue;
        const bool inComponent = std::any_of(std::begin(componentFolders), std::end(componentFolders),
                                             [&](const char* folder) { return p.rfind(folder, 0) == 0; });
        if (inComponent) continue;
        out.push_back(f);
    }
    return out;
}

HuggingFace::HuggingFace(std::string token, std::string baseUrl) : token_(std::move(token)), base_(std::move(baseUrl)) {
    while (!base_.empty() && base_.back() == '/') base_.pop_back();
}

net::HttpOptions HuggingFace::options() const {
    net::HttpOptions o;
    o.bearerToken = token_;
    o.timeoutSeconds = 30;
    return o;
}

std::string HuggingFace::get(const std::string& path, const std::string& what) const {
    const net::HttpResponse r = net::httpGet(base_ + path, options());
    if (!r.ok()) raise(r, what, r.body);
    return r.body;
}

std::vector<HfModel> HuggingFace::search(const std::string& query, int limit) const {
    const std::string path = "/api/models?search=" + net::urlEncode(query) + "&filter=text-to-image&sort=downloads&direction=-1&limit=" +
                             std::to_string(std::clamp(limit, 1, 100));
    const Json doc = Json::parse(get(path, "the search"), nullptr, false);
    if (!doc.is_array()) {
        throw HfError(HfErrorCode::BadAnswer, "Hugging Face answered the search with something that is not a list");
    }
    std::vector<HfModel> out;
    for (const auto& item : doc) {
        if (!item.is_object()) continue;
        HfModel m = parseModel(item);
        if (!m.id.empty()) out.push_back(std::move(m));
    }
    return out;
}

HfModelInfo HuggingFace::info(const std::string& repo) const {
    if (repo.find('/') == std::string::npos || repo.find("..") != std::string::npos) {
        throw HfError(HfErrorCode::NotFound, "'" + repo + "' is not a repository name (publisher/name)");
    }
    const std::string body = get("/api/models/" + repo + "?blobs=true", repo);
    const Json doc = Json::parse(body, nullptr, false);
    if (!doc.is_object()) {
        throw HfError(HfErrorCode::BadAnswer, "Hugging Face answered " + repo + " with something that is not a model description");
    }
    HfModelInfo info;
    info.model = parseModel(doc);
    if (info.model.id.empty()) info.model.id = repo;
    if (const auto siblings = doc.find("siblings"); siblings != doc.end() && siblings->is_array()) {
        for (const auto& s : *siblings) {
            if (!s.is_object()) continue;
            HfFile f;
            f.path = text(s, "rfilename");
            f.sizeBytes = static_cast<uint64_t>(number(s, "size"));
            if (const auto lfs = s.find("lfs"); lfs != s.end() && lfs->is_object()) {
                f.sha256 = text(*lfs, "sha256");
                if (f.sizeBytes == 0) f.sizeBytes = static_cast<uint64_t>(number(*lfs, "size"));
            }
            if (!f.path.empty()) info.files.push_back(std::move(f));
        }
    }
    return info;
}

std::string HuggingFace::fileUrl(const std::string& repo, const std::string& path, const std::string& revision) const {
    return base_ + "/" + repo + "/resolve/" + (revision.empty() ? "main" : revision) + "/" + path;
}

void HuggingFace::checkAccess(const std::string& repo, const std::string& path, const std::string& revision) const {
    const net::HttpResponse r = net::httpHead(fileUrl(repo, path, revision), options());
    if (!r.ok()) raise(r, repo + "/" + path, r.body);
}

std::string HuggingFace::whoami() const {
    const Json doc = Json::parse(get("/api/whoami-v2", "the token check"), nullptr, false);
    const std::string name = doc.is_object() ? text(doc, "name") : std::string();
    if (name.empty()) throw HfError(HfErrorCode::BadAnswer, "Hugging Face did not say who the token belongs to");
    return name;
}

}  // namespace gh::models
