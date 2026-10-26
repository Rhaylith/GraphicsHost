#pragma once

#include "net/HttpClient.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace gh::models {

enum class HfErrorCode { Network, NotFound, Unauthorized, Gated, RateLimited, BadAnswer, Other };

class HfError : public std::runtime_error {
public:
    HfError(HfErrorCode code, const std::string& message, int status = 0) : std::runtime_error(message), code_(code), status_(status) {}
    HfErrorCode code() const { return code_; }
    int status() const { return status_; }

private:
    HfErrorCode code_;
    int status_;
};

struct HfModel {
    std::string id;                  // "publisher/name"
    int64_t downloads = 0;
    int64_t likes = 0;
    std::string gated;               // "" (open), "auto" or "manual"
    std::string license;             // from the "license:<x>" tag
    std::string pipeline;            // "text-to-image"
    std::string library;             // "diffusers", ...
    std::vector<std::string> tags;
    std::string lastModified;
    std::string commit;              // sha of the revision the file list belongs to (model info only)
    bool isGated() const { return !gated.empty(); }
};

struct HfFile {
    std::string path;                // as in the repository, may contain folders
    uint64_t sizeBytes = 0;
    std::string sha256;              // of the content (LFS object id); empty for small non-LFS files
};

struct HfModelInfo {
    HfModel model;
    std::vector<HfFile> files;
};

// What a file in a repository probably is, by name and path: "checkpoint", "vae", "lora", "other".
std::string guessFileKind(const std::string& path);

// Files that look like something this application can use: .safetensors / .ckpt / .gguf above 20 MB, outside the
// component folders of a diffusers repository (unet/, text_encoder/, ...), which are parts of a pipeline, not
// stand-alone checkpoints.
std::vector<HfFile> usableFiles(const HfModelInfo& info);

// The Hugging Face web API (the part this application needs). baseUrl is a parameter so tests can use a local server.
class HuggingFace {
public:
    explicit HuggingFace(std::string token = {}, std::string baseUrl = "https://huggingface.co");

    // Text-to-image models matching the query, most downloaded first. Throws HfError.
    std::vector<HfModel> search(const std::string& query, int limit = 20) const;
    // One repository with its files (sizes and hashes). A gated repository without access still answers: isGated()
    // is then true and the file list may be empty. Throws HfError.
    HfModelInfo info(const std::string& repo) const;

    // https://huggingface.co/<repo>/resolve/<revision or main>/<path>
    std::string fileUrl(const std::string& repo, const std::string& path, const std::string& revision = {}) const;

    // Can this token download the file? Returns 200 for yes; throws HfError(Unauthorized / Gated / NotFound) otherwise.
    void checkAccess(const std::string& repo, const std::string& path, const std::string& revision = {}) const;

    // Who the token belongs to (GET /api/whoami-v2): the user name. Throws HfError(Unauthorized) for a bad token.
    std::string whoami() const;

    const std::string& token() const { return token_; }

private:
    net::HttpOptions options() const;
    std::string get(const std::string& path, const std::string& what) const;

    std::string token_;
    std::string base_;
};

}  // namespace gh::models

namespace gh::models {

// The Hugging Face client the application uses: the stored access token (HF_TOKEN or the Credential Manager) and
// GRAPHICSHOST_HUB_URL (a mirror, or a test server) when set.
HuggingFace makeConfiguredHub();

}  // namespace gh::models
