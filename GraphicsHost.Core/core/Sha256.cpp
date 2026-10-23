#include "core/Sha256.h"

#include <windows.h>
#include <bcrypt.h>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace gh::core {

namespace {

class Hasher {
public:
    Hasher() {
        if (BCryptOpenAlgorithmProvider(&alg_, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
            throw std::runtime_error("SHA-256 is not available");
        }
        if (BCryptCreateHash(alg_, &hash_, nullptr, 0, nullptr, 0, 0) < 0) {
            BCryptCloseAlgorithmProvider(alg_, 0);
            throw std::runtime_error("cannot create a SHA-256 hash");
        }
    }
    ~Hasher() {
        BCryptDestroyHash(hash_);
        BCryptCloseAlgorithmProvider(alg_, 0);
    }
    Hasher(const Hasher&) = delete;
    Hasher& operator=(const Hasher&) = delete;

    void update(const void* data, size_t size) {
        if (BCryptHashData(hash_, static_cast<PUCHAR>(const_cast<void*>(data)), static_cast<ULONG>(size), 0) < 0) {
            throw std::runtime_error("SHA-256 update failed");
        }
    }
    std::string finish() {
        UCHAR digest[32];
        if (BCryptFinishHash(hash_, digest, sizeof(digest), 0) < 0) {
            throw std::runtime_error("SHA-256 finish failed");
        }
        static const char* hex = "0123456789abcdef";
        std::string out;
        for (UCHAR b : digest) {
            out += hex[b >> 4];
            out += hex[b & 15];
        }
        return out;
    }

private:
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_HASH_HANDLE hash_ = nullptr;
};

}  // namespace

std::string sha256Hex(std::string_view data) {
    Hasher h;
    // BCryptHashData takes a 32-bit length: feed big inputs in blocks
    for (size_t pos = 0; pos < data.size(); pos += (1u << 30)) {
        h.update(data.data() + pos, std::min<size_t>(1u << 30, data.size() - pos));
    }
    return h.finish();
}

std::optional<std::string> sha256File(const std::filesystem::path& file,
                                      const std::function<bool(uint64_t, uint64_t)>& progress) {
    std::error_code ec;
    const uint64_t total = std::filesystem::file_size(file, ec);
    std::ifstream in(file, std::ios::binary);
    if (ec || !in) {
        return std::nullopt;
    }
    Hasher h;
    std::vector<char> block(1 << 20);
    uint64_t done = 0;
    while (in) {
        in.read(block.data(), static_cast<std::streamsize>(block.size()));
        const auto got = static_cast<size_t>(in.gcount());
        if (got == 0) break;
        h.update(block.data(), got);
        done += got;
        if (progress && !progress(done, total)) {
            return std::nullopt;
        }
    }
    return h.finish();
}

}  // namespace gh::core
