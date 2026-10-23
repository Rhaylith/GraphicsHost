#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace gh::core {

// SHA-256 as 64 lowercase hex characters, using the Windows CNG (bcrypt) provider.
std::string sha256Hex(std::string_view data);

// Hash of a whole file, read in 1 MiB blocks. progress (optional) gets (bytes read, total); return false from it to
// stop. Returns nullopt if the file cannot be read or the hashing was stopped.
std::optional<std::string> sha256File(const std::filesystem::path& file,
                                      const std::function<bool(uint64_t, uint64_t)>& progress = {});

}  // namespace gh::core
