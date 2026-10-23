#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace gh::core {

// Writes the file by writing a temporary file next to it and replacing the target, so a crash never leaves a
// half-written file. Throws std::runtime_error on failure.
void writeFileAtomic(const std::filesystem::path& path, std::string_view content);

// Whole file as bytes, or nullopt if it does not exist or cannot be read.
std::optional<std::string> readFile(const std::filesystem::path& path);

}  // namespace gh::core
