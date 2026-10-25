#pragma once

#include "core/Paths.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

// The repository folder above the test executable (empty when the tests run somewhere else).
inline std::filesystem::path repoRoot() {
    namespace fs = std::filesystem;
    fs::path root = gh::core::AppPaths::executableDir();
    while (!root.empty() && !fs::exists(root / "GraphicsHost.slnx")) {
        if (root == root.parent_path()) break;
        root = root.parent_path();
    }
    return fs::exists(root / "GraphicsHost.slnx") ? root : fs::path();
}

inline std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
