#include "core/Paths.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

namespace fs = std::filesystem;

// Clean-room rule of the workflow plan (workflow-implementation-plan.md, ground rules 2 and 3): no file of the project
// names the product its node-graph design was kept apart from. The word is stored as character codes so that this file
// does not contain it either.
TEST_CASE("naming: no source, document or data file names the reference product", "[unit][naming]") {
    const std::string forbidden = {char(99), char(111), char(109), char(102), char(121)};

    fs::path root = gh::core::AppPaths::executableDir();
    while (!root.empty() && !fs::exists(root / "GraphicsHost.slnx")) {
        if (root == root.parent_path()) break;
        root = root.parent_path();
    }
    if (!fs::exists(root / "GraphicsHost.slnx")) SKIP("the repository is not above the test executable");

    const std::set<std::string> skipDirs = {"build", "extern", ".git", ".vs", "DownloadedModels", "models", "out", "dist"};
    const std::set<std::string> textExt = {".cpp", ".h", ".hpp", ".c", ".md", ".txt", ".json", ".props", ".vcxproj", ".slnx", ".targets",
                                           ".proj", ".yml", ".yaml", ".js", ".html", ".css", ".ts", ".svg", ".ps1", ".cmd"};
    int scanned = 0;
    std::vector<std::string> hits;
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied), end; it != end; ++it) {
        if (it->is_directory() && skipDirs.count(it->path().filename().string())) {
            it.disable_recursion_pending();
            continue;
        }
        if (!it->is_regular_file() || !textExt.count(it->path().extension().string())) continue;
        std::ifstream in(it->path(), std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        ++scanned;
        if (text.find(forbidden) != std::string::npos) hits.push_back(fs::relative(it->path(), root).string());
    }
    CHECK(scanned > 100);   // the walk really covered the tree
    INFO("files that name the reference product: " << (hits.empty() ? std::string("none") : hits.front()));
    CHECK(hits.empty());
}
