#pragma once

#include "models/ModelCard.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace gh::models {

// Extra model folders per kind (workflow plan, section 6.3), read from model-paths.json in the data folder:
//   { "loras": ["\\\\nas\\loras", "D:/more/loras"], "checkpoints": ["E:/AI/checkpoints"] }
// Every file below such a folder is of that kind, whatever its header says, and is named by its path below the folder.
using ExtraRoots = std::map<Kind, std::vector<std::filesystem::path>>;

// A missing file gives no roots. Problems (not JSON, an unknown kind name, a value that is not a list of folders) are
// described in `warnings` and skipped; they never throw.
ExtraRoots loadModelPaths(const std::filesystem::path& file, std::vector<std::string>* warnings = nullptr);

}  // namespace gh::models
