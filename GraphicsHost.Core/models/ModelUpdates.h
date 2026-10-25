#pragma once

#include "models/HuggingFace.h"
#include "models/ModelCard.h"

#include <string>
#include <vector>

namespace gh::models {

enum class UpdateStatus {
    UpToDate,          // the file in the repository is the one that was downloaded
    UpdateAvailable,   // the repository now serves a different file under that name
    Missing,           // the repository no longer has the file (renamed or deleted)
    Unknown,           // could not find out (offline, gated without access, rate limit)
};
const char* toString(UpdateStatus status);

struct UpdateInfo {
    std::string file;          // local file name
    std::string repo;
    UpdateStatus status = UpdateStatus::Unknown;
    std::string localSha256;   // recorded when it was downloaded
    std::string latestSha256;  // what the repository serves now
    std::string latestCommit;
    std::string message;       // for people
};

// Checks the models that were downloaded with a source record (repo + file + hash). One request per repository.
// A model is up to date when the repository still serves a file with the recorded hash, even if other files in the
// repository changed. Models without a record are not listed.
std::vector<UpdateInfo> checkForUpdates(const std::vector<ModelCard>& cards, const HuggingFace& hub);

}  // namespace gh::models
