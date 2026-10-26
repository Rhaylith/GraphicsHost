#include "models/ModelUpdates.h"

#include <algorithm>
#include <map>

namespace gh::models {

const char* toString(UpdateStatus s) {
    switch (s) {
        case UpdateStatus::UpToDate: return "up-to-date";
        case UpdateStatus::UpdateAvailable: return "update-available";
        case UpdateStatus::Missing: return "missing";
        case UpdateStatus::Unknown: return "unknown";
    }
    return "unknown";
}

std::vector<UpdateInfo> checkForUpdates(const std::vector<ModelCard>& cards, const HuggingFace& hub) {
    struct Candidate {
        const ModelCard* card;
        std::string repo, file, sha;
    };
    std::map<std::string, std::vector<Candidate>> byRepo;
    for (const auto& card : cards) {
        if (!card.source.is_object()) continue;
        const std::string repo = card.source.value("repo", "");
        const std::string file = card.source.value("file", "");
        // the record holds the name in the repository ("file"), the hash of what we have ("sha256")
        if (repo.empty() || file.empty() || repo.find('/') == std::string::npos) continue;
        byRepo[repo].push_back({&card, repo, file, card.source.value("sha256", "")});
    }

    std::vector<UpdateInfo> out;
    for (const auto& [repo, candidates] : byRepo) {
        HfModelInfo info;
        std::string problem;
        bool ok = true;
        try {
            info = hub.info(repo);
        } catch (const HfError& e) {
            ok = false;
            problem = e.what();
        }
        for (const auto& c : candidates) {
            UpdateInfo u;
            u.file = c.card->file;
            u.repo = repo;
            u.localSha256 = c.sha;
            if (!ok) {
                u.status = UpdateStatus::Unknown;
                u.message = problem;
                out.push_back(std::move(u));
                continue;
            }
            u.latestCommit = info.model.commit;
            const auto it = std::find_if(info.files.begin(), info.files.end(), [&](const HfFile& f) { return f.path == c.file; });
            if (it == info.files.end()) {
                u.status = UpdateStatus::Missing;
                u.message = repo + " no longer has " + c.file;
            } else if (it->sha256.empty() || c.sha.empty()) {
                u.status = UpdateStatus::Unknown;
                u.latestSha256 = it->sha256;
                u.message = "no checksum to compare (the repository or the download record has none)";
            } else {
                u.latestSha256 = it->sha256;
                if (it->sha256 == c.sha) {
                    u.status = UpdateStatus::UpToDate;
                    u.message = "up to date";
                } else {
                    u.status = UpdateStatus::UpdateAvailable;
                    u.message = repo + " now serves a different " + c.file + "; download it again to update";
                }
            }
            out.push_back(std::move(u));
        }
    }
    std::sort(out.begin(), out.end(), [](const UpdateInfo& a, const UpdateInfo& b) { return a.file < b.file; });
    return out;
}

}  // namespace gh::models
