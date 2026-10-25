#pragma once

#include "models/Catalog.h"
#include "models/FitHints.h"

#include <optional>

namespace gh::models {

struct Recommendation {
    const CatalogEntry* entry = nullptr;   // nullptr when the catalog has no checkpoint at all
    FitHint fit;
    std::string why;                       // one sentence for the first-run wizard
};

// The checkpoint to suggest to a new user: among the recommended, open (not gated) catalog checkpoints, the best one that
// runs on the GPU as it is (SDXL before SD 1.x); without a usable GPU, or when nothing fits it, the one that runs
// best on this computer (the smallest).
Recommendation recommendCheckpoint(const Catalog& catalog, const HardwareBudget& budget);

}  // namespace gh::models
