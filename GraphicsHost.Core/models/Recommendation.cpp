#include "models/Recommendation.h"

#include <algorithm>

namespace gh::models {

Recommendation recommendCheckpoint(const Catalog& catalog, const HardwareBudget& budget) {
    struct Candidate {
        const CatalogEntry* entry;
        FitHint fit;
    };
    std::vector<Candidate> open, any;
    for (const auto& e : catalog.entries()) {
        if (e.kind != "checkpoint" || e.gated) continue;
        Candidate c{&e, fitHintFor(e.sizeBytes, e.family, e.kind, budget)};
        any.push_back(c);
        if (e.recommended) open.push_back(c);
    }
    if (open.empty()) open = any;
    Recommendation r;
    if (open.empty()) {
        r.why = "the catalog has no checkpoint to suggest";
        return r;
    }
    // fits the GPU as it is: the biggest such model (SDXL looks better than SD 1.x)
    const bool gpu = budget.gpuMemoryBytes > 0;
    std::vector<Candidate> good;
    for (const auto& c : open) {
        if (c.fit.status == engine::FitStatus::Fits) good.push_back(c);
    }
    if (gpu && !good.empty()) {
        const auto best = std::max_element(good.begin(), good.end(), [](const Candidate& a, const Candidate& b) { return a.entry->sizeBytes < b.entry->sizeBytes; });
        r.entry = best->entry;
        r.fit = best->fit;
        r.why = "it fits the memory of your graphics card";
        return r;
    }
    // otherwise whatever is smallest and still possible
    std::vector<Candidate> possible;
    for (const auto& c : open) {
        if (c.fit.status != engine::FitStatus::WontFit) possible.push_back(c);
    }
    if (possible.empty()) possible = open;
    const auto smallest = std::min_element(possible.begin(), possible.end(), [](const Candidate& a, const Candidate& b) { return a.entry->sizeBytes < b.entry->sizeBytes; });
    r.entry = smallest->entry;
    r.fit = smallest->fit;
    r.why = gpu ? "the larger models do not fit your graphics card, this one is the lightest"
                : "no usable graphics card was found: this is the lightest model, it runs on the CPU";
    return r;
}

}  // namespace gh::models
