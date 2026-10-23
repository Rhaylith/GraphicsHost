#pragma once

#include "engine/GenerationRequest.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace gh::engine {

struct SafetyVerdict {
    bool allowed = true;
    std::string reason;   // for people: what was matched, when it was not
};

// Decides whether a generation may run. The filter is optional and off by default (settings.safety.enabled, or the
// request's own safetyFilter flag); when it is on, a refused request never reaches the engine.
class ISafetyFilter {
public:
    virtual ~ISafetyFilter() = default;
    virtual SafetyVerdict checkPrompt(const std::string& prompt) const = 0;
    // A hook for a picture classifier. The built-in filter has none: it lets every picture through.
    virtual SafetyVerdict checkImage(const Image& image) const {
        (void)image;
        return {};
    }
};

// Refuses prompts that contain a listed word or phrase (whole words, any case, punctuation ignored). It is a word list,
// not a judgement of what a picture will show: it is easy to get around and it will refuse some harmless prompts.
// The negative prompt is not checked (it is where people write what they do not want).
class BlocklistFilter final : public ISafetyFilter {
public:
    explicit BlocklistFilter(std::vector<std::string> terms);
    SafetyVerdict checkPrompt(const std::string& prompt) const override;
    size_t termCount() const { return terms_.size(); }

    // One term per line; blank lines and lines starting with # are ignored. Missing files are skipped.
    static std::vector<std::string> readTerms(const std::vector<std::filesystem::path>& files);

private:
    std::vector<std::vector<std::string>> terms_;   // each term as its words
    std::vector<std::string> text_;                 // and as text, for the message
};

}  // namespace gh::engine
