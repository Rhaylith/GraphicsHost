#include "engine/SafetyFilter.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace gh::engine {

namespace {

// lower case words: everything that is not a letter or digit separates them
std::vector<std::string> words(const std::string& text) {
    std::vector<std::string> out;
    std::string current;
    for (const unsigned char c : text) {
        if (std::isalnum(c) || c >= 0x80) {   // bytes of non-ASCII letters stay in the word
            current += static_cast<char>(std::tolower(c));
        } else if (!current.empty()) {
            out.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) out.push_back(std::move(current));
    return out;
}

}  // namespace

BlocklistFilter::BlocklistFilter(std::vector<std::string> terms) {
    for (const auto& t : terms) {
        auto w = words(t);
        if (w.empty()) continue;
        std::string joined;
        for (const auto& word : w) joined += (joined.empty() ? "" : " ") + word;
        text_.push_back(joined);
        terms_.push_back(std::move(w));
    }
}

SafetyVerdict BlocklistFilter::checkPrompt(const std::string& prompt) const {
    const auto w = words(prompt);
    for (size_t t = 0; t < terms_.size(); ++t) {
        const auto& term = terms_[t];
        if (term.size() > w.size()) continue;
        for (size_t start = 0; start + term.size() <= w.size(); ++start) {
            if (std::equal(term.begin(), term.end(), w.begin() + static_cast<std::ptrdiff_t>(start))) {
                return {false, "the prompt contains \"" + text_[t] + "\", which the safety filter does not allow"};
            }
        }
    }
    return {};
}

std::vector<std::string> BlocklistFilter::readTerms(const std::vector<std::filesystem::path>& files) {
    std::vector<std::string> terms;
    for (const auto& file : files) {
        std::ifstream in(file);
        std::string line;
        while (std::getline(in, line)) {
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;
            const auto last = line.find_last_not_of(" \t\r\n");
            terms.push_back(line.substr(first, last - first + 1));
        }
    }
    return terms;
}

}  // namespace gh::engine
