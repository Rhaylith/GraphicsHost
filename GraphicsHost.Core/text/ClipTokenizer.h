#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gh::text {

// The byte-pair tokeniser of the CLIP text encoders (SD 1.x, SDXL): lower-cased words are split into bytes mapped to
// printable characters, the last symbol of a word carries "</w>", and the merges of the vocabulary file are applied in
// rank order. Ids: 0..255 the single bytes, 256..511 the same with "</w>", then one id per merge, then the two special
// tokens (49406 start, 49407 end).
class ClipTokenizer {
public:
    static constexpr int kStart = 49406;
    static constexpr int kEnd = 49407;
    static constexpr int kVocabSize = 49408;

    // `merges` is the vocabulary file: a "#version" line, then one "left right" pair per line.
    explicit ClipTokenizer(const std::string& merges);
    // The merges file shipped in data/text/ next to the executable.
    static ClipTokenizer fromFile(const std::filesystem::path& file);
    static std::filesystem::path defaultFile();

    // Token ids of a text, without start and end tokens.
    std::vector<int> encode(const std::string& text) const;
    // The text of an id sequence (words joined by spaces), for debugging and tests.
    std::string decode(const std::vector<int>& ids) const;

private:
    std::vector<int> bpe(const std::string& word) const;

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> tokens_;
    std::map<std::pair<std::string, std::string>, int> ranks_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

}  // namespace gh::text
