#pragma once

#include "text/ClipTokenizer.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gh::text {

// The prompt syntax (workflow plan, section 6.2), applied in this order:
//   1. comments are removed: "// to the end of the line" and "/* ... */"
//   2. choices {a|b|c} are resolved with the seed (nesting allowed); a pair of braces without "|" stays as it is
//   3. weights: (text:1.3) multiplies by 1.3, (text) by 1.1, [text] by 1/1.1; groups nest and their weights multiply
//   4. embedding:name and embedding:name:0.8 inject a textual-inversion vector
//   5. the word BREAK ends the current 75-token chunk
// A backslash makes the next character plain text: \( \) \[ \] \{ \} \| \\ \:

struct PromptPiece {
    enum class Kind { Text, Embedding, Break };
    Kind kind = Kind::Text;
    std::string text;    // Text: the words; Embedding: the embedding's name
    float weight = 1.0f;
};

struct ParsedPrompt {
    std::vector<PromptPiece> pieces;
    std::vector<std::string> embeddings;   // names used, in order of first use
};

std::string removeComments(const std::string& prompt);
std::string resolveChoices(const std::string& prompt, uint64_t seed);
ParsedPrompt parsePrompt(const std::string& prompt, uint64_t seed);

// ---- chunks for a CLIP encoder -------------------------------------------------------------------------------------------
constexpr int kChunkTokens = 75;
constexpr int kEmbeddingBase = 1'000'000;   // ids from here on stand for the vectors of an embedding (kEmbeddingBase + n)

struct EmbeddingUse {
    std::string name;
    float weight = 1.0f;
    int chunk = 0;
    int position = 0;    // index of its first vector in the chunk's ids (the start token is 0)
    int vectors = 1;
    int firstId = 0;     // the placeholder ids are firstId .. firstId + vectors - 1
};

struct Chunk {
    std::vector<int> ids;          // 77 entries: start, up to 75 tokens, end, padding
    std::vector<float> weights;    // one per id (1 for start, end and padding)
};

struct ChunkedPrompt {
    std::vector<Chunk> chunks;
    std::vector<EmbeddingUse> embeddings;
};

// How many vectors the named embedding holds (0: unknown, the name is then written as ordinary text).
using EmbeddingLookup = std::function<int(const std::string&)>;

// `padId` fills the unused positions: the end token for SD 1.x and the first encoder of SDXL, 0 for the second.
ChunkedPrompt buildChunks(const ClipTokenizer& tokenizer, const ParsedPrompt& prompt, int padId = ClipTokenizer::kEnd, const EmbeddingLookup& lookup = {});

}  // namespace gh::text
