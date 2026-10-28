#include "text/PromptParser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <random>

namespace gh::text {

namespace {

constexpr float kEmphasis = 1.1f;

bool isNameChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.' || c == '/' || c == '\\'; }

// index of the bracket closing the one at `open`, honouring backslash escapes and nesting of the same kind; npos if none
size_t closing(const std::string& s, size_t open, char opener, char closer) {
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        if (s[i] == '\\') {
            ++i;
            continue;
        }
        if (s[i] == opener) ++depth;
        else if (s[i] == closer && --depth == 0) return i;
    }
    return std::string::npos;
}

bool parseNumber(const std::string& s, float& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const float v = std::strtof(s.c_str(), &end);
    if (end != s.c_str() + s.size()) return false;
    out = v;
    return true;
}

class Weighter {
public:
    explicit Weighter(ParsedPrompt& out) : out_(out) {}

    void run(const std::string& s, float weight) {
        std::string plain;
        const auto flush = [&] {
            if (!plain.empty()) addText(plain, weight);
            plain.clear();
        };
        for (size_t i = 0; i < s.size();) {
            const char c = s[i];
            if (c == '\\' && i + 1 < s.size()) {
                plain += s[i + 1];
                i += 2;
                continue;
            }
            if (c == '(' || c == '[') {
                const char closer = c == '(' ? ')' : ']';
                const size_t end = closing(s, i, c, closer);
                if (end != std::string::npos) {
                    flush();
                    std::string inner = s.substr(i + 1, end - i - 1);
                    float w = c == '(' ? kEmphasis : 1.0f / kEmphasis;
                    // an explicit weight: the last unescaped ":number" of a round group
                    if (c == '(') {
                        const size_t colon = inner.rfind(':');
                        float value = 0;
                        if (colon != std::string::npos && (colon == 0 || inner[colon - 1] != '\\') && parseNumber(inner.substr(colon + 1), value)) {
                            w = value;
                            inner.erase(colon);
                        }
                    }
                    run(inner, weight * w);
                    i = end + 1;
                    continue;
                }
            }
            if (c == 'e' && s.compare(i, 10, "embedding:") == 0 && (i == 0 || !isNameChar(s[i - 1]))) {
                size_t j = i + 10;
                while (j < s.size() && isNameChar(s[j])) ++j;
                if (j > i + 10) {
                    flush();
                    PromptPiece piece;
                    piece.kind = PromptPiece::Kind::Embedding;
                    piece.text = s.substr(i + 10, j - i - 10);
                    piece.weight = weight;
                    // "embedding:name:0.8"
                    if (j < s.size() && s[j] == ':') {
                        size_t k = j + 1;
                        while (k < s.size() && (std::isdigit(static_cast<unsigned char>(s[k])) || s[k] == '.' || s[k] == '-')) ++k;
                        float value = 0;
                        if (parseNumber(s.substr(j + 1, k - j - 1), value)) {
                            piece.weight *= value;
                            j = k;
                        }
                    }
                    out_.pieces.push_back(piece);
                    if (std::find(out_.embeddings.begin(), out_.embeddings.end(), piece.text) == out_.embeddings.end()) out_.embeddings.push_back(piece.text);
                    i = j;
                    continue;
                }
            }
            if (c == 'B' && s.compare(i, 5, "BREAK") == 0 && (i == 0 || std::isspace(static_cast<unsigned char>(s[i - 1]))) &&
                (i + 5 == s.size() || std::isspace(static_cast<unsigned char>(s[i + 5])))) {
                flush();
                PromptPiece piece;
                piece.kind = PromptPiece::Kind::Break;
                out_.pieces.push_back(piece);
                i += 5;
                continue;
            }
            plain += c;
            ++i;
        }
        flush();
    }

private:
    void addText(const std::string& text, float weight) {
        if (!out_.pieces.empty() && out_.pieces.back().kind == PromptPiece::Kind::Text && out_.pieces.back().weight == weight) {
            out_.pieces.back().text += text;
            return;
        }
        PromptPiece piece;
        piece.text = text;
        piece.weight = weight;
        out_.pieces.push_back(piece);
    }
    ParsedPrompt& out_;
};

// ---- choices -----------------------------------------------------------------------------------------------------------

std::string choose(const std::string& s, std::mt19937_64& rng) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '\\' && i + 1 < s.size()) {   // the escape stays for the weight pass, which removes it
            out += s[i];
            out += s[i + 1];
            i += 2;
            continue;
        }
        if (s[i] == '{') {
            const size_t end = closing(s, i, '{', '}');
            if (end != std::string::npos) {
                const std::string inner = s.substr(i + 1, end - i - 1);
                // options split at the top-level bars
                std::vector<std::string> options;
                std::string current;
                int depth = 0;
                for (size_t k = 0; k < inner.size(); ++k) {
                    if (inner[k] == '\\' && k + 1 < inner.size()) {
                        current += inner[k];
                        current += inner[++k];
                        continue;
                    }
                    if (inner[k] == '{') ++depth;
                    if (inner[k] == '}') --depth;
                    if (inner[k] == '|' && depth == 0) {
                        options.push_back(current);
                        current.clear();
                    } else {
                        current += inner[k];
                    }
                }
                options.push_back(current);
                if (options.size() > 1) {
                    const size_t pick = static_cast<size_t>(rng() % options.size());
                    out += choose(options[pick], rng);
                    i = end + 1;
                    continue;
                }
                out += '{' + choose(inner, rng) + '}';   // no bar: literal braces, content still examined
                i = end + 1;
                continue;
            }
        }
        out += s[i++];
    }
    return out;
}

}  // namespace

std::string removeComments(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            out += s[i];
            out += s[i + 1];
            i += 2;
        } else if (s.compare(i, 2, "//") == 0) {
            while (i < s.size() && s[i] != '\n') ++i;
        } else if (s.compare(i, 2, "/*") == 0) {
            const size_t end = s.find("*/", i + 2);
            i = end == std::string::npos ? s.size() : end + 2;
            out += ' ';
        } else {
            out += s[i++];
        }
    }
    return out;
}

std::string resolveChoices(const std::string& prompt, uint64_t seed) {
    std::mt19937_64 rng(seed);
    return choose(prompt, rng);
}

ParsedPrompt parsePrompt(const std::string& prompt, uint64_t seed) {
    ParsedPrompt parsed;
    Weighter(parsed).run(resolveChoices(removeComments(prompt), seed), 1.0f);
    return parsed;
}

std::string plainText(const ParsedPrompt& prompt) {
    std::string out;
    for (const PromptPiece& piece : prompt.pieces) {
        switch (piece.kind) {
            case PromptPiece::Kind::Text: out += piece.text; break;
            case PromptPiece::Kind::Embedding: out += "embedding:" + piece.text; break;
            case PromptPiece::Kind::Break: out += "\n"; break;
        }
    }
    return out;
}

ChunkedPrompt buildChunks(const ClipTokenizer& tokenizer, const ParsedPrompt& prompt, int padId, const EmbeddingLookup& lookup) {
    ChunkedPrompt result;
    std::vector<int> ids;
    std::vector<float> weights;
    const auto flush = [&] {
        Chunk chunk;
        chunk.ids.push_back(ClipTokenizer::kStart);
        chunk.weights.push_back(1.0f);
        chunk.ids.insert(chunk.ids.end(), ids.begin(), ids.end());
        chunk.weights.insert(chunk.weights.end(), weights.begin(), weights.end());
        chunk.ids.push_back(ClipTokenizer::kEnd);
        chunk.weights.push_back(1.0f);
        while (static_cast<int>(chunk.ids.size()) < kChunkTokens + 2) {
            chunk.ids.push_back(padId);
            chunk.weights.push_back(1.0f);
        }
        result.chunks.push_back(std::move(chunk));
        ids.clear();
        weights.clear();
    };
    int nextEmbeddingId = kEmbeddingBase;
    const auto addTokens = [&](const std::vector<int>& tokens, float weight) {
        for (const int t : tokens) {
            if (static_cast<int>(ids.size()) == kChunkTokens) flush();
            ids.push_back(t);
            weights.push_back(weight);
        }
    };
    for (const PromptPiece& piece : prompt.pieces) {
        switch (piece.kind) {
            case PromptPiece::Kind::Break:
                if (!ids.empty() || result.chunks.empty()) flush();
                break;
            case PromptPiece::Kind::Text: addTokens(tokenizer.encode(piece.text), piece.weight); break;
            case PromptPiece::Kind::Embedding: {
                const int vectors = lookup ? lookup(piece.text) : 0;
                if (vectors <= 0) {   // unknown embedding: written as the words it was typed as
                    addTokens(tokenizer.encode("embedding:" + piece.text), piece.weight);
                    break;
                }
                if (static_cast<int>(ids.size()) + vectors > kChunkTokens) flush();
                EmbeddingUse use;
                use.name = piece.text;
                use.weight = piece.weight;
                use.chunk = static_cast<int>(result.chunks.size());
                use.position = 1 + static_cast<int>(ids.size());
                use.vectors = vectors;
                use.firstId = nextEmbeddingId;
                for (int v = 0; v < vectors; ++v) {
                    ids.push_back(nextEmbeddingId++);
                    weights.push_back(piece.weight);
                }
                result.embeddings.push_back(use);
                break;
            }
        }
    }
    if (!ids.empty() || result.chunks.empty()) flush();
    return result;
}

}  // namespace gh::text
