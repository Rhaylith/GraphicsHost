#include "text/ClipTokenizer.h"

#include "core/AtomicFile.h"
#include "core/Paths.h"

#include <windows.h>

#include <algorithm>
#include <climits>
#include <cwctype>
#include <sstream>
#include <stdexcept>

namespace gh::text {

namespace {

std::string utf8(uint32_t cp) {
    std::string out;
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

std::vector<uint32_t> codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = c;
        size_t n = 1;
        if (c >= 0xF0 && i + 3 < s.size() + 0) { cp = c & 0x07; n = 4; }
        else if (c >= 0xE0) { cp = c & 0x0F; n = 3; }
        else if (c >= 0xC0) { cp = c & 0x1F; n = 2; }
        if (i + n > s.size()) { out.push_back(c); ++i; continue; }
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

bool isLetter(uint32_t cp) {
    if (cp < 0x80) return std::isalpha(static_cast<int>(cp)) != 0;
    if (cp > 0xFFFF) return false;
    WORD type = 0;
    const wchar_t w = static_cast<wchar_t>(cp);
    return GetStringTypeW(CT_CTYPE1, &w, 1, &type) && (type & C1_ALPHA) && !(type & C1_DIGIT);
}
bool isDigit(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    if (cp > 0xFFFF) return false;
    WORD type = 0;
    const wchar_t w = static_cast<wchar_t>(cp);
    return GetStringTypeW(CT_CTYPE1, &w, 1, &type) && (type & C1_DIGIT);
}
bool isSpace(uint32_t cp) {
    if (cp < 0x80) return std::isspace(static_cast<int>(cp)) != 0;
    return cp == 0xA0 || cp == 0x2028 || cp == 0x2029 || cp == 0x3000 || (cp >= 0x2000 && cp <= 0x200A);
}
uint32_t lower(uint32_t cp) {
    if (cp < 0x80) return static_cast<uint32_t>(std::tolower(static_cast<int>(cp)));
    if (cp > 0xFFFF) return cp;
    return static_cast<uint32_t>(std::towlower(static_cast<wint_t>(cp)));
}

// The printable character each byte stands for in the vocabulary (bytes 33..126, 161..172 and 174..255 are themselves).
const std::vector<std::pair<int, uint32_t>>& byteMap() {
    static const std::vector<std::pair<int, uint32_t>> map = [] {
        std::vector<int> printable;
        for (int b = 33; b <= 126; ++b) printable.push_back(b);
        for (int b = 161; b <= 172; ++b) printable.push_back(b);
        for (int b = 174; b <= 255; ++b) printable.push_back(b);
        std::vector<std::pair<int, uint32_t>> m;
        for (int b : printable) m.push_back({b, static_cast<uint32_t>(b)});
        uint32_t extra = 0;
        for (int b = 0; b < 256; ++b)
            if (std::find(printable.begin(), printable.end(), b) == printable.end()) m.push_back({b, 256 + extra++});
        return m;
    }();
    return map;
}

}  // namespace

ClipTokenizer::ClipTokenizer(const std::string& merges) {
    for (const auto& [byte, cp] : byteMap()) {
        (void)byte;
        tokens_.push_back(utf8(cp));
    }
    const size_t base = tokens_.size();
    for (size_t i = 0; i < base; ++i) tokens_.push_back(tokens_[i] + "</w>");
    std::istringstream in(merges);
    std::string line;
    std::getline(in, line);   // "#version: ..."
    int rank = 0;
    while (std::getline(in, line) && rank < kVocabSize - 2 - static_cast<int>(tokens_.size())) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t space = line.find(' ');
        if (space == std::string::npos) continue;
        const std::string a = line.substr(0, space), b = line.substr(space + 1);
        ranks_[{a, b}] = rank++;
        tokens_.push_back(a + b);
    }
    if (static_cast<int>(tokens_.size()) != kVocabSize - 2) throw std::runtime_error("the CLIP merges file is incomplete (" + std::to_string(tokens_.size()) + " entries)");
    tokens_.push_back("<|startoftext|>");
    tokens_.push_back("<|endoftext|>");
    for (size_t i = 0; i < tokens_.size(); ++i) vocab_.emplace(tokens_[i], static_cast<int>(i));
}

std::filesystem::path ClipTokenizer::defaultFile() {
    return core::AppPaths::executableDir() / "data" / "text" / "clip_merges.txt";
}

ClipTokenizer ClipTokenizer::fromFile(const std::filesystem::path& file) {
    const auto text = core::readFile(file);
    if (!text) throw std::runtime_error("cannot read the CLIP vocabulary " + file.string());
    return ClipTokenizer(*text);
}

std::vector<int> ClipTokenizer::bpe(const std::string& word) const {
    if (const auto hit = cache_.find(word); hit != cache_.end()) return hit->second;
    std::vector<std::string> symbols;
    std::map<int, uint32_t> toChar(byteMap().begin(), byteMap().end());
    for (size_t i = 0; i < word.size(); ++i) symbols.push_back(utf8(toChar.at(static_cast<unsigned char>(word[i]))));
    if (!symbols.empty()) symbols.back() += "</w>";
    while (symbols.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = ranks_.find({symbols[i], symbols[i + 1]});
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == INT_MAX) break;
        symbols[at] += symbols[at + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(at) + 1);
    }
    std::vector<int> ids;
    for (const auto& s : symbols) {
        const auto it = vocab_.find(s);
        if (it != vocab_.end()) ids.push_back(it->second);
    }
    cache_[word] = ids;
    return ids;
}

std::vector<int> ClipTokenizer::encode(const std::string& input) const {
    // whitespace cleaned (runs become one space, ends trimmed) and lower-cased
    std::vector<uint32_t> cps;
    bool pendingSpace = false;
    for (const uint32_t cp : codepoints(input)) {
        if (isSpace(cp)) {
            pendingSpace = !cps.empty();
            continue;
        }
        if (pendingSpace) cps.push_back(' ');
        pendingSpace = false;
        cps.push_back(lower(cp));
    }

    std::vector<int> ids;
    const auto matches = [&](size_t at, const char* literal) {
        const std::string l = literal;
        for (size_t k = 0; k < l.size(); ++k)
            if (at + k >= cps.size() || cps[at + k] != static_cast<unsigned char>(l[k])) return false;
        return true;
    };
    size_t i = 0;
    while (i < cps.size()) {
        if (cps[i] == ' ') {
            ++i;
            continue;
        }
        size_t n = 0;
        if (matches(i, "<|startoftext|>")) {
            ids.push_back(kStart);
            i += 15;
            continue;
        }
        if (matches(i, "<|endoftext|>")) {
            ids.push_back(kEnd);
            i += 13;
            continue;
        }
        for (const char* c : {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"})
            if (matches(i, c)) {
                n = std::char_traits<char>::length(c);
                break;
            }
        if (n == 0) {
            if (isLetter(cps[i])) {
                while (i + n < cps.size() && isLetter(cps[i + n])) ++n;
            } else if (isDigit(cps[i])) {
                n = 1;
            } else {
                while (i + n < cps.size() && cps[i + n] != ' ' && !isLetter(cps[i + n]) && !isDigit(cps[i + n])) {
                    // a contraction starts a new piece
                    if (n > 0 && cps[i + n] == '\'' && (matches(i + n, "'s") || matches(i + n, "'t") || matches(i + n, "'re") || matches(i + n, "'ve") || matches(i + n, "'m") || matches(i + n, "'ll") || matches(i + n, "'d"))) break;
                    ++n;
                }
                if (n == 0) n = 1;
            }
        }
        std::string word;
        for (size_t k = 0; k < n; ++k) word += utf8(cps[i + k]);
        for (const int id : bpe(word)) ids.push_back(id);
        i += n;
    }
    return ids;
}

std::string ClipTokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (const int id : ids) {
        if (id < 0 || id >= static_cast<int>(tokens_.size())) continue;
        std::string t = tokens_[static_cast<size_t>(id)];
        const bool endsWord = t.size() >= 4 && t.compare(t.size() - 4, 4, "</w>") == 0;
        if (endsWord) t.erase(t.size() - 4);
        // map the printable stand-ins back to bytes
        std::string bytes;
        for (const uint32_t cp : codepoints(t)) {
            for (const auto& [b, c] : byteMap())
                if (c == cp) {
                    bytes += static_cast<char>(b);
                    break;
                }
        }
        out += bytes;
        if (endsWord) out += ' ';
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

}  // namespace gh::text
