#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gh::server {

inline std::string base64Encode(const std::vector<uint8_t>& bytes) {
    static const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        const uint32_t v = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
        out += table[v >> 18];
        out += table[(v >> 12) & 63];
        out += table[(v >> 6) & 63];
        out += table[v & 63];
    }
    if (i + 1 == bytes.size()) {
        const uint32_t v = bytes[i] << 16;
        out += table[v >> 18];
        out += table[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == bytes.size()) {
        const uint32_t v = (bytes[i] << 16) | (bytes[i + 1] << 8);
        out += table[v >> 18];
        out += table[(v >> 12) & 63];
        out += table[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

// Accepts padded or unpadded base64, and ignores white space. Returns false on any other character.
inline bool base64Decode(const std::string& text, std::vector<uint8_t>& out) {
    out.clear();
    out.reserve(text.size() / 4 * 3);
    uint32_t buffer = 0;
    int bits = 0;
    for (const char c : text) {
        int v;
        if (c >= 'A' && c <= 'Z') v = c - 'A';
        else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
        else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+' || c == '-') v = 62;   // also the URL-safe alphabet
        else if (c == '/' || c == '_') v = 63;
        else if (c == '=' || c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        else return false;
        buffer = (buffer << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buffer >> bits) & 0xff));
        }
    }
    return true;
}

}  // namespace gh::server
