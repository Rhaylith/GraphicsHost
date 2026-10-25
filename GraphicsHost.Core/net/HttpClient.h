#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gh::net {

struct HttpOptions {
    std::string bearerToken;                 // Authorization: Bearer ... (dropped by curl on redirects to other hosts)
    std::vector<std::string> headers;        // extra "Name: value" lines
    long timeoutSeconds = 30;
    uint64_t maxBodyBytes = 32ull * 1024 * 1024;
    bool followRedirects = true;
};

struct HttpResponse {
    int status = 0;                          // 0 when the request did not get an answer
    std::string body;
    std::map<std::string, std::string> headers;  // lower-case names
    std::string error;                       // transport problem (DNS, TLS, timeout, body too large), empty otherwise
    bool ok() const { return error.empty() && status >= 200 && status < 300; }
};

// Blocking GET (and HEAD) with libcurl. Never throws.
HttpResponse httpGet(const std::string& url, const HttpOptions& options = {});
HttpResponse httpHead(const std::string& url, const HttpOptions& options = {});

// Percent-encodes a query value.
std::string urlEncode(const std::string& text);

}  // namespace gh::net
