#include "net/HttpClient.h"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <memory>

namespace gh::net {

namespace {

const bool g_curlReady = [] { return curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK; }();

struct State {
    HttpResponse* response = nullptr;
    uint64_t maxBody = 0;
    bool tooLarge = false;
};

size_t onBody(char* data, size_t size, size_t count, void* user) {
    auto* s = static_cast<State*>(user);
    const size_t n = size * count;
    if (s->response->body.size() + n > s->maxBody) {
        s->tooLarge = true;
        return 0;
    }
    s->response->body.append(data, n);
    return n;
}

size_t onHeader(char* data, size_t size, size_t count, void* user) {
    auto* s = static_cast<State*>(user);
    const size_t n = size * count;
    const std::string line(data, n);
    const auto colon = line.find(':');
    if (colon != std::string::npos) {
        std::string name = line.substr(0, colon);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::string value = line.substr(colon + 1);
        const auto first = value.find_first_not_of(" \t");
        const auto last = value.find_last_not_of(" \t\r\n");
        value = first == std::string::npos ? std::string() : value.substr(first, last - first + 1);
        s->response->headers[name] = value;  // the last response in a redirect chain wins
    } else if (line.rfind("HTTP/", 0) == 0) {
        s->response->headers.clear();  // a new response (after a redirect): forget the previous headers
    }
    return n;
}

HttpResponse perform(const std::string& url, const HttpOptions& o, bool headOnly) {
    HttpResponse response;
    CURL* h = curl_easy_init();
    if (!h) {
        response.error = "cannot initialise libcurl";
        return response;
    }
    State state{&response, o.maxBodyBytes, false};
    curl_slist* headers = nullptr;
    if (!o.bearerToken.empty()) headers = curl_slist_append(headers, ("Authorization: Bearer " + o.bearerToken).c_str());
    for (const auto& line : o.headers) headers = curl_slist_append(headers, line.c_str());
    headers = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_USERAGENT, "GraphicsHost/0.1");
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, o.followRedirects ? 1L : 0L);
    curl_easy_setopt(h, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, o.timeoutSeconds);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");  // let curl decompress
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, onBody);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &state);
    curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, onHeader);
    curl_easy_setopt(h, CURLOPT_HEADERDATA, &state);
    if (headOnly) curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
    const CURLcode rc = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    response.status = static_cast<int>(status);
    if (state.tooLarge) {
        response.error = "the answer is larger than " + std::to_string(o.maxBodyBytes) + " bytes";
    } else if (rc != CURLE_OK) {
        response.error = curl_easy_strerror(rc);
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(h);
    return response;
}

}  // namespace

HttpResponse httpGet(const std::string& url, const HttpOptions& options) {
    return perform(url, options, false);
}

HttpResponse httpHead(const std::string& url, const HttpOptions& options) {
    return perform(url, options, true);
}

std::string urlEncode(const std::string& text) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : text) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

}  // namespace gh::net
