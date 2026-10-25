#include "server/Auth.h"

#include <algorithm>

namespace gh::server {

bool Auth::isLoopback(const std::string& a) {
    return a == "::1" || a == "localhost" || a.rfind("127.", 0) == 0 || a == "::ffff:127.0.0.1" || a.rfind("::ffff:127.", 0) == 0;
}

std::string Auth::secretFrom(const std::string& authorization, const std::string& apiKey) {
    auto trim = [](std::string s) {
        const auto first = s.find_first_not_of(" \t");
        const auto last = s.find_last_not_of(" \t");
        return first == std::string::npos ? std::string() : s.substr(first, last - first + 1);
    };
    std::string lower = authorization;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower.rfind("bearer ", 0) == 0) return trim(authorization.substr(7));
    if (!apiKey.empty()) return trim(apiKey);
    return {};
}

AuthDecision Auth::decide(const std::string& secret, const std::string& remote, const std::string& permission) {
    AuthDecision d;
    if (!secret.empty()) {
        const auto token = tokens_.verify(secret);
        if (!token) {
            d.status = 401;
            d.code = "invalid_api_key";
            d.message = "the access token is not valid (it may have been revoked)";
            return d;
        }
        if (!token->allows(permission)) {
            d.status = 403;
            d.code = "insufficient_permissions";
            d.message = "this token ('" + token->name + "') does not have the '" + permission + "' permission";
            return d;
        }
        d.allowed = true;
        d.principal = token->name;
        return d;
    }
    const bool mustHaveToken = requireToken_ && requireToken_();
    if (isLoopback(remote) && !mustHaveToken) {
        d.allowed = true;
        d.principal = "(local)";
        return d;
    }
    d.status = 401;
    d.code = "missing_api_key";
    d.message = "an access token is required: send 'Authorization: Bearer <token>' (create one with: graphicshost token create <name>)";
    return d;
}

}  // namespace gh::server
