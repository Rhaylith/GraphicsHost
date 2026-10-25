#pragma once

#include "server/TokenStore.h"

#include <functional>
#include <string>

namespace gh::server {

struct AuthDecision {
    bool allowed = false;
    int status = 0;             // 401 or 403 when not allowed
    std::string code;           // missing_api_key | invalid_api_key | insufficient_permissions
    std::string message;
    std::string principal;      // token name, or "(local)" for a trusted local client
};

// Who may call what.
//  - A request with a token is judged by that token: unknown token 401, missing permission 403.
//  - A request without a token is allowed only from this computer (loopback), and only while the setting
//    server.requireToken is off. Everyone else needs a token.
//  - Starting the server for the LAN is refused until a token exists (see ApiServer::start).
class Auth {
public:
    Auth(TokenStore& tokens, std::function<bool()> requireTokenEverywhere)
        : tokens_(tokens), requireToken_(std::move(requireTokenEverywhere)) {}

    AuthDecision decide(const std::string& secret, const std::string& remoteAddress, const std::string& permission);

    static bool isLoopback(const std::string& address);
    // "Authorization: Bearer X" or "x-api-key: X" (what Anthropic and some OpenAI clients send); empty if neither
    static std::string secretFrom(const std::string& authorizationHeader, const std::string& apiKeyHeader);

private:
    TokenStore& tokens_;
    std::function<bool()> requireToken_;
};

}  // namespace gh::server
