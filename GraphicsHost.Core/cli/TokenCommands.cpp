#include "cli/Cli.h"
#include "cli/Commands.h"

#include <iomanip>
#include <ostream>

namespace gh::cli {

using Json = nlohmann::json;

namespace {

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> parts;
    std::string current;
    for (const char c : text) {
        if (c == ',' || c == ' ') {
            if (!current.empty()) parts.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) parts.push_back(current);
    return parts;
}

}  // namespace

int runToken(host::AppHost& host, const std::string& action, const std::vector<std::string>& args,
             const std::string& permissions, bool json, std::ostream& out, std::ostream& err) {
    auto& store = host.tokens();
    try {
        if (action == "create" && args.size() == 1) {
            const auto created = store.create(args[0], split(permissions.empty() ? "read,generate" : permissions));
            if (json) {
                out << Json{{"id", created.info.id}, {"name", created.info.name}, {"permissions", created.info.permissions}, {"secret", created.secret}}.dump(2) << "\n";
            } else {
                out << "token '" << created.info.name << "' (" << created.info.id << ") with permissions: ";
                for (size_t i = 0; i < created.info.permissions.size(); ++i) out << (i ? ", " : "") << created.info.permissions[i];
                out << "\n\n    " << created.secret << "\n\nCopy it now: this is the only time it is shown. Send it as  Authorization: Bearer <token>\n";
            }
            return kOk;
        }
        if (action == "list" && args.empty()) {
            const auto tokens = store.list();
            if (json) {
                Json list = Json::array();
                for (const auto& t : tokens) list.push_back({{"id", t.id}, {"name", t.name}, {"permissions", t.permissions}, {"created", t.created}, {"lastUsed", t.lastUsed}});
                out << Json{{"tokens", list}}.dump(2) << "\n";
            } else if (tokens.empty()) {
                out << "no tokens (the server accepts requests from this computer without one)\n";
            } else {
                for (const auto& t : tokens) {
                    out << std::left << std::setw(14) << t.id << std::setw(24) << t.name;
                    for (size_t i = 0; i < t.permissions.size(); ++i) out << (i ? "," : "") << t.permissions[i];
                    out << "   last used: " << (t.lastUsed.empty() ? "never" : t.lastUsed) << "\n";
                }
            }
            return kOk;
        }
        if (action == "revoke" && args.size() == 1) {
            if (!store.revoke(args[0])) {
                err << "error: no token with the id or name '" << args[0] << "'\n";
                return kNotFound;
            }
            out << "revoked " << args[0] << "\n";
            return kOk;
        }
    } catch (const std::invalid_argument& e) {
        err << "error: " << e.what() << "\n";
        return kUsage;
    } catch (const std::exception& e) {
        err << "error: " << e.what() << "\n";
        return kFailure;
    }
    err << "usage: graphicshost token create NAME [--perm read,generate,manage,admin] | list | revoke ID_OR_NAME\n";
    return kUsage;
}

}  // namespace gh::cli
