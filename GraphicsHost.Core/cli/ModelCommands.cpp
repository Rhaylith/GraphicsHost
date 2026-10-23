#include "cli/Cli.h"
#include "cli/Commands.h"

#include "core/SecretStore.h"
#include "models/FitHints.h"
#include "models/HuggingFace.h"
#include "models/ModelUpdates.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>

namespace gh::cli {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace gh::models;

namespace {

std::string size(uint64_t bytes) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(bytes >= (1ull << 30) ? 2 : 0) << (bytes >= (1ull << 30) ? static_cast<double>(bytes) / (1ull << 30) : static_cast<double>(bytes) / (1ull << 20))
      << (bytes >= (1ull << 30) ? " GiB" : " MiB");
    return s.str();
}

std::string hubUrl() {
    const char* hub = std::getenv("GRAPHICSHOST_HUB_URL");
    return hub && *hub ? hub : "https://huggingface.co";
}

HuggingFace makeHub() {
    return HuggingFace(core::huggingFaceToken().value_or(std::string()), hubUrl());
}

HardwareBudget budget(host::AppHost& host) {
    const auto* runtime = host.selectedRuntime();
    return budgetFrom(host.gpu(), runtime && runtime->backend == "vulkan");
}

int exitCodeFor(const HfError& e) {
    switch (e.code()) {
        case HfErrorCode::Network: return kServerUnreachable;
        case HfErrorCode::NotFound: return kNotFound;
        case HfErrorCode::Unauthorized:
        case HfErrorCode::Gated: return kNotAuthorised;
        default: return kFailure;
    }
}

int exitCodeFor(const ModelServiceError& e) {
    using C = ModelServiceError::Code;
    switch (e.code()) {
        case C::UnknownEntry:
        case C::NotFound: return kNotFound;
        case C::NeedsToken:
        case C::Gated: return kNotAuthorised;
        case C::Network: return kServerUnreachable;
        default: return kFailure;
    }
}

bool installed(host::AppHost& host, const CatalogEntry& e) {
    std::error_code ec;
    const fs::path p = host.modelsDir() / e.relativePath();
    return fs::is_regular_file(p, ec) && fs::file_size(p, ec) == e.sizeBytes;
}

std::atomic<bool> g_interrupted{false};
BOOL WINAPI onInterrupt(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
        g_interrupted = true;
        return TRUE;
    }
    return FALSE;
}

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    const auto last = s.find_last_not_of(" \t\r\n");
    return first == std::string::npos ? std::string() : s.substr(first, last - first + 1);
}

}  // namespace

int runCatalog(host::AppHost& host, const std::string& text, const std::string& family, const std::string& kind, bool json,
               std::ostream& out, std::ostream&) {
    const Catalog& catalog = host.catalog();
    const auto budgetNow = budget(host);
    CatalogFilter filter;
    filter.text = text;
    filter.family = family;
    filter.kind = kind;
    const auto found = catalog.search(filter);
    if (json) {
        Json list = Json::array();
        for (const auto* e : found) {
            const FitHint hint = fitHintFor(e->sizeBytes, e->family, e->kind, budgetNow);
            list.push_back({{"key", e->key}, {"name", e->name}, {"kind", e->kind}, {"family", e->family}, {"repo", e->repo},
                            {"file", e->file}, {"sizeBytes", e->sizeBytes}, {"sha256", e->sha256}, {"license", e->license},
                            {"gated", e->gated}, {"recommended", e->recommended}, {"notes", e->notes}, {"tags", e->tags},
                            {"installed", installed(host, *e)}, {"fit", {{"label", hint.label}, {"detail", hint.detail}, {"device", hint.device}}}});
        }
        out << Json{{"entries", list}, {"warnings", catalog.warnings()}}.dump(2) << "\n";
        return kOk;
    }
    out << std::left << std::setw(20) << "KEY" << std::setw(32) << "NAME" << std::setw(8) << "FAMILY" << std::right << std::setw(10) << "SIZE" << "  "
        << std::left << std::setw(15) << "FITS" << "STATE\n";
    for (const auto* e : found) {
        const FitHint hint = fitHintFor(e->sizeBytes, e->family, e->kind, budgetNow);
        out << std::left << std::setw(20) << e->key << std::setw(32) << e->name.substr(0, 31) << std::setw(8) << e->family << std::right << std::setw(10)
            << size(e->sizeBytes) << "  " << std::left << std::setw(15) << hint.label << (installed(host, *e) ? "installed" : "") << (e->gated ? " gated" : "") << "\n";
    }
    if (found.empty()) out << "nothing matches\n";
    else out << "\nget one with: graphicshost get <key>   (companions such as a VAE come along)\n";
    for (const auto& w : catalog.warnings()) out << "warning: " << w << "\n";
    return kOk;
}

int runSearch(host::AppHost&, const std::string& text, int limit, bool json, std::ostream& out, std::ostream& err) {
    try {
        const auto models = makeHub().search(text, limit);
        if (json) {
            Json list = Json::array();
            for (const auto& m : models) {
                list.push_back({{"id", m.id}, {"downloads", m.downloads}, {"likes", m.likes}, {"gated", m.gated}, {"license", m.license},
                                {"library", m.library}, {"lastModified", m.lastModified}});
            }
            out << Json{{"models", list}}.dump(2) << "\n";
            return kOk;
        }
        out << std::left << std::setw(52) << "REPOSITORY" << std::right << std::setw(11) << "DOWNLOADS" << std::setw(8) << "LIKES" << "  " << std::left << std::setw(8) << "ACCESS" << "LICENSE\n";
        for (const auto& m : models) {
            out << std::left << std::setw(52) << m.id.substr(0, 51) << std::right << std::setw(11) << m.downloads << std::setw(8) << m.likes << "  " << std::left
                << std::setw(8) << (m.isGated() ? "gated" : "open") << m.license << "\n";
        }
        if (models.empty()) out << "nothing found\n";
        else out << "\nsee the files: graphicshost info <repository>    download one: graphicshost get <repository> <file>\n";
        return kOk;
    } catch (const HfError& e) {
        err << "error: " << e.what() << "\n";
        return exitCodeFor(e);
    }
}

int runInfo(host::AppHost& host, const std::string& repo, bool json, std::ostream& out, std::ostream& err) {
    try {
        const HfModelInfo info = makeHub().info(repo);
        const auto files = usableFiles(info);
        const auto budgetNow = budget(host);
        Json list = Json::array();
        for (const auto& f : files) {
            const std::string kind = guessFileKind(f.path);
            const FitHint hint = fitHintFor(f.sizeBytes, f.sizeBytes > 5500ull * 1024 * 1024 ? "SDXL" : "SD 1.x", kind, budgetNow);
            list.push_back({{"path", f.path}, {"sizeBytes", f.sizeBytes}, {"sha256", f.sha256}, {"kind", kind}, {"fit", hint.label}, {"detail", hint.detail}});
        }
        if (json) {
            out << Json{{"id", info.model.id}, {"gated", info.model.gated}, {"license", info.model.license}, {"commit", info.model.commit}, {"files", list}}.dump(2) << "\n";
            return kOk;
        }
        out << info.model.id << "   license: " << (info.model.license.empty() ? "unknown" : info.model.license) << (info.model.isGated() ? "   GATED (needs a token with access)" : "") << "\n";
        if (files.empty()) out << "  no checkpoint files found (this may be a diffusers-format repository, which GraphicsHost cannot use directly)\n";
        for (const auto& f : list) {
            out << "  " << std::left << std::setw(52) << f["path"].get<std::string>() << std::right << std::setw(10) << size(f["sizeBytes"].get<uint64_t>()) << "  "
                << std::left << std::setw(11) << f["kind"].get<std::string>() << f["fit"].get<std::string>() << "\n";
        }
        return kOk;
    } catch (const HfError& e) {
        err << "error: " << e.what() << "\n";
        return exitCodeFor(e);
    }
}

int runGet(host::AppHost& host, const GetInvocation& in, bool json, bool quiet, std::ostream& out, std::ostream& err) {
    ModelService& service = host.modelService();
    DownloadPlan plan;
    try {
        if (service.catalog().find(in.target)) {
            plan = service.planCatalog(in.target, !in.noCompanions);
        } else if (in.target.find('/') != std::string::npos) {
            std::string file = in.file;
            if (file.empty()) {  // only one usable file in the repository: no need to ask
                const auto files = usableFiles(makeHub().info(in.target));
                if (files.size() == 1) {
                    file = files[0].path;
                } else {
                    err << "error: " << in.target << " has " << files.size() << " usable files; name the one you want:\n";
                    for (const auto& f : files) err << "  " << f.path << "  (" << size(f.sizeBytes) << ")\n";
                    return kUsage;
                }
            }
            plan = service.planRepoFile(in.target, file);
        } else {
            err << "error: '" << in.target << "' is neither a catalog key (graphicshost catalog) nor a repository (publisher/name)\n";
            return kNotFound;
        }
    } catch (const ModelServiceError& e) {
        err << "error: " << e.what() << "\n";
        return exitCodeFor(e);
    } catch (const HfError& e) {
        err << "error: " << e.what() << "\n";
        return exitCodeFor(e);
    }

    if (json && in.dryRun) {
        out << toJson(plan).dump(2) << "\n";
        return kOk;
    }
    if (!quiet || in.dryRun) {
        std::ostream& o = in.dryRun ? out : err;
        for (const auto& i : plan.items) {
            o << (in.dryRun ? "would get " : "getting ") << i.file << "  (" << size(i.sizeBytes) << ", " << i.role << ")" << (i.present ? "  already here" : "") << "\n";
            o << "    -> " << i.destination.string() << "\n";
        }
        for (const auto& n : plan.notes) o << "    " << n << "\n";
        if (plan.items.size() > 0 && plan.items[0].role == "checkpoint") {
            const auto hint = fitHintFor(plan.items[0].sizeBytes, plan.items[0].repo.find("xl") != std::string::npos || plan.items[0].sizeBytes > 5500ull * 1024 * 1024 ? "SDXL" : "SD 1.x", "checkpoint", budget(host));
            o << "    on this computer: " << hint.label << " (" << hint.detail << ")\n";
        }
    }
    if (in.dryRun) return kOk;

    g_interrupted = false;
    SetConsoleCtrlHandler(onInterrupt, TRUE);
    const auto downloads = service.start(plan);
    int lastLine = 0;
    bool interrupted = false;
    for (;;) {
        bool allDone = true;
        std::ostringstream line;
        for (size_t i = 0; i < downloads.size(); ++i) {
            const auto& d = downloads[i];
            if (!d->finished()) allDone = false;
            const uint64_t total = d->bytesTotal();
            line << (i ? "  |  " : "") << d->request().label.substr(0, 28) << " ";
            if (d->state() == DownloadState::Queued) line << "queued";
            else if (d->state() == DownloadState::Completed) line << "done";
            else if (d->state() == DownloadState::Failed) line << "failed";
            else if (total > 0) line << std::fixed << std::setprecision(0) << (100.0 * static_cast<double>(d->bytesDone()) / static_cast<double>(total)) << "% " << std::setprecision(1) << d->bytesPerSecond() / 1e6 << " MB/s";
            else line << size(d->bytesDone());
        }
        if (!quiet && !json && !downloads.empty()) {
            const std::string text = line.str();
            err << "\r" << text << std::string(lastLine > static_cast<int>(text.size()) ? lastLine - text.size() : 0, ' ') << std::flush;
            lastLine = static_cast<int>(text.size());
        }
        if (allDone) break;
        if (g_interrupted && !interrupted) {
            interrupted = true;
            for (const auto& d : downloads) host.downloads().pause(d->id());
        }
        if (interrupted && std::all_of(downloads.begin(), downloads.end(), [](const auto& d) { return d->finished() || d->state() == DownloadState::Paused; })) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (lastLine > 0) err << "\n";
    SetConsoleCtrlHandler(onInterrupt, FALSE);

    if (interrupted) {
        err << "paused: the partial files are kept; run the same command again to continue\n";
        return kCancelled;
    }
    int code = kOk;
    Json results = Json::array();
    for (const auto& d : downloads) {
        Json r = {{"file", d->request().destination.string()}, {"state", toString(d->state())}, {"sha256", d->sha256()}};
        if (d->state() != DownloadState::Completed) {
            err << "error: " << d->request().label << ": " << d->message() << "\n";
            r["error"] = d->message();
            code = d->error() == DownloadError::Unauthorized ? kNotAuthorised : d->error() == DownloadError::Network ? kServerUnreachable : kFailure;
        }
        results.push_back(std::move(r));
    }
    for (const auto& i : plan.items) {
        if (i.present) results.push_back({{"file", i.destination.string()}, {"state", "present"}});
    }
    if (json) {
        out << Json{{"downloads", results}, {"plan", toJson(plan)}}.dump(2) << "\n";
    } else if (code == kOk) {
        out << "ready: " << plan.items[0].destination.string() << "\n";
        out << "use it:  graphicshost gen \"your prompt\" -m " << plan.items[0].destination.stem().string() << "\n";
    }
    return code;
}

int runDownloads(host::AppHost& host, const std::string& action, bool json, std::ostream& out, std::ostream& err) {
    std::vector<std::pair<fs::path, uint64_t>> parts;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(host.modelsDir(), fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == ".part") parts.emplace_back(it->path(), it->file_size(ec));
    }
    if (action == "clear") {
        for (const auto& [p, s] : parts) fs::remove(p, ec);
        out << "removed " << parts.size() << " partial download(s)\n";
        return kOk;
    }
    if (action != "list") {
        err << "usage: graphicshost downloads [list | clear]\n";
        return kUsage;
    }
    if (json) {
        Json list = Json::array();
        for (const auto& [p, s] : parts) list.push_back({{"file", p.string()}, {"bytes", s}});
        out << Json{{"partial", list}}.dump(2) << "\n";
    } else if (parts.empty()) {
        out << "no interrupted downloads\n";
    } else {
        for (const auto& [p, s] : parts) out << "  " << size(s) << "  " << p.string() << "\n";
        out << "run the same `get` command again to continue them, or `downloads clear` to delete them\n";
    }
    return kOk;
}

int runUpdates(host::AppHost& host, bool json, std::ostream& out, std::ostream&) {
    const auto cards = host.library().scan();
    const auto updates = checkForUpdates(cards, makeHub());
    if (json) {
        Json list = Json::array();
        for (const auto& u : updates) {
            list.push_back({{"file", u.file}, {"repo", u.repo}, {"status", toString(u.status)}, {"message", u.message}, {"latestSha256", u.latestSha256}});
        }
        out << Json{{"updates", list}}.dump(2) << "\n";
        return kOk;
    }
    if (updates.empty()) out << "none of the models here came from a download with a source record\n";
    for (const auto& u : updates) {
        out << "  " << std::left << std::setw(40) << u.file << std::setw(18) << toString(u.status) << u.message << "\n";
    }
    return kOk;
}

int runLogin(host::AppHost&, const std::string& tokenArg, bool json, std::ostream& out, std::ostream& err) {
    std::string token = trim(tokenArg);
    if (token.empty()) {
        err << "Paste your Hugging Face access token (create one at https://huggingface.co/settings/tokens), then press Enter: " << std::flush;
        std::string line;
        std::getline(std::cin, line);
        token = trim(line);
    }
    if (token.empty()) {
        err << "error: no token given\n";
        return kUsage;
    }
    try {
        const std::string name = HuggingFace(token, hubUrl()).whoami();
        if (!core::storeSecret(core::kHuggingFaceSecret, token)) {
            err << "error: the token is valid but could not be stored in the Windows Credential Manager\n";
            return kFailure;
        }
        if (json) out << Json{{"user", name}}.dump() << "\n";
        else out << "logged in as " << name << " (the token is kept in the Windows Credential Manager)\n";
        return kOk;
    } catch (const HfError& e) {
        err << "error: " << e.what() << "\n";
        return exitCodeFor(e);
    }
}

int runLogout(bool json, std::ostream& out, std::ostream&) {
    const bool had = core::deleteSecret(core::kHuggingFaceSecret);
    if (json) out << Json{{"removed", had}}.dump() << "\n";
    else out << (had ? "the stored token was removed\n" : "no stored token\n");
    return kOk;
}

int runWhoami(host::AppHost&, bool json, std::ostream& out, std::ostream& err) {
    const auto token = core::huggingFaceToken();
    if (!token) {
        if (json) out << Json{{"user", nullptr}}.dump() << "\n";
        else out << "not logged in (graphicshost login)\n";
        return kOk;
    }
    try {
        const std::string name = HuggingFace(*token, hubUrl()).whoami();
        if (json) out << Json{{"user", name}}.dump() << "\n";
        else out << name << "\n";
        return kOk;
    } catch (const HfError& e) {
        err << "error: " << e.what() << "\n";
        return exitCodeFor(e);
    }
}

}  // namespace gh::cli
