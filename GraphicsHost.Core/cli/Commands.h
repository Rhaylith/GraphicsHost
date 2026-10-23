#pragma once

#include "host/AppHost.h"

#include <nlohmann/json.hpp>

#include <iosfwd>
#include <string>
#include <vector>

namespace gh::cli {

// Turns text from the command line into a JSON value of the same type as sample (bool, integer, float or string).
bool coerceJsonValue(const nlohmann::json& sample, const std::string& text, nlohmann::json& out, std::string& problem);

struct GenInvocation {
    std::string prompt;
    std::string model;                 // path, or a name looked up in the models folder
    std::string preset;                // optional
    std::string outPath;               // optional: a file (single image, has an extension) or a folder
    nlohmann::json request = nlohmann::json::object();  // only the flags the user gave
    nlohmann::json load = nlohmann::json::object();     // only the flags the user gave
    int quality = 92;                  // JPEG / lossy WebP
    bool dryRun = false;               // show what would happen, do not load or generate
    bool noGallery = false;            // do not add to the gallery
    bool noFitCheck = false;           // skip the memory estimate (and its automatic adjustments)
};

// Each returns an exit code (cli::ExitCode). Normal output goes to out, diagnostics and progress to err.
int runGen(host::AppHost& host, const GenInvocation& invocation, bool json, bool quiet, std::ostream& out,
           std::ostream& err);
int runLs(host::AppHost& host, bool json, std::ostream& out, std::ostream& err);
// preset list | show NAME | set NAME key value... | delete NAME | import FILE [NAME] | export NAME FILE
int runPreset(host::AppHost& host, const std::string& action, const std::vector<std::string>& args, bool json,
              std::ostream& out, std::ostream& err);
// defaults list | show MODEL | set MODEL key value... | clear MODEL   (MODEL is a checkpoint file name)
int runDefaults(host::AppHost& host, const std::string& action, const std::vector<std::string>& args, bool json,
                std::ostream& out, std::ostream& err);
// gallery list [text] | show ID | rm ID | rescan
int runGallery(host::AppHost& host, const std::string& action, const std::vector<std::string>& args, int limit,
               bool json, std::ostream& out, std::ostream& err);


struct GetInvocation {
    std::string target;        // a catalog key, or a Hugging Face repository "publisher/name"
    std::string file;          // for a repository: the file to download (optional when only one fits)
    bool noCompanions = false; // do not fetch the VAE that belongs to a catalog model
    bool dryRun = false;       // show the plan, download nothing
};

// Model discovery and downloads.
int runCatalog(host::AppHost& host, const std::string& text, const std::string& family, const std::string& kind, bool json,
               std::ostream& out, std::ostream& err);
int runSearch(host::AppHost& host, const std::string& text, int limit, bool json, std::ostream& out, std::ostream& err);
int runInfo(host::AppHost& host, const std::string& repo, bool json, std::ostream& out, std::ostream& err);
int runGet(host::AppHost& host, const GetInvocation& invocation, bool json, bool quiet, std::ostream& out, std::ostream& err);
// downloads list | clear : the partial files that interrupted downloads left behind (get continues them)
int runDownloads(host::AppHost& host, const std::string& action, bool json, std::ostream& out, std::ostream& err);
int runUpdates(host::AppHost& host, bool json, std::ostream& out, std::ostream& err);
// login [token] (the token is read from stdin when not given), logout, whoami
int runLogin(host::AppHost& host, const std::string& token, bool json, std::ostream& out, std::ostream& err);
int runLogout(bool json, std::ostream& out, std::ostream& err);
int runWhoami(host::AppHost& host, bool json, std::ostream& out, std::ostream& err);


// token create NAME [--perm ...] | list | revoke ID_OR_NAME  (API access tokens)
int runToken(host::AppHost& host, const std::string& action, const std::vector<std::string>& args,
             const std::string& permissions, bool json, std::ostream& out, std::ostream& err);

}  // namespace gh::cli
