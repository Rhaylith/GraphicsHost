#include "models/Catalog.h"

#include "core/AtomicFile.h"

#include <algorithm>
#include <cctype>

namespace gh::models {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isHex64(const std::string& s) {
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

// A single path component of a repository or file name: no traversal out of the models folder.
bool safeRelative(const std::string& p) {
    if (p.empty() || p.front() == '/' || p.find("..") != std::string::npos || p.find('\\') != std::string::npos ||
        p.find(':') != std::string::npos) {
        return false;
    }
    return true;
}

std::optional<CatalogEntry> parseEntry(const Json& j, std::string& problem) {
    if (!j.is_object()) {
        problem = "not an object";
        return std::nullopt;
    }
    CatalogEntry e;
    try {
        e.key = j.value("key", "");
        e.name = j.value("name", e.key);
        e.kind = j.value("kind", "checkpoint");
        e.family = j.value("family", "");
        e.repo = j.value("repo", "");
        e.file = j.value("file", "");
        e.revision = j.value("revision", "");
        e.sizeBytes = j.value("sizeBytes", uint64_t{0});
        e.sha256 = lower(j.value("sha256", ""));
        e.license = j.value("license", "");
        e.gated = j.value("gated", false);
        e.recommended = j.value("recommended", false);
        e.minVramGb = j.value("minVramGb", 0);
        e.notes = j.value("notes", "");
        if (j.contains("tags") && j["tags"].is_array()) e.tags = j["tags"].get<std::vector<std::string>>();
        if (j.contains("defaults") && j["defaults"].is_object()) e.defaults = j["defaults"];
        if (j.contains("companions") && j["companions"].is_array()) {
            for (const auto& c : j["companions"]) {
                CatalogCompanion cc;
                cc.role = c.value("role", "");
                cc.key = c.value("key", "");
                cc.autoDownload = c.value("auto", false);
                cc.note = c.value("note", "");
                if (cc.key.empty()) {
                    problem = "a companion has no key";
                    return std::nullopt;
                }
                e.companions.push_back(std::move(cc));
            }
        }
    } catch (const std::exception& ex) {
        problem = ex.what();
        return std::nullopt;
    }
    if (e.key.empty()) { problem = "no key"; return std::nullopt; }
    if (e.repo.find('/') == std::string::npos || !safeRelative(e.repo)) { problem = "repo must look like publisher/name"; return std::nullopt; }
    if (!safeRelative(e.file)) { problem = "file is missing or unsafe"; return std::nullopt; }
    if (e.kind != "checkpoint" && e.kind != "vae" && e.kind != "lora") { problem = "unknown kind '" + e.kind + "'"; return std::nullopt; }
    if (e.sizeBytes == 0) { problem = "sizeBytes is missing"; return std::nullopt; }
    if (!isHex64(e.sha256)) { problem = "sha256 must be 64 hex characters"; return std::nullopt; }
    return e;
}

}  // namespace

std::string CatalogEntry::url() const {
    return "https://huggingface.co/" + repo + "/resolve/" + (revision.empty() ? "main" : revision) + "/" + file;
}

fs::path CatalogEntry::relativePath() const {
    return fs::path(repo) / fs::path(file);
}

Catalog Catalog::fromJson(const Json& doc, std::vector<std::string>* warnings) {
    Catalog c;
    if (!doc.is_object() || !doc.contains("entries") || !doc["entries"].is_array()) {
        c.warnings_.push_back("catalog has no entries list");
    } else {
        for (const auto& item : doc["entries"]) {
            std::string problem;
            if (auto e = parseEntry(item, problem)) {
                c.entries_.push_back(std::move(*e));
            } else {
                c.warnings_.push_back("catalog entry '" + (item.is_object() ? item.value("key", std::string("?")) : std::string("?")) + "' skipped: " + problem);
            }
        }
    }
    if (warnings) warnings->insert(warnings->end(), c.warnings_.begin(), c.warnings_.end());
    return c;
}

Catalog Catalog::load(const std::vector<fs::path>& files) {
    Catalog merged;
    for (const auto& file : files) {
        const auto text = core::readFile(file);
        if (!text) continue;  // a catalog that is not there is normal (the user one)
        const Json doc = Json::parse(*text, nullptr, false);
        if (doc.is_discarded()) {
            merged.warnings_.push_back(file.string() + " is not valid JSON");
            continue;
        }
        Catalog part = fromJson(doc);
        for (auto& w : part.warnings_) merged.warnings_.push_back(file.filename().string() + ": " + w);
        for (auto& e : part.entries_) {
            const auto it = std::find_if(merged.entries_.begin(), merged.entries_.end(), [&](const CatalogEntry& x) { return x.key == e.key; });
            if (it != merged.entries_.end()) *it = std::move(e);
            else merged.entries_.push_back(std::move(e));
        }
    }
    return merged;
}

const CatalogEntry* Catalog::find(const std::string& key) const {
    const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const CatalogEntry& e) { return e.key == key; });
    return it == entries_.end() ? nullptr : &*it;
}

std::vector<const CatalogEntry*> Catalog::search(const CatalogFilter& f) const {
    std::vector<const CatalogEntry*> out;
    const std::string needle = lower(f.text);
    for (const auto& e : entries_) {
        if (!f.family.empty() && lower(e.family) != lower(f.family)) continue;
        if (!f.kind.empty() && e.kind != f.kind) continue;
        if (f.onlyRecommended && !e.recommended) continue;
        if (!needle.empty()) {
            std::string hay = lower(e.key + " " + e.name + " " + e.family + " " + e.notes + " " + e.repo);
            for (const auto& t : e.tags) hay += " " + lower(t);
            if (hay.find(needle) == std::string::npos) continue;
        }
        out.push_back(&e);
    }
    return out;
}

std::vector<const CatalogEntry*> Catalog::withCompanions(const CatalogEntry& entry, bool autoOnly) const {
    std::vector<const CatalogEntry*> out{&entry};
    for (const auto& c : entry.companions) {
        if (autoOnly && !c.autoDownload) continue;
        if (const CatalogEntry* companion = find(c.key)) out.push_back(companion);
    }
    return out;
}

}  // namespace gh::models
