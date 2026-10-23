#include "core/SecretStore.h"

#include <windows.h>
#include <wincred.h>

#include <cstdlib>
#include <vector>

#pragma comment(lib, "advapi32.lib")

namespace gh::core {

namespace {

std::wstring targetName(const std::string& name) {
    const std::string full = "GraphicsHost/" + name;
    return std::wstring(full.begin(), full.end());  // names are ASCII
}

std::optional<std::string> environment(const char* name) {
    char* value = nullptr;
    size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return std::nullopt;
    std::string result(value);
    std::free(value);
    if (result.empty()) return std::nullopt;
    return result;
}

}  // namespace

bool storeSecret(const std::string& name, const std::string& value) {
    const std::wstring target = targetName(name);
    CREDENTIALW credential{};
    credential.Type = CRED_TYPE_GENERIC;
    credential.TargetName = const_cast<LPWSTR>(target.c_str());
    credential.CredentialBlobSize = static_cast<DWORD>(value.size());
    credential.CredentialBlob = reinterpret_cast<LPBYTE>(const_cast<char*>(value.data()));
    credential.Persist = CRED_PERSIST_LOCAL_MACHINE;
    return CredWriteW(&credential, 0) != FALSE;
}

std::optional<std::string> readSecret(const std::string& name) {
    const std::wstring target = targetName(name);
    PCREDENTIALW credential = nullptr;
    if (!CredReadW(target.c_str(), CRED_TYPE_GENERIC, 0, &credential)) {
        return std::nullopt;
    }
    std::string value(reinterpret_cast<const char*>(credential->CredentialBlob), credential->CredentialBlobSize);
    CredFree(credential);
    return value;
}

bool deleteSecret(const std::string& name) {
    return CredDeleteW(targetName(name).c_str(), CRED_TYPE_GENERIC, 0) != FALSE;
}

std::optional<std::string> huggingFaceToken() {
    if (auto t = environment("HF_TOKEN")) return t;
    if (auto t = environment("HUGGING_FACE_HUB_TOKEN")) return t;
    return readSecret(kHuggingFaceSecret);
}

}  // namespace gh::core
