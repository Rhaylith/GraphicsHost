#pragma once

#include <optional>
#include <string>

namespace gh::core {

// Secrets (the Hugging Face token) are kept in the Windows Credential Manager, per Windows user, never in
// settings.json. Names are stored as "GraphicsHost/<name>".
bool storeSecret(const std::string& name, const std::string& value);
std::optional<std::string> readSecret(const std::string& name);
bool deleteSecret(const std::string& name);

// The Hugging Face token to use: the HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable, otherwise the stored one.
std::optional<std::string> huggingFaceToken();
inline constexpr const char* kHuggingFaceSecret = "huggingface-token";

}  // namespace gh::core
