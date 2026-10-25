#include "core/SecretStore.h"
#include "models/HuggingFace.h"

#include <cstdlib>

namespace gh::models {

HuggingFace makeConfiguredHub() {
    const char* url = std::getenv("GRAPHICSHOST_HUB_URL");
    return HuggingFace(core::huggingFaceToken().value_or(std::string()), url && *url ? url : "https://huggingface.co");
}

}  // namespace gh::models
