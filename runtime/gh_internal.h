#pragma once
// Shared by the translation units of the runtime DLLs. Not part of the ABI.

#include <stable-diffusion.h>

#include <string>

struct gh_model {
    sd_ctx_t* ctx = nullptr;
    std::string family;
    void* session = nullptr;   // an open engine scope (gh_session_begin), owned by gh_components.cpp
};
