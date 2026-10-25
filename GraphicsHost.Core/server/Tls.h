#pragma once

#include <filesystem>
#include <string>

namespace gh::server {

// Makes sure there is a certificate to serve https with: a self-signed one (EC P-256, valid for localhost, 127.0.0.1 and
// ::1, about two years) in `dir`, made again when it is missing, unreadable or about to expire. Fills in the two PEM
// paths. Returns an empty string on success, otherwise why not.
std::string ensureSelfSignedCertificate(const std::filesystem::path& dir, std::filesystem::path& certFile, std::filesystem::path& keyFile);

}  // namespace gh::server
