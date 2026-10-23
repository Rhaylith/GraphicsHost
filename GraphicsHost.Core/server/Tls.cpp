#include "server/Tls.h"

#include <openssl/asn1.h>
#include <openssl/bio.h>
#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include <memory>

namespace gh::server {

namespace fs = std::filesystem;

namespace {

template <typename T, void (*Free)(T*)>
struct Deleter {
    void operator()(T* p) const { if (p) Free(p); }
};
using Pkey = std::unique_ptr<EVP_PKEY, Deleter<EVP_PKEY, EVP_PKEY_free>>;
using Cert = std::unique_ptr<X509, Deleter<X509, X509_free>>;
using Bio = std::unique_ptr<BIO, int (*)(BIO*)>;

std::string lastOpenSslError() {
    char buffer[256] = {};
    const unsigned long e = ERR_get_error();
    if (e == 0) return "unknown error";
    ERR_error_string_n(e, buffer, sizeof(buffer));
    return buffer;
}

// A usable certificate that is not about to expire, with a key file next to it.
bool stillGood(const fs::path& certFile, const fs::path& keyFile) {
    std::error_code ec;
    if (!fs::is_regular_file(certFile, ec) || !fs::is_regular_file(keyFile, ec)) return false;
    Bio in(BIO_new_file(certFile.string().c_str(), "rb"), BIO_free);
    if (!in) return false;
    Cert cert(PEM_read_bio_X509(in.get(), nullptr, nullptr, nullptr));
    if (!cert) return false;
    int days = 0, seconds = 0;
    if (!ASN1_TIME_diff(&days, &seconds, nullptr, X509_get0_notAfter(cert.get()))) return false;
    return days > 30;
}

}  // namespace

std::string ensureSelfSignedCertificate(const fs::path& dir, fs::path& certFile, fs::path& keyFile) {
    certFile = dir / "server-cert.pem";
    keyFile = dir / "server-key.pem";
    if (stillGood(certFile, keyFile)) return {};

    std::error_code ec;
    fs::create_directories(dir, ec);
    Pkey key(EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256"));
    if (!key) return "cannot make a key: " + lastOpenSslError();

    Cert cert(X509_new());
    if (!cert) return "cannot make a certificate: " + lastOpenSslError();
    X509_set_version(cert.get(), 2);
    unsigned char serial[8];
    RAND_bytes(serial, sizeof(serial));
    serial[0] &= 0x7f;   // positive
    BIGNUM* bn = BN_bin2bn(serial, sizeof(serial), nullptr);
    BN_to_ASN1_INTEGER(bn, X509_get_serialNumber(cert.get()));
    BN_free(bn);
    X509_gmtime_adj(X509_getm_notBefore(cert.get()), -3600);                 // an hour back: clocks differ
    X509_gmtime_adj(X509_getm_notAfter(cert.get()), 60L * 60 * 24 * 825);   // the longest browsers accept
    X509_set_pubkey(cert.get(), key.get());

    X509_NAME* name = X509_get_subject_name(cert.get());
    X509_NAME_add_entry_by_txt(name, "O", MBSTRING_ASC, reinterpret_cast<const unsigned char*>("GraphicsHost"), -1, -1, 0);
    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, reinterpret_cast<const unsigned char*>("localhost"), -1, -1, 0);
    X509_set_issuer_name(cert.get(), name);

    X509V3_CTX ctx;
    X509V3_set_ctx_nodb(&ctx);
    X509V3_set_ctx(&ctx, cert.get(), cert.get(), nullptr, nullptr, 0);
    auto addExtension = [&](int nid, const char* value) {
        X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &ctx, nid, value);
        if (!ext) return false;
        const bool ok = X509_add_ext(cert.get(), ext, -1) == 1;
        X509_EXTENSION_free(ext);
        return ok;
    };
    if (!addExtension(NID_subject_alt_name, "DNS:localhost,IP:127.0.0.1,IP:::1") || !addExtension(NID_basic_constraints, "critical,CA:FALSE") ||
        !addExtension(NID_key_usage, "critical,digitalSignature") || !addExtension(NID_ext_key_usage, "serverAuth")) {
        return "cannot describe the certificate: " + lastOpenSslError();
    }
    if (X509_sign(cert.get(), key.get(), EVP_sha256()) == 0) return "cannot sign the certificate: " + lastOpenSslError();

    Bio certOut(BIO_new_file(certFile.string().c_str(), "wb"), BIO_free);
    Bio keyOut(BIO_new_file(keyFile.string().c_str(), "wb"), BIO_free);
    if (!certOut || !keyOut) return "cannot write into " + dir.string();
    if (!PEM_write_bio_X509(certOut.get(), cert.get()) || !PEM_write_bio_PrivateKey(keyOut.get(), key.get(), nullptr, nullptr, 0, nullptr, nullptr)) {
        return "cannot write the certificate: " + lastOpenSslError();
    }
    return {};
}

}  // namespace gh::server
