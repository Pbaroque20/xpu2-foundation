// Rhizome's node identity (RHIZOME.md, 1): an ECDSA P-256 key pair made once and kept in the node's folder, and the node's
// unique hash, the SHA-256 of its public key. Every message a node sends is signed with its key; a receiver checks
// the signature with the public key the node introduced itself with, and that the key hashes to the hash it claims.
// The rhizome key, a secret every member is given, proves membership: a keyed hash (HMAC-SHA-256) of the other's
// challenge. Windows CNG (bcrypt) does the hashing and the signatures.
#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <cstdio>
#include <memory>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace xpu2::interactive::rhizome {

using Bytes = std::vector<unsigned char>;
// A node's unique hash: the SHA-256 of its public key.
using Hash = std::array<unsigned char, 32>;

inline void ok(NTSTATUS s, const char* what) {
    if (s < 0) {
        char text[96];
        std::snprintf(text, sizeof text, "%s failed (0x%08lx)", what, (unsigned long)s);
        throw std::runtime_error(text);
    }
}

inline std::string hex(const unsigned char* p, size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s(n * 2, '0');
    for (size_t i = 0; i < n; ++i) s[2 * i] = digits[p[i] >> 4], s[2 * i + 1] = digits[p[i] & 15];
    return s;
}
inline std::string hex(const Hash& h) { return hex(h.data(), h.size()); }
// The first 12 digits, for the Terminal.
inline std::string shortHex(const Hash& h) { return hex(h).substr(0, 12); }

// SHA-256, or HMAC-SHA-256 with a key.
inline Hash sha256(const void* data, size_t n, const void* key = nullptr, size_t keyBytes = 0) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    ok(BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, key ? BCRYPT_ALG_HANDLE_HMAC_FLAG : 0), "opening SHA-256");
    BCRYPT_HASH_HANDLE h = nullptr;
    Hash out{};
    NTSTATUS s = BCryptCreateHash(alg, &h, nullptr, 0, (PUCHAR)key, ULONG(keyBytes), 0);
    if (s >= 0) s = BCryptHashData(h, (PUCHAR)data, ULONG(n), 0);
    if (s >= 0) s = BCryptFinishHash(h, out.data(), ULONG(out.size()), 0);
    if (h) BCryptDestroyHash(h);
    BCryptCloseAlgorithmProvider(alg, 0);
    ok(s, "hashing");
    return out;
}
inline Hash sha256(const Bytes& b) { return sha256(b.data(), b.size()); }

// Random bytes from the system's generator (challenges, the audit's seed).
inline void randomBytes(void* out, size_t n) {
    ok(BCryptGenRandom(nullptr, (PUCHAR)out, ULONG(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG), "random bytes");
}

// The rhizome key's proof: HMAC-SHA-256 of a challenge under the rhizome key, bound to the prover's hash.
inline Hash membershipProof(const std::string& rhizomeKey, const Hash& challenge, const Hash& prover) {
    unsigned char m[64];
    std::memcpy(m, challenge.data(), 32);
    std::memcpy(m + 32, prover.data(), 32);
    return sha256(m, sizeof m, rhizomeKey.data(), rhizomeKey.size());
}

// Checks a signature over data with a public key (BCRYPT_ECCPUBLIC_BLOB, P-256).
inline bool verify(const Bytes& publicBlob, const void* data, size_t n, const Bytes& signature) {
    if (publicBlob.size() != sizeof(BCRYPT_ECCKEY_BLOB) + 64 || signature.size() != 64) return false;
    BCRYPT_ALG_HANDLE alg = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_ECDSA_P256_ALGORITHM, nullptr, 0) < 0) return false;
    BCRYPT_KEY_HANDLE key = nullptr;
    bool good = false;
    if (BCryptImportKeyPair(alg, nullptr, BCRYPT_ECCPUBLIC_BLOB, &key, (PUCHAR)publicBlob.data(), ULONG(publicBlob.size()), 0) >= 0) {
        const Hash digest = sha256(data, n);
        good = BCryptVerifySignature(key, nullptr, (PUCHAR)digest.data(), ULONG(digest.size()), (PUCHAR)signature.data(),
                                     ULONG(signature.size()), 0) >= 0;
        BCryptDestroyKey(key);
    }
    BCryptCloseAlgorithmProvider(alg, 0);
    return good;
}

// A node's identity: its key pair and its hash.
class Identity {
public:
    Identity(const Identity&) = delete;
    Identity& operator=(const Identity&) = delete;
    ~Identity() {
        if (key_) BCryptDestroyKey(key_);
        if (alg_) BCryptCloseAlgorithmProvider(alg_, 0);
    }

    // A new key pair, kept nowhere (tests; a node that must not keep its hash).
    static std::unique_ptr<Identity> make() {
        auto id = std::unique_ptr<Identity>(new Identity());
        ok(BCryptGenerateKeyPair(id->alg_, &id->key_, 256, 0), "making a key pair");
        ok(BCryptFinalizeKeyPair(id->key_, 0), "finishing the key pair");
        id->derive();
        return id;
    }
    // The node's identity kept in folder/node.key: read if it is there, else made and written. The file holds the
    // private key; it is the node's alone (on Windows the folder's permissions are the user's).
    static std::unique_ptr<Identity> load(const std::filesystem::path& folder) {
        const auto path = folder / "node.key";
        if (std::filesystem::exists(path)) {
            std::ifstream in(path, std::ios::binary);
            Bytes blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            auto id = std::unique_ptr<Identity>(new Identity());
            ok(BCryptImportKeyPair(id->alg_, nullptr, BCRYPT_ECCPRIVATE_BLOB, &id->key_, blob.data(), ULONG(blob.size()), 0),
               "reading the node's key");
            id->derive();
            return id;
        }
        auto id = make();
        std::filesystem::create_directories(folder);
        const Bytes blob = id->exportBlob(BCRYPT_ECCPRIVATE_BLOB);
        const auto temp = folder / "node.key.part";
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(blob.data()), std::streamsize(blob.size()));
            if (!out) throw std::runtime_error("writing the node's key to " + path.string());
        }
        std::filesystem::rename(temp, path);
        return id;
    }

    const Hash& hash() const { return hash_; }
    const Bytes& publicKey() const { return public_; }
    // An ECDSA signature (r and s, 64 bytes) of SHA-256(data).
    Bytes sign(const void* data, size_t n) const {
        const Hash digest = sha256(data, n);
        Bytes sig(64);
        ULONG got = 0;
        ok(BCryptSignHash(key_, nullptr, (PUCHAR)digest.data(), ULONG(digest.size()), sig.data(), ULONG(sig.size()), &got, 0), "signing");
        sig.resize(got);
        return sig;
    }

private:
    Identity() { ok(BCryptOpenAlgorithmProvider(&alg_, BCRYPT_ECDSA_P256_ALGORITHM, nullptr, 0), "opening ECDSA P-256"); }
    Bytes exportBlob(LPCWSTR type) const {
        ULONG n = 0;
        ok(BCryptExportKey(key_, nullptr, type, nullptr, 0, &n, 0), "sizing the key");
        Bytes b(n);
        ok(BCryptExportKey(key_, nullptr, type, b.data(), n, &n, 0), "exporting the key");
        b.resize(n);
        return b;
    }
    void derive() {
        public_ = exportBlob(BCRYPT_ECCPUBLIC_BLOB);
        hash_ = sha256(public_);
    }
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_KEY_HANDLE key_ = nullptr;
    Bytes public_;
    Hash hash_{};
};

// Whether a public key is the one a hash names.
inline bool keyMatches(const Bytes& publicBlob, const Hash& claimed) { return sha256(publicBlob) == claimed; }

}  // namespace xpu2::interactive::rhizome
