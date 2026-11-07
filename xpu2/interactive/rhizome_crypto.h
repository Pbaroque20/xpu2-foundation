// Rhizome's encrypted traffic (RHIZOME.md, 1 and plan 8). Each connection makes a key exchange of its own: both sides
// make a one-time ECDH P-256 key and send its public half in their Hello and Welcome, which each signs with its node key,
// so no machine in between can put its own in. From the shared secret, the two ephemeral keys, the two nodes' hashes
// and the rhizome key come two AES-256 keys, one for each direction. Every frame after the handshake then travels as a
// sealed record (AES-GCM): its header (magic, length) is the associated data, its nonce the direction's count of
// records, so a record changed, cut, played again or put out of order fails to open and the connection is closed.
// Scene files, films and pieces cross the network unread. Windows CNG (bcrypt) does it all.
#pragma once
#include "rhizome_identity.h"
#include <optional>

namespace xpu2::interactive::rhizome {

constexpr uint32_t SealedMagic = 0x455A4852u;  // "RHZE"
constexpr size_t SealedHeaderBytes = 8;       // magic, the ciphertext's length (with its tag)
constexpr size_t TagBytes = 16;

// A one-time ECDH P-256 key pair: its public half to send, and the secret it shares with another's public half.
class Ephemeral {
public:
    Ephemeral(const Ephemeral&) = delete;
    Ephemeral& operator=(const Ephemeral&) = delete;
    Ephemeral() {
        ok(BCryptOpenAlgorithmProvider(&alg_, BCRYPT_ECDH_P256_ALGORITHM, nullptr, 0), "opening ECDH P-256");
        ok(BCryptGenerateKeyPair(alg_, &key_, 256, 0), "making a one-time key");
        ok(BCryptFinalizeKeyPair(key_, 0), "finishing the one-time key");
        ULONG n = 0;
        ok(BCryptExportKey(key_, nullptr, BCRYPT_ECCPUBLIC_BLOB, nullptr, 0, &n, 0), "sizing the one-time key");
        public_.resize(n);
        ok(BCryptExportKey(key_, nullptr, BCRYPT_ECCPUBLIC_BLOB, public_.data(), n, &n, 0), "exporting the one-time key");
        public_.resize(n);
    }
    ~Ephemeral() {
        if (key_) BCryptDestroyKey(key_);
        if (alg_) BCryptCloseAlgorithmProvider(alg_, 0);
    }
    const Bytes& publicKey() const { return public_; }
    // The shared secret with another's public half (32 bytes), or nothing where that is not a P-256 public key.
    std::optional<Bytes> agree(const Bytes& other) const {
        if (other.size() != sizeof(BCRYPT_ECCKEY_BLOB) + 64) return std::nullopt;
        BCRYPT_ECCKEY_BLOB head;
        std::memcpy(&head, other.data(), sizeof head);
        if (head.dwMagic != BCRYPT_ECDH_PUBLIC_P256_MAGIC || head.cbKey != 32) return std::nullopt;
        BCRYPT_KEY_HANDLE peer = nullptr;
        if (BCryptImportKeyPair(alg_, nullptr, BCRYPT_ECCPUBLIC_BLOB, &peer, (PUCHAR)other.data(), ULONG(other.size()), 0) < 0)
            return std::nullopt;
        BCRYPT_SECRET_HANDLE secret = nullptr;
        std::optional<Bytes> out;
        if (BCryptSecretAgreement(key_, peer, &secret, 0) >= 0) {
            Bytes raw(32);
            ULONG got = 0;
            if (BCryptDeriveKey(secret, BCRYPT_KDF_RAW_SECRET, nullptr, raw.data(), ULONG(raw.size()), &got, 0) >= 0 && got == 32)
                out = std::move(raw);
            BCryptDestroySecret(secret);
        }
        BCryptDestroyKey(peer);
        return out;
    }

private:
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_KEY_HANDLE key_ = nullptr;
    Bytes public_;
};

// One direction's AES-256-GCM key and its count of records. A record's nonce is that count (never reused: each
// connection's keys are its own, and the count only rises).
class Seal {
public:
    Seal(const Seal&) = delete;
    Seal& operator=(const Seal&) = delete;
    explicit Seal(const Hash& key) {
        ok(BCryptOpenAlgorithmProvider(&alg_, BCRYPT_AES_ALGORITHM, nullptr, 0), "opening AES");
        ok(BCryptSetProperty(alg_, BCRYPT_CHAINING_MODE, (PUCHAR)BCRYPT_CHAIN_MODE_GCM, sizeof(BCRYPT_CHAIN_MODE_GCM), 0), "AES-GCM");
        ok(BCryptGenerateSymmetricKey(alg_, &key_, nullptr, 0, (PUCHAR)key.data(), ULONG(key.size()), 0), "the AES key");
    }
    ~Seal() {
        if (key_) BCryptDestroyKey(key_);
        if (alg_) BCryptCloseAlgorithmProvider(alg_, 0);
    }
    // A sealed record of a frame: header, ciphertext, tag.
    Bytes seal(const Bytes& plain) {
        Bytes out(SealedHeaderBytes + plain.size() + TagBytes);
        header(out.data(), uint32_t(plain.size() + TagBytes));
        unsigned char nonce[12];
        nonceFor(nonce, count_++);
        BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO info;
        BCRYPT_INIT_AUTH_MODE_INFO(info);
        info.pbNonce = nonce, info.cbNonce = sizeof nonce;
        info.pbAuthData = out.data(), info.cbAuthData = SealedHeaderBytes;
        info.pbTag = out.data() + SealedHeaderBytes + plain.size(), info.cbTag = TagBytes;
        ULONG got = 0;
        ok(BCryptEncrypt(key_, (PUCHAR)plain.data(), ULONG(plain.size()), &info, nullptr, 0, out.data() + SealedHeaderBytes,
                         ULONG(plain.size()), &got, 0),
           "sealing a record");
        return out;
    }
    // A record's frame (header as read, the ciphertext with its tag after it), or nothing: changed, out of order, or
    // not this direction's.
    std::optional<Bytes> open(const unsigned char* head, const Bytes& body) {
        if (body.size() < TagBytes) return std::nullopt;
        const size_t n = body.size() - TagBytes;
        Bytes plain(n);
        unsigned char nonce[12];
        nonceFor(nonce, count_++);
        BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO info;
        BCRYPT_INIT_AUTH_MODE_INFO(info);
        info.pbNonce = nonce, info.cbNonce = sizeof nonce;
        info.pbAuthData = (PUCHAR)head, info.cbAuthData = SealedHeaderBytes;
        info.pbTag = (PUCHAR)body.data() + n, info.cbTag = TagBytes;
        ULONG got = 0;
        if (BCryptDecrypt(key_, (PUCHAR)body.data(), ULONG(n), &info, nullptr, 0, plain.data(), ULONG(n), &got, 0) < 0) return std::nullopt;
        return plain;
    }
    // A sealed record's length from its header, or nothing (not a sealed record).
    static std::optional<uint32_t> length(const unsigned char* head, uint32_t limit) {
        uint32_t magic, n;
        std::memcpy(&magic, head, 4), std::memcpy(&n, head + 4, 4);
        if (magic != SealedMagic || n < TagBytes || n > limit) return std::nullopt;
        return n;
    }

private:
    static void header(unsigned char* p, uint32_t n) {
        const uint32_t magic = SealedMagic;
        std::memcpy(p, &magic, 4), std::memcpy(p + 4, &n, 4);
    }
    static void nonceFor(unsigned char* nonce, uint64_t count) {
        std::memset(nonce, 0, 4);
        std::memcpy(nonce + 4, &count, 8);
    }
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_KEY_HANDLE key_ = nullptr;
    uint64_t count_ = 0;
};

// A connection's two keys, from the shared secret: the dialler's (to the one it dialled) and the other's. Both sides
// work out the same two from what each sent in its signed handshake, and the rhizome key binds them to the swarm.
struct SessionKeys {
    Hash out, in;  // this side's sending and receiving keys
};
inline SessionKeys sessionKeys(const Bytes& secret, const std::string& rhizomeKey, bool dialler, const Bytes& diallerKey,
                               const Bytes& otherKey, const Hash& diallerHash, const Hash& otherHash) {
    Bytes salt;
    auto add = [&](const void* p, size_t n) { salt.insert(salt.end(), (const unsigned char*)p, (const unsigned char*)p + n); };
    add("RHZK", 4);
    add(diallerKey.data(), diallerKey.size()), add(otherKey.data(), otherKey.size());
    add(diallerHash.data(), 32), add(otherHash.data(), 32);
    const Hash saltHash = sha256(salt);
    Bytes material = secret;
    material.insert(material.end(), rhizomeKey.begin(), rhizomeKey.end());
    const Hash prk = sha256(material.data(), material.size(), saltHash.data(), saltHash.size());
    const Hash fromDialler = sha256("dialler to dialled", 18, prk.data(), prk.size());
    const Hash toDialler = sha256("dialled to dialler", 18, prk.data(), prk.size());
    return dialler ? SessionKeys{fromDialler, toDialler} : SessionKeys{toDialler, fromDialler};
}

}  // namespace xpu2::interactive::rhizome
