//! The AirPlay 1 crypto: the receiver key, the `Apple-Response` to a sender's challenge,
//! the session AES key and the per-packet AES-128-CBC.
//!
//! AirPlay 1 senders only talk to receivers that prove they hold the AirPort Express
//! private key. That key was extracted from the device in 2011 and has been public since;
//! shairport-sync and every other open AirPlay 1 receiver embed it, and so does Cadenza
//! (`airport_express.pem`). It protects nothing secret here: senders use it to wrap the
//! per-session AES key, and anyone on the LAN can unwrap it with the same public file.
//! (So the `rsa` crate's timing side channel, RUSTSEC-2023-0071, does not matter either.)

use aes::cipher::{BlockDecryptMut, KeyIvInit};
use base64::alphabet;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use base64::Engine as _;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::{Oaep, Pkcs1v15Sign, RsaPrivateKey};
use std::net::IpAddr;
use std::sync::OnceLock;

const KEY_PEM: &str = include_str!("airport_express.pem");

/// Base64 as senders write it: padding optional.
const B64_IN: GeneralPurpose = GeneralPurpose::new(
    &alphabet::STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);
/// Base64 as receivers answer: no padding.
const B64_OUT: GeneralPurpose = GeneralPurpose::new(
    &alphabet::STANDARD,
    GeneralPurposeConfig::new().with_encode_padding(false),
);

/// Decode sender base64 (whitespace and padding tolerated).
pub fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    B64_IN.decode(clean.trim_end_matches('=')).ok()
}

fn key() -> &'static RsaPrivateKey {
    static K: OnceLock<RsaPrivateKey> = OnceLock::new();
    K.get_or_init(|| RsaPrivateKey::from_pkcs1_pem(KEY_PEM).expect("embedded AirPort key"))
}

/// The `Apple-Response` header for an `Apple-Challenge`: the challenge, the address the
/// sender connected to and our hardware address, zero-padded to 32 bytes and signed raw
/// (PKCS#1 v1.5, no digest). `None` for a malformed challenge.
pub fn apple_response(challenge_b64: &str, local: IpAddr, hw: [u8; 6]) -> Option<String> {
    let challenge = b64_decode(challenge_b64)?;
    if challenge.len() > 16 {
        return None;
    }
    let mut buf = challenge;
    match local {
        IpAddr::V4(a) => buf.extend_from_slice(&a.octets()),
        IpAddr::V6(a) => match a.to_ipv4_mapped() {
            Some(v4) => buf.extend_from_slice(&v4.octets()),
            None => buf.extend_from_slice(&a.octets()),
        },
    }
    buf.extend_from_slice(&hw);
    if buf.len() < 32 {
        buf.resize(32, 0);
    }
    let sig = key().sign(Pkcs1v15Sign::new_unprefixed(), &buf).ok()?;
    Some(B64_OUT.encode(sig))
}

/// Unwrap the session key from the SDP's `rsaaeskey` (RSA-OAEP with SHA-1).
pub fn aes_key(rsaaeskey_b64: &str) -> Option<[u8; 16]> {
    let wrapped = b64_decode(rsaaeskey_b64)?;
    let k = key().decrypt(Oaep::new::<sha1::Sha1>(), &wrapped).ok()?;
    k.try_into().ok()
}

/// The SDP's `aesiv`.
pub fn aes_iv(aesiv_b64: &str) -> Option<[u8; 16]> {
    b64_decode(aesiv_b64)?.try_into().ok()
}

/// Session cipher: every packet is AES-128-CBC from the same IV, whole blocks only; a
/// trailing partial block is sent in the clear.
#[derive(Clone)]
pub struct Cipher {
    key: [u8; 16],
    iv: [u8; 16],
}

impl Cipher {
    /// A cipher for `key` / `iv`.
    pub fn new(key: [u8; 16], iv: [u8; 16]) -> Self {
        Self { key, iv }
    }

    /// Decrypt one RTP payload in place.
    pub fn decrypt(&self, payload: &mut [u8]) {
        let whole = payload.len() / 16 * 16;
        let mut dec = cbc::Decryptor::<aes::Aes128>::new(&self.key.into(), &self.iv.into());
        for block in payload[..whole].chunks_exact_mut(16) {
            dec.decrypt_block_mut(block.into());
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use aes::cipher::BlockEncryptMut;
    use rsa::traits::PublicKeyParts;
    use rsa::RsaPublicKey;

    /// What a sender does to a payload.
    pub(crate) fn encrypt(key: [u8; 16], iv: [u8; 16], payload: &mut [u8]) {
        let whole = payload.len() / 16 * 16;
        let mut enc = cbc::Encryptor::<aes::Aes128>::new(&key.into(), &iv.into());
        for block in payload[..whole].chunks_exact_mut(16) {
            enc.encrypt_block_mut(block.into());
        }
    }

    /// What a sender does to the session key.
    pub(crate) fn wrap_key(k: [u8; 16]) -> String {
        let public = RsaPublicKey::from(key());
        let wrapped = public
            .encrypt(&mut rand_core(), Oaep::new::<sha1::Sha1>(), &k)
            .unwrap();
        B64_OUT.encode(wrapped)
    }

    fn rand_core() -> impl rsa::rand_core::CryptoRngCore {
        struct Counter(u64);
        impl rsa::rand_core::RngCore for Counter {
            fn next_u32(&mut self) -> u32 {
                self.next_u64() as u32
            }
            fn next_u64(&mut self) -> u64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                self.0 >> 11
            }
            fn fill_bytes(&mut self, dest: &mut [u8]) {
                for b in dest {
                    *b = (self.next_u64() >> 24) as u8;
                }
            }
            fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rsa::rand_core::Error> {
                self.fill_bytes(dest);
                Ok(())
            }
        }
        impl rsa::rand_core::CryptoRng for Counter {}
        Counter(42)
    }

    #[test]
    fn embedded_key_is_the_airport_express_key() {
        let n = key().n().to_bytes_be();
        assert_eq!(n.len(), 256);
        // The well-known public modulus begins e7 d7 44 f2 a2 e2 78 8b …
        assert_eq!(&n[..8], &[0xe7, 0xd7, 0x44, 0xf2, 0xa2, 0xe2, 0x78, 0x8b]);
        assert_eq!(key().e().to_bytes_be(), vec![1, 0, 1]);
    }

    #[test]
    fn apple_response_verifies_with_the_public_key() {
        let challenge = [7u8; 16];
        let ip: IpAddr = "192.168.1.20".parse().unwrap();
        let hw = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
        let resp = apple_response(&B64_OUT.encode(challenge), ip, hw).unwrap();
        assert!(!resp.ends_with('='));
        // The same bytes signed by OpenSSL (`pkeyutl -sign`, PKCS#1 padding).
        assert_eq!(
            resp,
            concat!(
                "czc8p8n0IixCWpCqg32lQhRp4jrXsFP1fm4EXKz72L4T3NkOQTWCVp1yaRb+tg6X",
                "byD/n8AxHDphXubcXH+Xdh7ywpWJyTOcBuHJSYn3/iWGcRJDquP8bvJwGRtGkPTi",
                "9HC6jOx2CuCAXRtF6wcGwM5pVLhqPpOOo4dTQRI43t6rw3ga5fXAn34gnN0WAryx",
                "k0eXpWuqUiwNaF/2hYcqMlcw/xSZmUFp0RIHEWkpxFnDE/LDxL19haCAVbBKovhN",
                "LWeAAguwBayPnS8faF4Gx6OF4hcsqeDSQHsbOqZfrRBpkRZZ2o0PIkVdZI88/7Pv",
                "odOsaiGUUAxBFh3mGjLXkQ",
            )
        );
        let sig = b64_decode(&resp).unwrap();
        let mut signed = challenge.to_vec();
        signed.extend_from_slice(&[192, 168, 1, 20]);
        signed.extend_from_slice(&hw);
        signed.resize(32, 0);
        RsaPublicKey::from(key())
            .verify(Pkcs1v15Sign::new_unprefixed(), &signed, &sig)
            .unwrap();
        // A v4-mapped IPv6 socket address signs the IPv4 bytes.
        let mapped: IpAddr = "::ffff:192.168.1.20".parse().unwrap();
        assert_eq!(
            apple_response(&B64_OUT.encode(challenge), mapped, hw).unwrap(),
            resp
        );
        assert!(apple_response("!!!", ip, hw).is_none());
    }

    #[test]
    fn session_key_unwraps_and_packets_decrypt() {
        let k = [0x5au8; 16];
        let iv = [0x33u8; 16];
        assert_eq!(aes_key(&wrap_key(k)), Some(k));
        assert_eq!(aes_iv(&B64_OUT.encode(iv)), Some(iv));
        let plain: Vec<u8> = (0..45u8).collect(); // 2 whole blocks + 13 clear bytes
        let mut p = plain.clone();
        encrypt(k, iv, &mut p);
        assert_ne!(p[..32], plain[..32]);
        assert_eq!(p[32..], plain[32..]);
        Cipher::new(k, iv).decrypt(&mut p);
        assert_eq!(p, plain);
    }
}
