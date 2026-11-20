"""Ed25519 signatures (RFC 8032), in plain Python: the app checks the signature of a model package list with a key
it trusts (instruction 2026-11-20 §4-1) and needs no package for that. Verifying one list takes a few
milliseconds; nothing here is used for anything that has to be fast or secret on the user's computer (the app only
verifies; signing is for the publishing tool and the tests).

The arithmetic follows the RFC's own reference code (section 6); checked against its test vectors
(tests/test_modelpkg.py)."""
from __future__ import annotations

import hashlib

_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _recover_x(y: int, sign: int):
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _I % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * _inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)          # extended coordinates (X, Y, Z, T)


def _add(a, b):
    A = (a[1] - a[0]) * (b[1] - b[0]) % _P
    B = (a[1] + a[0]) * (b[1] + b[0]) % _P
    C = 2 * a[3] * b[3] * _D % _P
    Dd = 2 * a[2] * b[2] % _P
    E, F, G_, H = B - A, Dd - C, Dd + C, B + A
    return (E * F % _P, G_ * H % _P, F * G_ % _P, E * H % _P)


def _mul(s: int, p):
    q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            q = _add(q, p)
        p = _add(p, p)
        s >>= 1
    return q


def _equal(a, b) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _compress(p) -> bytes:
    zi = _inv(p[2])
    x, y = p[0] * zi % _P, p[1] * zi % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(s: bytes):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _h(*parts: bytes) -> int:
    return int.from_bytes(hashlib.sha512(b"".join(parts)).digest(), "little")


def _expand(secret: bytes):
    if len(secret) != 32:
        raise ValueError("an Ed25519 secret key is 32 bytes")
    h = hashlib.sha512(secret).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(secret: bytes) -> bytes:
    a, _ = _expand(secret)
    return _compress(_mul(a, _G))


def sign(secret: bytes, msg: bytes) -> bytes:
    a, prefix = _expand(secret)
    A = _compress(_mul(a, _G))
    r = _h(prefix, msg) % _L
    R = _compress(_mul(r, _G))
    h = _h(R, A, msg) % _L
    s = (r + h * a) % _L
    return R + int.to_bytes(s, 32, "little")


def verify(public: bytes, msg: bytes, signature: bytes) -> bool:
    """True only for a signature of exactly these bytes by the owner of this public key. Never raises."""
    try:
        if len(public) != 32 or len(signature) != 64:
            return False
        A = _decompress(public)
        R = _decompress(signature[:32])
        if A is None or R is None:
            return False
        s = int.from_bytes(signature[32:], "little")
        if s >= _L:
            return False
        h = _h(signature[:32], public, msg) % _L
        return _equal(_mul(s, _G), _add(R, _mul(h, A)))
    except Exception:
        return False
