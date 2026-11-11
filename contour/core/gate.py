"""Start-up password. Only a salted hash is kept in the code, never the password itself.

This keeps the app closed to anyone who simply gets the installer. It is not protection against someone who
takes the Python source apart: the check runs on the user's machine and can be removed from the code there.
"""
from __future__ import annotations

import hashlib
import hmac

_SALT = b"contour.sewoo.gate.v1"
_HASH = "edf18d8b621a2d0add58c1160696a0284c906f505f97fabaaebf099b28aa976b"
_ROUNDS = 200_000


def _h(pw: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), _SALT, _ROUNDS).hex()


def check(pw: str) -> bool:
    return hmac.compare_digest(_h(pw.strip()), _HASH)
