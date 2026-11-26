"""The retouch's look, held per project (1.26.3; the review of 11/26: the engine's defaults change between versions, and a cut of a
delivered project rendered again after an update came out with another skin than the cuts delivered before it).

A look is the engine's defaults of one version. The newest look is the engine as it is; an older one is the set of settings that
differ, put back for as long as a project that keeps it is worked on (use()). The look a project keeps is Project.look:
"" = the newest (it follows the app), "1.26.1" = the defaults 1.26.1 rendered with.

    LOOK = the newest look's name. A version that changes the defaults adds the outgoing values under the outgoing version's name.

Not part of a look: how fast (the threads), what a slider does once it is moved (the sliders a project saves are its own)."""
from __future__ import annotations

import copy
import threading
from typing import Dict, Optional

LOOK = "1.26.2"

# the settings 1.26.2 changed (module.NAME -> the 1.26.1 value): v1.26.1 against 1.26.2, every module-level setting compared
# (tools/look_diff.py); the ones added in 1.26.2 at the value that leaves them out. tests/test_looks.py renders the frozen baseline
# projects with them and holds the result to what 1.26.1 rendered.
LOOKS: Dict[str, Dict[str, object]] = {
    "1.26.1": {
        "engine.CHIN_EDGE_GATE": False,
        "engine.CHIN_HI": 0.085,
        "engine.CHIN_KEEP_EDGE": 0.0,
        "engine.CHIN_KEEP_LIP": 0.0,
        "engine.CHIN_LO": 0.045,
        "engine.DARK_CHROMA": 0.8,
        "engine.FACE_GAP_BRIDGE": 0,
        "engine.FACE_HAIR_HOLD": 0.0,
        "engine.FACE_NOSE_ADD": 0.0,
        "engine.FINAL_BUMP_EDGE": None,
        "engine.FINAL_BUMP_LIPS": 0.0,
        "engine.FINAL_MOLE": 0.0,
        "engine.FINAL_NASO_IN_BAND": 0.0,
        "engine.FINAL_NASO_SKIN": 0.0,
        "engine.FINAL_NASO_SLIDER": False,
        "engine.FINAL_NOSE": 0.0,
        "engine.FOLD_BASE": {"nasolabial": 0.06, "mouth_lines": 0.06},
        "engine.FOLD_COVER_GUARD": 0.0,
        "engine.FOLD_EDGE_FADE": 0.0,
        "engine.FOLD_LEVEL_LIFT_ONLY": False,
        "engine.FOLD_PARSE_GUARD": 0.0,
        "engine.FOLD_TOTAL_CAP": (0.035, 0.025),
        "engine.FOLD_YAW": None,
        "engine.FOLD_ZONE_OLD": True,
        "engine.HAND_GUARD": 0.0,
        "engine.HOLD_MOLE": (0.3, 0.45),
        "engine.MOLE_KEEP": 0.0,
        "engine.MOUTH_OPEN_KEEP": 0.4,
        "engine.NASO_BULGE": 0.1,
        "engine.NASO_DEEP_NOSE": 0.0,
        "engine.NASO_FOLD": {"broad": 1.1, "broad_cap": 0.08, "level": 1.7, "excl": (0.05, 0.02), "edge": (0.012, 0.02), "cap": (0.1, 0.004)},
        "engine.NASO_NOSTRIL_HOLD": 0.0,
        "engine.NASO_REACH": 0.07,
        "engine.NASO_WIDE_MUL": 1.0,
        "engine.NASO_WING_TOP": 0.0,
        "engine.NASO_WING_TOP_FROM": 0.012,
        "engine.NASO_WING_TOP_TO": 0.05,
        "engine.NOSE_SIDES": False,
        "engine.NOSE_SIDES_WING": 0.12,
        "engine.SKIN_BROW_HOLD": 0.0,
        "engine.SKIN_EDGE_KEEP": 0.0,
        "engine.SKIN_FS_G": None,
        "skin.FS_G": (1.35, 0.70, 0.36, 0.33, 0.45, 0.80),
        "skin.FS_EDGE_GRAD": None,
    },
}
LABELS = {"1.26.1": "1.26.1 (IWALY 얼굴 기본값, LNGSHOT 수정 이전)", "1.26.2": "1.26.2 (최신)"}

_lock = threading.RLock()
_base: Dict[str, object] = {}         # module.NAME -> the newest look's value (the engine as imported), taken before the first change
_now = [LOOK]


def _mod(name: str):
    from . import engine as E, skin as S
    return {"engine": E, "skin": S}[name]


def names() -> list:
    """The looks a project can keep, newest first."""
    return [LOOK] + sorted(LOOKS, reverse=True)


def label(look: str) -> str:
    return LABELS.get(look or LOOK, look or LOOK)


def known(look: str) -> bool:
    return (look or LOOK) == LOOK or look in LOOKS


def current() -> str:
    return _now[0]


def use(look: Optional[str]) -> str:
    """Puts the engine in look `look` ("" / None / an unknown name: the newest). Returns the look now in use."""
    look = look or LOOK
    if look != LOOK and look not in LOOKS:
        look = LOOK
    with _lock:
        if look == _now[0]:
            return look
        keys = set()
        for d in LOOKS.values():
            keys.update(d)
        for k in keys:
            if k not in _base:
                m, n = k.split(".", 1)
                _base[k] = copy.deepcopy(getattr(_mod(m), n))
        over = LOOKS.get(look, {})
        for k in keys:
            m, n = k.split(".", 1)
            setattr(_mod(m), n, copy.deepcopy(over[k] if k in over else _base[k]))
        _now[0] = look
    return look


def for_project(project) -> str:
    """The look a project keeps ("" = the newest)."""
    return str(getattr(project, "look", "") or "")


class held:
    """with held(project): ... the project's look for the block, the one before it after."""

    def __init__(self, project):
        self.look = for_project(project)

    def __enter__(self):
        with _lock:
            self.before = current()
            use(self.look)
        return self

    def __exit__(self, *exc):
        use(self.before)
        return False
