"""Project data model (JSON) — cuts, people, reshape parameters, protection, output."""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .media import MediaInfo

PROJECT_EXT = ".dpprj"              # project folder (older projects end in .contour and open the same)
PROJECT_EXTS = (".dpprj", ".contour")

from .params import ALL_KEYS, GEOMETRY_KEYS, PARAM, RETOUCH_KEYS, WIDTH_KEYS, default as param_default

PART_KEYS = WIDTH_KEYS                     # kept for older imports

PERSON_COLORS = ["#4CC9F0", "#F5B841", "#9BE15D", "#B18CFF", "#FF8A4C", "#3ED6B5", "#7AA2FF", "#E6E26B",
                 "#5FE3E0", "#C9A0FF"]

OUTPUT_FORMATS = [
    ("prores_422_hq", "ProRes 422 HQ"),
    ("prores_4444", "ProRes 4444"),
    ("prores_4444_xq", "ProRes 4444 XQ"),
    ("prores_422", "ProRes 422"),
    ("exr_half", "OpenEXR 시퀀스 (half)"),
]


def safe_name(s: str) -> str:
    """Folder/file safe: spaces -> underscores, no '&' or path separators."""
    s = s.strip().replace("&", "and")
    s = re.sub(r"[\\/:*?\"<>|]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s or "untitled"


# --------------------------------------------------------------------------- animated values

SKIN_REGIONS = [("face", "얼굴"), ("neck", "목"), ("shoulder", "어깨·쇄골"), ("arm", "팔"), ("hand", "손"), ("leg", "다리")]


def norm_scope(v) -> str:
    """Skin regions as 'face,neck,...'. Older projects: 'face' (face only), 'all' (all visible skin)."""
    keys = [k for k, _ in SKIN_REGIONS]
    if v == "all":
        return ",".join(keys)
    got = [k for k in keys if k in str(v or "").split(",")]
    return ",".join(got) or "face"


def scope_set(v) -> set:
    return set(norm_scope(v).split(","))


def norm_body_scope(v) -> str:
    """Where the body skin effects go: 'all' (전신) or body regions as 'neck,arm,...' (params.BODY_REGIONS).
    Anything else, and nothing at all, is 전신."""
    from .params import BODY_ALL, BODY_REGIONS
    keys = [k for k, _ in BODY_REGIONS]
    got = [k for k in keys if k in str(v or "").split(",")]
    return ",".join(got) if got and len(got) < len(keys) else BODY_ALL


def body_scope_set(v) -> set:
    """The body regions of a scope ('all': every one of them)."""
    from .params import BODY_ALL, BODY_REGIONS
    s = norm_body_scope(v)
    return {k for k, _ in BODY_REGIONS} if s == BODY_ALL else set(s.split(","))


BODY_MARK_KINDS = ("remove", "exclude")      # 몸 잡티 marks by hand: this spot goes / nothing is removed here


def norm_body_marks(v) -> list:
    """Body blemish marks as saved: [{id, kind, t, pts [[x, y], ...] and r in source pixels of frame t, part,
    uv [[u, v], ...], ru}] (retouch.bodyskin anchors them to a body part; a mark without a part stays where it
    was drawn). Marks that cannot be read are left out."""
    out = []
    for m in v or []:
        try:
            if m.get("kind") not in BODY_MARK_KINDS:
                continue
            pts = [[float(x), float(y)] for x, y in m.get("pts") or []]
            if not pts:
                continue
            d = dict(id=str(m.get("id") or uuid.uuid4().hex[:8]), kind=m["kind"], t=int(m.get("t", 0)),
                     pts=pts, r=float(m.get("r", 6.0)), part=str(m.get("part") or ""))
            uv = [[float(a), float(b)] for a, b in m.get("uv") or []]
            if d["part"] and len(uv) == len(pts):
                d["uv"] = uv
                d["ru"] = float(m.get("ru", 0.0))
            else:
                d["part"] = ""
            out.append(d)
        except Exception:
            continue
    return out


@dataclass
class Curve:
    base: float = 0.0
    keys: Dict[int, float] = field(default_factory=dict)
    smooth: bool = True

    def value(self, frame: float) -> float:
        if not self.keys:
            return self.base
        ks = sorted(self.keys)
        if frame <= ks[0]:
            return self.keys[ks[0]]
        if frame >= ks[-1]:
            return self.keys[ks[-1]]
        lo = max(k for k in ks if k <= frame)
        hi = min(k for k in ks if k >= frame)
        if hi == lo:
            return self.keys[lo]
        t = (frame - lo) / (hi - lo)
        if self.smooth:
            t = t * t * (3 - 2 * t)
        return self.keys[lo] * (1 - t) + self.keys[hi] * t

    def set_at(self, frame: int, v: float) -> None:
        """Animated curves get a key at `frame`; static curves change their base value."""
        if self.keys:
            self.keys[int(frame)] = float(v)
        else:
            self.base = float(v)

    def add_key(self, frame: int) -> None:
        self.keys[int(frame)] = self.value(frame)

    def remove_key(self, frame: int) -> None:
        if int(frame) in self.keys:
            v = self.keys.pop(int(frame))
            if not self.keys:
                self.base = v

    def has_key(self, frame: int) -> bool:
        return int(frame) in self.keys

    def is_animated(self) -> bool:
        return bool(self.keys)

    def to_dict(self) -> dict:
        return {"base": self.base, "keys": {str(k): v for k, v in self.keys.items()}, "smooth": self.smooth}

    @classmethod
    def from_dict(cls, d) -> "Curve":
        if isinstance(d, (int, float)):
            return cls(base=float(d))
        d = d or {}
        return cls(base=float(d.get("base", 0.0)), keys={int(k): float(v) for k, v in (d.get("keys") or {}).items()},
                   smooth=bool(d.get("smooth", True)))


@dataclass
class PartParams:
    """Per person settings (see core/params.py for every key, range and meaning).

    Static extras: skin texture kept, skin scope, grain size, range multipliers per body part and the
    user's face sculpt (landmark offsets in face units: x along the eye line, y towards the chin,
    1.0 = distance between the eye centres)."""
    amounts: Dict[str, Curve] = field(default_factory=lambda: {k: Curve(base=param_default(k)) for k in ALL_KEYS})
    widths: Dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in WIDTH_KEYS})
    strength: Curve = field(default_factory=lambda: Curve(base=100.0))
    skin_detail: float = 45.0        # % of fine texture kept when smoothing
    skin_scope: str = "face,neck"    # skin regions, comma separated: face, neck, shoulder, arm, hand, leg
    grain_size: float = 1.0          # grain particle size (px at 1080p)
    blemish_size: float = 50.0       # 0..100 largest spot the blemish removal takes (share of the face width)
    # body skin (1.23; amounts 'body_skin' and 'body_blemish', 0 = off)
    body_detail: float = 60.0        # 결 보존: % of the fine texture (pores, fine hair) the body smoothing gives back
    body_blemish_size: float = 40.0  # 0..100 largest spot the body blemish removal takes (share of the body part's size)
    body_scope: str = "all"          # 'all' (전신) or body regions: neck, arm, torso, leg (norm_body_scope)
    body_memo: Dict[str, dict] = field(default_factory=dict)   # an effect switched off: its curve, to come back as it was
    body_marks: List[dict] = field(default_factory=list)       # 몸 잡티 marks by hand (norm_body_marks)
    cloth: str = "include"           # body edits and clothes: include | skirt (skirt, dress below the waist kept) | all
    feather: float = 50.0            # 0..100 how softly shape changes fade into their surroundings
    sculpt: Dict[int, List[float]] = field(default_factory=dict)   # dense face point index -> [du, dv]
    body_sculpt: Dict[str, float] = field(default_factory=dict)    # "segment:k:side" -> edge change (fraction)
    # which side a body point edit means: 'person' (1.4: the person's own side, stays put when she turns
    # around) or 'measured' (1.3 projects: the side as measured in the picture, kept as it was made)
    body_sides: str = "person"
    # the skin tab's basic adjustment the detail sliders were last set from: {"level": light|std|strong,
    # "remove": [mouth, naso, acne]}; empty once any skin or colour value is changed by hand (shown unselected)
    skin_basic: Dict[str, object] = field(default_factory=dict)

    def feather_factor(self) -> float:
        """0..100 -> 0.5..2.5 (50 = 1.5)."""
        return 0.5 + max(0.0, min(100.0, self.feather)) / 50.0

    def amount(self, part: str, frame: float) -> float:
        c = self.amounts.get(part)
        return c.value(frame) if c else param_default(part)

    def _active(self, k: str) -> bool:
        c = self.amounts.get(k)
        if c is None:
            return False
        d = param_default(k)
        if k == "sculpt":
            return bool(self.sculpt) and (abs(c.base) > 1e-6 or any(abs(v) > 1e-6 for v in c.keys.values()))
        if k == "body_sculpt":
            return bool(self.body_sculpt) and (abs(c.base) > 1e-6 or any(abs(v) > 1e-6 for v in c.keys.values()))
        return abs(c.base - d) > 1e-6 or any(abs(v - d) > 1e-6 for v in c.keys.values())

    def any_active(self) -> bool:
        return any(self._active(k) for k in self.amounts)

    def geometry_active(self) -> bool:
        return any(self._active(k) for k in GEOMETRY_KEYS)

    def retouch_active(self) -> bool:
        return any(self._active(k) for k in RETOUCH_KEYS)

    def skin_active(self) -> bool:
        return self.retouch_active()

    # ---- body skin
    def body_on(self, key: str) -> bool:
        """A body skin effect (body_skin | body_blemish) is switched on: its value is above 0 somewhere."""
        c = self.amounts.get(key)
        return c is not None and (c.base > 1e-6 or any(v > 1e-6 for v in c.keys.values()))

    def set_body_on(self, key: str, on: bool) -> None:
        """Switch a body skin effect on or off. Off keeps its values (keys too) to come back to; on the first
        time it starts from params.BODY_START."""
        from .params import BODY_SKIN_KEYS, BODY_START
        if key not in BODY_SKIN_KEYS or on == self.body_on(key):
            return
        if on:
            memo = self.body_memo.pop(key, None)
            c = Curve.from_dict(memo) if memo else Curve(base=BODY_START[key])
            if not (c.base > 1e-6 or any(v > 1e-6 for v in c.keys.values())):
                c = Curve(base=BODY_START[key])
            self.amounts[key] = c
        else:
            self.body_memo[key] = self.amounts[key].to_dict()
            self.amounts[key] = Curve(base=0.0)

    def copy_body_from(self, src: "PartParams", marks: bool = False) -> None:
        """The body skin settings that are not sliders (texture kept, largest spot, regions, the switched-off
        values). marks: the hand marks too (they belong to one cut's frames and one person's body: only for a copy
        of the same person in the same cut)."""
        self.body_detail = float(src.body_detail)
        self.body_blemish_size = float(src.body_blemish_size)
        self.body_scope = norm_body_scope(src.body_scope)
        self.body_memo = {k: dict(v) for k, v in src.body_memo.items()}
        if marks:
            self.body_marks = norm_body_marks(src.body_marks)

    def keyed_parts(self) -> List[str]:
        return [k for k in ALL_KEYS if self.amounts.get(k) is not None and self.amounts[k].keys]

    def to_dict(self) -> dict:
        return {"amounts": {k: c.to_dict() for k, c in self.amounts.items()}, "widths": dict(self.widths),
                "strength": self.strength.to_dict(), "skin_detail": self.skin_detail, "skin_scope": self.skin_scope,
                "grain_size": self.grain_size, "feather": self.feather, "blemish_size": self.blemish_size, "cloth": self.cloth,
                **self._body_dict(),
                "sculpt": {str(k): list(v) for k, v in self.sculpt.items()},
                "body_sculpt": dict(self.body_sculpt), "body_sides": self.body_sides,
                **({"skin_basic": {"level": self.skin_basic.get("level", "std"),
                                   "remove": list(self.skin_basic.get("remove", [])),
                                   **({"parts": dict(self.skin_basic["parts"])} if self.skin_basic.get("parts") is not None else {})}}
                   if self.skin_basic else {})}

    def _body_dict(self) -> dict:
        """The body skin settings, only what is not as a new person has it (a project that never used them is
        written as before)."""
        out = {}
        if abs(self.body_detail - 60.0) > 1e-9:
            out["body_detail"] = self.body_detail
        if abs(self.body_blemish_size - 40.0) > 1e-9:
            out["body_blemish_size"] = self.body_blemish_size
        if norm_body_scope(self.body_scope) != "all":
            out["body_scope"] = norm_body_scope(self.body_scope)
        if self.body_memo:
            out["body_memo"] = {k: dict(v) for k, v in self.body_memo.items()}
        if self.body_marks:
            out["body_marks"] = norm_body_marks(self.body_marks)
        return out

    @classmethod
    def from_dict(cls, d) -> "PartParams":
        p = cls()
        d = d or {}
        am = dict(d.get("amounts") or {})
        # 1.0/1.1: one 'arms' value covered the whole arm
        if "arms" in am:
            for k in ("upper_arm", "forearm"):
                am.setdefault(k, am["arms"])
            am.pop("arms")
        for k, v in am.items():
            if k in PARAM:
                p.amounts[k] = Curve.from_dict(v)
        for k in ALL_KEYS:
            p.amounts.setdefault(k, Curve(base=param_default(k)))
        w = dict(d.get("widths") or {})
        if "arms" in w:
            for k in ("upper_arm", "forearm"):
                w.setdefault(k, w["arms"])
        p.widths.update({k: float(v) for k, v in w.items() if k in WIDTH_KEYS})
        if "strength" in d:
            p.strength = Curve.from_dict(d["strength"])
        p.skin_detail = float(d.get("skin_detail", 45.0))
        p.skin_scope = norm_scope(d.get("skin_scope", "face"))
        p.grain_size = float(d.get("grain_size", 1.0))
        p.blemish_size = float(d.get("blemish_size", 50.0))
        p.body_detail = min(100.0, max(0.0, float(d.get("body_detail", 60.0))))
        p.body_blemish_size = min(100.0, max(0.0, float(d.get("body_blemish_size", 40.0))))
        p.body_scope = norm_body_scope(d.get("body_scope", "all"))
        from .params import BODY_SKIN_KEYS
        p.body_memo = {k: dict(v) for k, v in (d.get("body_memo") or {}).items() if k in BODY_SKIN_KEYS and isinstance(v, dict)}
        p.body_marks = norm_body_marks(d.get("body_marks"))
        p.cloth = d.get("cloth", "include") if d.get("cloth") in ("include", "skirt", "all") else "include"
        p.feather = float(d.get("feather", 50.0))
        p.sculpt = {int(k): [float(v[0]), float(v[1])] for k, v in (d.get("sculpt") or {}).items()}
        p.body_sculpt = {str(k): float(v) for k, v in (d.get("body_sculpt") or {}).items()}
        p.body_sides = d.get("body_sides") or ("measured" if p.body_sculpt else "person")
        sb = d.get("skin_basic") or {}
        if isinstance(sb, dict) and sb.get("level") in ("light", "std", "strong"):
            p.skin_basic = {"level": sb["level"], "remove": [str(x) for x in (sb.get("remove") or [])]}
            if isinstance(sb.get("parts"), dict):
                p.skin_basic["parts"] = {str(k): str(v) for k, v in sb["parts"].items()}    # (1.26) the parts' steps: eye whites, knees, elbows
            p._basic_upgrade()
        return p

    def _basic_upgrade(self) -> None:
        """A basic adjustment made by an older version: its level's values of then become the level's values of
        now, only where the value is still exactly the old one and has no keys (set by hand: left as it is)."""
        from .params import (SKIN_LEVEL_VALUES, SKIN_LEVEL_VALUES_V1, SKIN_REMOVE, SKIN_REMOVE_VALUES,
                             SKIN_REMOVE_VALUES_V1)
        lv = self.skin_basic.get("level")
        old, new = dict(SKIN_LEVEL_VALUES_V1.get(lv, {})), dict(SKIN_LEVEL_VALUES.get(lv, {}))
        rm = set(self.skin_basic.get("remove", []))
        for sw, key, _ in SKIN_REMOVE:
            if sw in rm and lv in SKIN_REMOVE_VALUES_V1:
                old[key], new[key] = SKIN_REMOVE_VALUES_V1[lv], SKIN_REMOVE_VALUES[lv]
        for k, v_old in old.items():
            v_new = new.get(k, v_old)
            c = self.amounts.get(k)
            if v_new == v_old or c is None or c.keys:
                continue
            if abs(c.base - v_old) < 1e-6:
                c.base = float(v_new)

    def copy_static(self, frame: float) -> "PartParams":
        """Values evaluated at `frame`, without keyframes (used for cut-to-cut copy)."""
        p = PartParams()
        for k in ALL_KEYS:
            p.amounts[k] = Curve(base=self.amount(k, frame))
        for k in WIDTH_KEYS:
            p.widths[k] = self.widths.get(k, 1.0)
        p.strength = Curve(base=self.strength.value(frame))
        p.skin_detail = self.skin_detail
        p.skin_scope = self.skin_scope
        p.grain_size = self.grain_size
        p.blemish_size = self.blemish_size
        p.copy_body_from(self)
        p.cloth = self.cloth
        p.feather = self.feather
        p.sculpt = {k: list(v) for k, v in self.sculpt.items()}
        p.body_sculpt = dict(self.body_sculpt)
        p.body_sides = self.body_sides
        p.skin_basic = {"level": self.skin_basic["level"], "remove": list(self.skin_basic.get("remove", []))} \
            if self.skin_basic else {}
        if self.skin_basic and self.skin_basic.get("parts") is not None:
            p.skin_basic["parts"] = dict(self.skin_basic["parts"])
        return p


# --------------------------------------------------------------------------- people & protection

@dataclass
class Person:
    id: int
    name: str
    color: str
    role: str = "protect"           # target | protect | ignore
    profile: List[float] = field(default_factory=list)   # appearance descriptor
    first: int = 0
    last: int = 0
    reid: List[float] = field(default_factory=list)      # body re-identification feature (unit vector)
    faces: List[List[float]] = field(default_factory=list)  # a few face features (unit vectors)
    member: str = ""                 # who this is across the edited timeline (Project.members id)
    member_conf: float = 0.0         # how sure the automatic link is (0..1); user links are 1
    still: bool = False              # a poster, photo or screen on the set (found by analysis; protected)
    hold: str = ""                   # "added" (found by the person finder only) or "unconfirmed" (the finder never
                                     # confirmed the track): kept out of the automatic targets until the user says

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ProtectLine:
    id: str
    keys: Dict[int, List[float]] = field(default_factory=dict)  # frame -> [x1,y1,x2,y2] source px
    enabled: bool = True

    def to_dict(self):
        return {"id": self.id, "keys": {str(k): v for k, v in self.keys.items()}, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d):
        return cls(id=d.get("id") or uuid.uuid4().hex[:8], keys={int(k): list(v) for k, v in (d.get("keys") or {}).items()},
                   enabled=bool(d.get("enabled", True)))


@dataclass
class ProtectShape:
    id: str
    kind: str = "ellipse"            # ellipse | rect
    keys: Dict[int, List[float]] = field(default_factory=dict)  # frame -> [cx,cy,rx,ry]
    enabled: bool = True

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "keys": {str(k): v for k, v in self.keys.items()}, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d):
        return cls(id=d.get("id") or uuid.uuid4().hex[:8], kind=d.get("kind", "ellipse"),
                   keys={int(k): list(v) for k, v in (d.get("keys") or {}).items()}, enabled=bool(d.get("enabled", True)))


def _reach(v) -> float:
    v = float(v)
    return 1.9 if abs(v - 2.6) < 1e-6 else v     # 1.0/1.1 default spread too far into the background


@dataclass
class ProtectSettings:
    others: bool = True             # keep non-target people untouched
    falloff_pct: float = 3.0        # transition width around protected areas, % of frame height
    border: bool = True
    reach: float = 1.9              # how far the warp spreads beyond the body edge (x limb half-width)
    lines: List[ProtectLine] = field(default_factory=list)
    shapes: List[ProtectShape] = field(default_factory=list)

    def to_dict(self):
        return {"others": self.others, "falloff_pct": self.falloff_pct, "border": self.border, "reach": self.reach,
                "lines": [l.to_dict() for l in self.lines], "shapes": [s.to_dict() for s in self.shapes]}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(others=bool(d.get("others", True)), falloff_pct=float(d.get("falloff_pct", 3.0)),
                   border=bool(d.get("border", True)), reach=_reach(d.get("reach", 1.9)),
                   lines=[ProtectLine.from_dict(x) for x in d.get("lines", [])],
                   shapes=[ProtectShape.from_dict(x) for x in d.get("shapes", [])])


@dataclass
class TrackingSettings:
    models: List[str] = field(default_factory=lambda: ["pose_rtmpose_x", "pose_rtmw_x", "pose_rtmpose_m"])
    face: bool = True
    point_tracker: bool = True
    second_pass: bool = True
    smoothing: float = 0.35          # 0..1 extra smoothing on top of the bidirectional filter
    conf_threshold: float = 0.55     # frames under this are marked for review
    adapt: bool = True               # clip adaptation (learned from corrections)
    refiner: bool = True             # use the active learned refiner model
    flip_tta: bool = False           # also estimate on mirrored crops (more stable, about 2x slower)
    detail: bool = True              # dedicated face (106 points) and hand (21 points) models
    reid: bool = True                # person re-identification (body + face) to keep members apart
    members: int = 0                 # people in the shot for the whole-shot identity solve (0 = count them)
    video_track: str = "off"         # SAM 2 video tracking of identities (experimental): auto (with a GPU) | on | off
    seg: str = "sam2"                # person matte: sam2 (SAM 2 per person masks + RVM) | rvm

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        t = cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
        t.adapt = True          # learning is always on (no switch in the app any more)
        t.refiner = True
        # accuracy is always decided by the app (1.14: no switches): every cut gets the automatic settings
        auto = cls()
        for k in ("models", "face", "point_tracker", "second_pass", "smoothing", "flip_tta", "detail", "reid",
                  "video_track", "seg"):
            setattr(t, k, getattr(auto, k))
        return t


@dataclass
class OutputSettings:
    format: str = "prores_4444_xq"
    out_dir: str = ""
    name_pattern: str = "{cut}_DIAP"            # (1.26; "{cut}_reshape" before, from when the app reshaped bodies only)
    include_audio: bool = True
    matte_person: bool = False
    matte_warp: bool = False
    output_cs: str = ""              # empty = keep source encoding (recommended)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class QCIssue:
    kind: str
    start: int
    end: int
    severity: str          # warn | error
    message: str
    person: int = -1
    status: str = "open"   # open | ignored | fixed
    value: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class CutStats:
    active_seconds: float = 0.0
    corrections: int = 0
    analysis_runs: int = 0
    analysis_seconds: float = 0.0    # how long the last whole analysis of the cut took (for the review report)
    auto_ok_rate: float = -1.0       # share of person-frames tracked confidently before any correction
    renders: int = 0
    render_seconds: float = 0.0
    last_output: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# --------------------------------------------------------------------------- cut & project

@dataclass
class Cut:
    id: str
    name: str
    source: str
    info: MediaInfo
    in_frame: int = 0
    out_frame: int = -1              # exclusive; -1 = end
    input_cs: str = ""               # empty = project default
    cs_auto: str = ""                # how the app chose input_cs by itself ('' = chosen by hand / not looked at)
    cs_analysed: Optional[str] = None  # input_cs when the cut was last analysed (None: before 1.16, unknown)
    models_used: dict = field(default_factory=dict)   # model key -> fingerprint of the files the last analysis ran with
    model_versions: dict = field(default_factory=dict)   # model key -> package version it ran with (models installed
                                                         # as packages only; written only when there are any)
    lut_path: str = ""
    mode: str = "warp"               # warp | separate
    persons: List[Person] = field(default_factory=list)
    params: Dict[int, PartParams] = field(default_factory=dict)
    joint_keys: Dict[int, Dict[int, Dict[int, List[float]]]] = field(default_factory=dict)  # pid -> joint -> frame -> [x,y]
    locked: Dict[int, List[List[int]]] = field(default_factory=dict)       # pid -> [[a,b], ...] inclusive
    confirmed: Dict[int, List[List[int]]] = field(default_factory=dict)    # pid -> [[a,b], ...] inclusive
    hidden: Dict[int, Dict[int, List[List[int]]]] = field(default_factory=dict)  # pid -> joint -> [[a,b]] forced hidden
    protect: ProtectSettings = field(default_factory=ProtectSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    qc: List[QCIssue] = field(default_factory=list)
    stats: CutStats = field(default_factory=CutStats)
    analyzed: bool = False
    analysis_version: int = 0
    created: float = field(default_factory=time.time)
    benchmark_id: str = ""
    learn: bool = True               # include this cut's corrections in the learning dataset
    pending_copy: dict = field(default_factory=dict)   # settings copied before analysis; applied afterwards
    removed: List[int] = field(default_factory=list)     # person ids the user deleted (not brought back by re-analysis)
    edit_points: List[int] = field(default_factory=list)  # source frames where a new shot starts (from an editing XML)
    retouch_ref: str = ""            # the retoucher's own result of this plate (learned from after analysis)
    style_learned: dict = field(default_factory=dict)   # last style sample learned from retouch_ref
    tl: dict = field(default_factory=dict)       # place in an edited timeline: index, kind, rec_in, rec_out, name, note
    status: dict = field(default_factory=dict)   # beauty status: auto, reason, user, reviewed, metrics, work
    version: int = 1                              # work version (renders are named _v<version>)
    versions: dict = field(default_factory=dict)  # str(version) -> snapshot of the work at that version

    @property
    def n_frames(self) -> int:
        return self.info.frames

    @property
    def work_range(self) -> Tuple[int, int]:
        end = self.out_frame if self.out_frame > 0 else self.info.frames
        return max(0, self.in_frame), min(self.info.frames, end)

    def person(self, pid: int) -> Optional[Person]:
        return next((p for p in self.persons if p.id == pid), None)

    def targets(self) -> List[Person]:
        return [p for p in self.persons if p.role == "target"]

    def params_for(self, pid: int) -> PartParams:
        if pid not in self.params:
            self.params[pid] = PartParams()
        return self.params[pid]

    def is_locked(self, pid: int, frame: int) -> bool:
        return any(a <= frame <= b for a, b in self.locked.get(pid, []))

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "name": self.name, "source": self.source, "info": self.info.to_dict(),
            "in_frame": self.in_frame, "out_frame": self.out_frame, "input_cs": self.input_cs, "cs_auto": self.cs_auto, "cs_analysed": self.cs_analysed, "models_used": self.models_used, "lut_path": self.lut_path,
            "mode": self.mode, "persons": [p.to_dict() for p in self.persons],
            "params": {str(k): v.to_dict() for k, v in self.params.items()},
            "joint_keys": {str(pid): {str(j): {str(f): xy for f, xy in fr.items()} for j, fr in jd.items()}
                           for pid, jd in self.joint_keys.items()},
            "locked": {str(k): v for k, v in self.locked.items()},
            "confirmed": {str(k): v for k, v in self.confirmed.items()},
            "hidden": {str(pid): {str(j): r for j, r in jd.items()} for pid, jd in self.hidden.items()},
            "protect": self.protect.to_dict(), "tracking": self.tracking.to_dict(), "output": self.output.to_dict(),
            "qc": [q.to_dict() for q in self.qc], "stats": self.stats.to_dict(), "analyzed": self.analyzed,
            "analysis_version": self.analysis_version, "created": self.created, "benchmark_id": self.benchmark_id,
            "learn": self.learn, "pending_copy": self.pending_copy, "removed": list(self.removed),
            "edit_points": list(self.edit_points), "retouch_ref": self.retouch_ref,
            "style_learned": self.style_learned, "tl": self.tl, "status": self.status,
            "version": self.version, "versions": self.versions,
        }
        if self.model_versions:
            d["model_versions"] = dict(self.model_versions)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Cut":
        c = cls(id=d["id"], name=d["name"], source=d["source"], info=MediaInfo.from_dict(d["info"]))
        c.in_frame = int(d.get("in_frame", 0))
        c.out_frame = int(d.get("out_frame", -1))
        c.input_cs = d.get("input_cs", "")
        c.cs_auto = d.get("cs_auto", "")
        c.cs_analysed = d.get("cs_analysed")
        c.models_used = dict(d.get("models_used") or {})
        c.model_versions = {str(k): str(v) for k, v in (d.get("model_versions") or {}).items()}
        c.lut_path = d.get("lut_path", "")
        c.mode = d.get("mode", "warp")
        c.persons = [Person.from_dict(p) for p in d.get("persons", [])]
        c.params = {int(k): PartParams.from_dict(v) for k, v in (d.get("params") or {}).items()}
        c.joint_keys = {int(pid): {int(j): {int(f): list(xy) for f, xy in fr.items()} for j, fr in jd.items()}
                        for pid, jd in (d.get("joint_keys") or {}).items()}
        c.locked = {int(k): [list(r) for r in v] for k, v in (d.get("locked") or {}).items()}
        c.confirmed = {int(k): [list(r) for r in v] for k, v in (d.get("confirmed") or {}).items()}
        c.hidden = {int(pid): {int(j): [list(r) for r in rs] for j, rs in jd.items()}
                    for pid, jd in (d.get("hidden") or {}).items()}
        c.protect = ProtectSettings.from_dict(d.get("protect"))
        c.tracking = TrackingSettings.from_dict(d.get("tracking"))
        c.output = OutputSettings.from_dict(d.get("output"))
        c.qc = [QCIssue.from_dict(q) for q in d.get("qc", [])]
        c.stats = CutStats.from_dict(d.get("stats"))
        c.analyzed = bool(d.get("analyzed", False))
        c.analysis_version = int(d.get("analysis_version", 0))
        c.created = float(d.get("created", time.time()))
        c.benchmark_id = d.get("benchmark_id", "")
        c.learn = bool(d.get("learn", True))
        c.pending_copy = dict(d.get("pending_copy") or {})
        c.removed = [int(x) for x in (d.get("removed") or [])]
        c.edit_points = [int(x) for x in (d.get("edit_points") or [])]
        c.retouch_ref = str(d.get("retouch_ref") or "")
        c.style_learned = dict(d.get("style_learned") or {})
        c.tl = dict(d.get("tl") or {})
        c.status = dict(d.get("status") or {})
        c.version = max(1, int(d.get("version", 1)))
        c.versions = dict(d.get("versions") or {})
        return c


_DIRS: Dict[str, float] = {}       # a project's folders this process made or saw there, and when
DIR_MEMO_S = 20.0                  # for so long a folder is taken to be there still (0: asked every time, as before 1.27)


def _made(d: Path) -> Path:
    """d, made when it is not there. With the project on a network drive every mkdir of a folder that is there is a
    round trip (two: pathlib asks whether it is a folder when the making fails), and opening one cut asked 46 times
    (LNGSHOT on Y:, 11/26: seconds when the drive is busy). A folder made or seen is taken to be there for
    DIR_MEMO_S; one removed from outside meanwhile is made again after that (forget_dirs: at once)."""
    k = str(d)
    now = time.monotonic()
    t = _DIRS.get(k)
    if t is None or now - t >= DIR_MEMO_S:
        d.mkdir(parents=True, exist_ok=True)
        if len(_DIRS) > 4096:
            _DIRS.clear()
        _DIRS[k] = now
    return d


def forget_dirs() -> None:
    """After folders of a project were removed or moved: asked again."""
    _DIRS.clear()


def _complete(meta: Path) -> bool:
    try:
        return bool(json.loads(meta.read_text(encoding="utf-8")).get("complete"))
    except Exception:
        return False


@dataclass
class Project:
    name: str
    path: str                         # folder ending with .dpprj (or .contour from before 1.13)
    cuts: List[Cut] = field(default_factory=list)
    input_cs: str = ""                # default input colourspace for new cuts ('' = passthrough)
    created: float = field(default_factory=time.time)
    modified: float = field(default_factory=time.time)
    # Learning from this project's footage (the crops and samples collected from its cuts, and the automatic
    # learning that starts from them): only when it was switched on by hand for this project (set_learn). A
    # project that never chose (made before 1.24, whose file has no 'learn' or the old default true) is off.
    learn: bool = False
    learn_set: bool = False           # 'learn' was chosen by hand under that rule (instruction 2026-11-20 §4-4)
    learn_before: bool = False        # (not saved) the file said true under the old rule: said once on opening
    client: str = ""
    timeline: dict = field(default_factory=dict)   # an edited timeline the cuts come from (see core/timeline.py)
    members: dict = field(default_factory=dict)    # people across cuts: id -> {name, color, user, links}
    member_cannot: list = field(default_factory=list)   # [[cut, pid], [cut, pid]] the user said are different people

    @property
    def folder(self) -> Path:
        return Path(self.path)

    def cut_dir(self, cut: Cut) -> Path:
        return _made(self.folder / "cuts" / cut.id)

    def cache_dir(self, cut: Cut) -> Path:
        """Analysis results that only an analysis makes again (mattes, person labels, face layer): kept with the
        project. Proxies are in proxy_dir (this computer's cache)."""
        return _made(self.cut_dir(cut) / "cache")

    # ------------------------------------------------------------------ proxies (this computer's cache, 1.16)
    def cache_key(self) -> str:
        """The project's folder name in the cache: its creation time (stays when the project is renamed or moved)."""
        return f"{int(round(float(self.created) * 1000))}"

    def local_cache(self) -> Path:
        from .settings import settings
        root = settings().cache_path
        key = self.cache_key()
        d = next((p for p in root.glob(key + "_*") if p.is_dir()), None) if root.exists() else None
        if d is None:
            safe = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", self.name).strip("_")[:40] or "project"
            d = root / f"{key}_{safe}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def proxy_dir(self, cut: Cut) -> Path:
        """Where the cut's proxy is: this computer's cache. A proxy an older version made in the project folder is
        used where it is until it is cleared (설정 > 캐시 정리), then it is made again here."""
        old = self.cut_dir(cut) / "cache"
        if _complete(old / "proxy_meta.json"):
            return old
        sh = old / "proxy_shared.json"
        if sh.exists():
            try:
                if Path(json.loads(sh.read_text(encoding="utf-8"))["path"]).exists():
                    return old
            except Exception:
                pass
            try:
                sh.unlink()                 # a link to a bake proxy that is gone
            except OSError:
                pass
        d = self.local_cache() / "cuts" / cut.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def bake_cache_dir(self) -> Path:
        old = self.folder / "bake_cache"
        if _complete(old / "proxy_meta.json"):
            return old
        d = self.local_cache() / "bake_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cut(self, cid: str) -> Optional[Cut]:
        return next((c for c in self.cuts if c.id == cid), None)

    def add_cut(self, info: MediaInfo, name: str = "") -> Cut:
        cid = uuid.uuid4().hex[:10]
        nm = safe_name(name or Path(info.path).stem)
        existing = {c.name for c in self.cuts}
        base, n = nm, 2
        while nm in existing:
            nm = f"{base}_{n}"
            n += 1
        c = Cut(id=cid, name=nm, source=info.path, info=info, input_cs=self.input_cs)
        self.cuts.append(c)
        return c

    def set_learn(self, on: bool) -> None:
        """The person's own choice for this project (the project menu)."""
        self.learn, self.learn_set = bool(on), True

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"format": 1, "name": self.name, "cuts": [c.to_dict() for c in self.cuts], "input_cs": self.input_cs,
                "created": self.created, "modified": self.modified, "learn": bool(self.learn and self.learn_set),
                "learn_set": bool(self.learn_set), "client": self.client, "timeline": self.timeline,
                "members": self.members, "member_cannot": self.member_cannot}

    def save(self) -> None:
        self.modified = time.time()
        self.folder.mkdir(parents=True, exist_ok=True)
        f = self.folder / "project.json"
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        if f.exists():
            bak = self.folder / "project.backup.json"
            try:
                bak.write_bytes(f.read_bytes())
            except OSError:
                pass
        tmp.replace(f)

    @classmethod
    def load(cls, path: str) -> "Project":
        p = Path(path)
        if p.is_file():
            p = p.parent
        d = json.loads((p / "project.json").read_text(encoding="utf-8"))
        prj = cls(name=d.get("name", p.stem), path=str(p))
        prj.cuts = [Cut.from_dict(c) for c in d.get("cuts", [])]
        prj.input_cs = d.get("input_cs", "")
        prj.created = float(d.get("created", time.time()))
        prj.modified = float(d.get("modified", time.time()))
        prj.learn_set = bool(d.get("learn_set", False))
        prj.learn = bool(d.get("learn", False)) and prj.learn_set
        prj.learn_before = not prj.learn_set and bool(d.get("learn", True))
        prj.client = d.get("client", "")
        prj.timeline = dict(d.get("timeline") or {})
        prj.members = dict(d.get("members") or {})
        prj.member_cannot = list(d.get("member_cannot") or [])
        return prj

    def timeline_cuts(self) -> List[Cut]:
        """Cuts of the edited timeline, in edit order."""
        return sorted([c for c in self.cuts if c.tl], key=lambda c: c.tl.get("index", 0))

    @classmethod
    def create(cls, parent_dir: str, name: str) -> "Project":
        nm = safe_name(name)
        folder = Path(parent_dir) / f"{nm}{PROJECT_EXT}"
        n = 2
        while folder.exists():
            folder = Path(parent_dir) / f"{nm}_{n}{PROJECT_EXT}"
            n += 1
        prj = cls(name=nm, path=str(folder))
        prj.save()
        return prj
