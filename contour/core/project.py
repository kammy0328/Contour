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

PROJECT_EXT = ".contour"

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

    def keyed_parts(self) -> List[str]:
        return [k for k in ALL_KEYS if self.amounts.get(k) is not None and self.amounts[k].keys]

    def to_dict(self) -> dict:
        return {"amounts": {k: c.to_dict() for k, c in self.amounts.items()}, "widths": dict(self.widths),
                "strength": self.strength.to_dict(), "skin_detail": self.skin_detail, "skin_scope": self.skin_scope,
                "grain_size": self.grain_size, "feather": self.feather, "blemish_size": self.blemish_size, "cloth": self.cloth,
                "sculpt": {str(k): list(v) for k, v in self.sculpt.items()},
                "body_sculpt": dict(self.body_sculpt), "body_sides": self.body_sides,
                **({"skin_basic": {"level": self.skin_basic.get("level", "std"),
                                   "remove": list(self.skin_basic.get("remove", []))}} if self.skin_basic else {})}

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
        p.cloth = d.get("cloth", "include") if d.get("cloth") in ("include", "skirt", "all") else "include"
        p.feather = float(d.get("feather", 50.0))
        p.sculpt = {int(k): [float(v[0]), float(v[1])] for k, v in (d.get("sculpt") or {}).items()}
        p.body_sculpt = {str(k): float(v) for k, v in (d.get("body_sculpt") or {}).items()}
        p.body_sides = d.get("body_sides") or ("measured" if p.body_sculpt else "person")
        sb = d.get("skin_basic") or {}
        if isinstance(sb, dict) and sb.get("level") in ("light", "std", "strong"):
            p.skin_basic = {"level": sb["level"], "remove": [str(x) for x in (sb.get("remove") or [])]}
        return p

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
        p.cloth = self.cloth
        p.feather = self.feather
        p.sculpt = {k: list(v) for k, v in self.sculpt.items()}
        p.body_sculpt = dict(self.body_sculpt)
        p.body_sides = self.body_sides
        p.skin_basic = {"level": self.skin_basic["level"], "remove": list(self.skin_basic.get("remove", []))} \
            if self.skin_basic else {}
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
        return t


@dataclass
class OutputSettings:
    format: str = "prores_4444_xq"
    out_dir: str = ""
    name_pattern: str = "{cut}_reshape"
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
        return {
            "id": self.id, "name": self.name, "source": self.source, "info": self.info.to_dict(),
            "in_frame": self.in_frame, "out_frame": self.out_frame, "input_cs": self.input_cs, "lut_path": self.lut_path,
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

    @classmethod
    def from_dict(cls, d: dict) -> "Cut":
        c = cls(id=d["id"], name=d["name"], source=d["source"], info=MediaInfo.from_dict(d["info"]))
        c.in_frame = int(d.get("in_frame", 0))
        c.out_frame = int(d.get("out_frame", -1))
        c.input_cs = d.get("input_cs", "")
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


@dataclass
class Project:
    name: str
    path: str                         # folder ending with .contour
    cuts: List[Cut] = field(default_factory=list)
    input_cs: str = ""                # default input colourspace for new cuts ('' = passthrough)
    created: float = field(default_factory=time.time)
    modified: float = field(default_factory=time.time)
    learn: bool = True                # project-level switch for collecting learning data
    client: str = ""
    timeline: dict = field(default_factory=dict)   # an edited timeline the cuts come from (see core/timeline.py)
    members: dict = field(default_factory=dict)    # people across cuts: id -> {name, color, user, links}
    member_cannot: list = field(default_factory=list)   # [[cut, pid], [cut, pid]] the user said are different people

    @property
    def folder(self) -> Path:
        return Path(self.path)

    def cut_dir(self, cut: Cut) -> Path:
        d = self.folder / "cuts" / cut.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cache_dir(self, cut: Cut) -> Path:
        d = self.cut_dir(cut) / "cache"
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

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"format": 1, "name": self.name, "cuts": [c.to_dict() for c in self.cuts], "input_cs": self.input_cs,
                "created": self.created, "modified": self.modified, "learn": self.learn, "client": self.client, "timeline": self.timeline,
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
        prj.learn = bool(d.get("learn", True))
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
