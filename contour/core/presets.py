"""Named slider presets, kept in Contour_Data/presets so every project (and a newer app version) sees them.

A preset holds the values of the groups it was saved with (body, face, skin, ...), the static extras of those
groups (range multipliers, skin texture, grain size, ...) and, if asked, the face and body point edits.
Applying it changes only those groups: a 'skin only' preset leaves the shape sliders alone."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from .params import GROUPS, PARAMS, default as param_default
from .project import Curve, PartParams
from .settings import settings

VERSION = 1
GROUP_KEYS: Dict[str, List[str]] = {g: [p.key for p in PARAMS if p.group == g] for g, _, _ in GROUPS}
GROUP_TITLE: Dict[str, str] = {g: t for g, t, _ in GROUPS}


# 1.14: no shipped presets any more. The two that earlier versions wrote ("수작업_기준", "레퍼런스_기준") are
# removed once from the presets folder (only when they are still the shipped ones, not a preset the user saved
# under the same name).
BUILTIN: List[dict] = []
_RETIRED = {"수작업_기준": "Contour 1.5.2", "레퍼런스_기준": "Contour 1.8.0"}


def folder() -> Path:
    p = settings().data_path / "presets"
    p.mkdir(parents=True, exist_ok=True)
    mark = p / ".builtin_retired"
    if not mark.exists():
        for name, created in _RETIRED.items():
            f = p / f"{name}.json"
            try:
                if f.exists() and json.loads(f.read_text(encoding="utf-8")).get("created") == created:
                    f.unlink()
            except Exception:
                pass
        try:
            mark.write_text("1", encoding="utf-8")
        except OSError:
            pass
    return p


def _file(name: str) -> Path:
    safe = re.sub(r'[\\/:*?"<>|&]', "_", name.strip()).replace(" ", "_") or "preset"
    return folder() / f"{safe}.json"


def list_presets() -> List[dict]:
    out = []
    for f in sorted(folder().glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["_path"] = str(f)
            out.append(d)
        except Exception:
            continue
    out.sort(key=lambda d: d.get("name", "").lower())
    return out


def active_groups(pp: PartParams) -> List[str]:
    """Groups with at least one value away from its default (or point edits)."""
    out = []
    for g, keys in GROUP_KEYS.items():
        if any(pp._active(k) for k in keys):
            out.append(g)
    return out


def make(pp: PartParams, frame: float, name: str, groups: List[str], points: bool, strength: bool = True) -> dict:
    vals = {}
    for g in groups:
        for k in GROUP_KEYS.get(g, []):
            vals[k] = round(float(pp.amount(k, frame)), 4)
    d = dict(version=VERSION, name=name.strip(), created=time.strftime("%Y-%m-%d %H:%M"), groups=list(groups),
             values=vals, feather=pp.feather)
    if strength:
        d["strength"] = float(pp.strength.value(frame))
    if "body" in groups:
        d["widths"] = dict(pp.widths)
        d["cloth"] = getattr(pp, "cloth", "include")
    if "skin" in groups:
        d["skin"] = dict(detail=pp.skin_detail, scope=pp.skin_scope, grain_size=pp.grain_size,
                         blemish_size=getattr(pp, "blemish_size", 50.0))
        if getattr(pp, "skin_basic", None):
            d["skin"]["basic"] = {"level": pp.skin_basic.get("level", "std"),
                                  "remove": list(pp.skin_basic.get("remove", []))}
    if "bodyskin" in groups:
        d["bodyskin"] = dict(detail=pp.body_detail, blemish_size=pp.body_blemish_size, scope=pp.body_scope)
    if points:
        if pp.sculpt:
            d["sculpt"] = {str(k): list(v) for k, v in pp.sculpt.items()}
        if pp.body_sculpt:
            d["body_sculpt"] = dict(pp.body_sculpt)
            d["body_sides"] = pp.body_sides
    return d


def save(d: dict, overwrite: bool = True) -> Path:
    f = _file(d["name"])
    if f.exists() and not overwrite:
        raise FileExistsError(str(f))
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return f


def delete(name: str) -> None:
    f = _file(name)
    if f.exists():
        f.unlink()


def apply(pp: PartParams, d: dict) -> None:
    """Put a preset's groups onto one person (as static values: keyframes of those sliders are replaced)."""
    vals = d.get("values") or {}
    keep = set(d.get("keep") or [])          # sliders the preset leaves as they are (학습된_스타일: 잡티)
    for g in d.get("groups") or []:
        for k in GROUP_KEYS.get(g, []):
            if k in keep and k not in vals:
                continue
            v = vals.get(k, param_default(k))
            pp.amounts[k] = Curve(base=float(v))
    if "strength" in d:
        pp.strength = Curve(base=float(d["strength"]))
    if "feather" in d:
        pp.feather = float(d["feather"])
    for k, v in (d.get("widths") or {}).items():
        if k in pp.widths:
            pp.widths[k] = float(v)
    if d.get("cloth") in ("include", "skirt", "all"):
        pp.cloth = d["cloth"]
    sk = d.get("skin")
    if sk:
        pp.skin_detail = float(sk.get("detail", pp.skin_detail))
        if sk.get("scope"):
            from .project import norm_scope
            pp.skin_scope = norm_scope(sk["scope"])
        pp.grain_size = float(sk.get("grain_size", pp.grain_size))
        pp.blemish_size = float(sk.get("blemish_size", getattr(pp, "blemish_size", 50.0)))
        b = sk.get("basic") or {}
        pp.skin_basic = {"level": b["level"], "remove": list(b.get("remove") or [])} \
            if isinstance(b, dict) and b.get("level") in ("light", "std", "strong") else {}
    bs = d.get("bodyskin")
    if bs:
        from .project import norm_body_scope
        pp.body_detail = float(bs.get("detail", pp.body_detail))
        pp.body_blemish_size = float(bs.get("blemish_size", pp.body_blemish_size))
        pp.body_scope = norm_body_scope(bs.get("scope", pp.body_scope))
    if "bodyskin" in (d.get("groups") or []):
        pp.body_memo = {}                 # the preset says what is on and off: nothing kept from before to come back to
    if "sculpt" in d:
        pp.sculpt = {int(k): [float(v[0]), float(v[1])] for k, v in d["sculpt"].items()}
    if "body_sculpt" in d:
        pp.body_sculpt = {str(k): float(v) for k, v in d["body_sculpt"].items()}
        pp.body_sides = d.get("body_sides", "measured")


def summary(d: dict) -> str:
    gs = [GROUP_TITLE.get(g, g) for g in d.get("groups") or []]
    extra = []
    if d.get("sculpt") or d.get("body_sculpt"):
        extra.append("점 편집")
    return " · ".join(gs + extra)
