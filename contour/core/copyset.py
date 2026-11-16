"""Copy reshape settings between cuts, mapping people by appearance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .project import Cut, PartParams


@dataclass
class CopyOptions:
    amounts: bool = True       # part amounts (value at the current frame, no keyframes)
    widths: bool = True        # width multipliers
    mode: bool = True          # warp / separate
    protect: bool = True       # protection settings (not line/shape positions)
    tracking: bool = False     # smoothing / threshold
    output: bool = False       # output format settings
    roles: bool = True         # target/protect roles and names


def _sim(a: List[float], b: List[float]) -> float:
    """Colour similarity 0..1 (mean Bhattacharyya coefficient of the torso and leg histograms)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    from ..track.assoc import app_similarity
    return app_similarity(np.asarray(a, np.float32), np.asarray(b, np.float32))


def person_similarity(a, b) -> float:
    """0..1: how likely two people of different cuts are the same member. Faces decide when both were seen
    (outfits change between scenes of a music video, faces do not); else body re-ID and colours."""
    col = _sim(a.profile, b.profile)
    body = None
    if a.reid and b.reid and len(a.reid) == len(b.reid):
        c = float(np.dot(a.reid, b.reid))
        body = float(np.clip((c - 0.55) / 0.4, 0, 1))
    if a.faces and b.faces:
        # mean over the two small galleries (a single best pair is unreliable among similar faces)
        fs = float((np.asarray(a.faces) @ np.asarray(b.faces).T).mean())
        face = float(np.clip((fs - 0.22) / 0.18, 0, 1))
        return 0.65 * face + 0.35 * (body if body is not None else col)
    if body is not None:
        return 0.6 * body + 0.4 * col
    return col


def _cues(a, b):
    col = _sim(a.profile, b.profile)
    body = float(np.dot(a.reid, b.reid)) if (a.reid and b.reid and len(a.reid) == len(b.reid)) else np.nan
    face = float((np.asarray(a.faces) @ np.asarray(b.faces).T).mean()) if (a.faces and b.faces) else np.nan
    return face, body, col


def _fused(dst_p, src_p) -> np.ndarray:
    """Each cue (face, body re-ID, colours) standardised within each row and column, then averaged.

    How telling a cue is differs from shoot to shoot (identical outfits make body re-ID flat, a dark wide shot
    makes faces noisy); standardising lets a cue that clearly singles out one person count, and a flat one fade.
    On group-dance tests this matched every member between two halves of a shot, where any fixed mix of the
    raw scores missed some."""
    T = np.array([[_cues(d, s) for s in src_p] for d in dst_p], np.float64)      # (nd, ns, 3)
    W = (1.0, 1.0, 0.5)
    out = np.zeros(T.shape[:2])
    for axis in (1, 0):
        S = np.zeros(T.shape[:2])
        N = np.zeros(T.shape[:2])
        for c, w in enumerate(W):
            M = T[:, :, c]
            Mv = M if axis == 1 else M.T
            Sv = S if axis == 1 else S.T
            Nv = N if axis == 1 else N.T
            for i in range(Mv.shape[0]):
                r = Mv[i]
                ok = np.isfinite(r)
                if ok.sum() < 3:
                    continue
                Sv[i, ok] += w * (r[ok] - r[ok].mean()) / (r[ok].std() + 1e-6)
                Nv[i, ok] += w
        out += S / np.maximum(N, 1e-6)
    return out / 2


def _plausible(dst_p, src_p) -> np.ndarray:
    ok = np.zeros((len(dst_p), len(src_p)), bool)
    for i, d in enumerate(dst_p):
        for j, s in enumerate(src_p):
            face, body, col = _cues(d, s)
            ok[i, j] = (np.isfinite(face) and face >= 0.25) or (np.isfinite(body) and body >= 0.6) or col >= 0.75
    return ok


def suggest_mapping(src: Cut, dst: Cut) -> Dict[int, Optional[int]]:
    """dst person id -> src person id (or None)."""
    from ..track.assoc import assign
    if not src.persons or not dst.persons:
        return {p.id: None for p in dst.persons}
    sim = np.array([[person_similarity(d, s) for s in src.persons] for d in dst.persons])
    out: Dict[int, Optional[int]] = {p.id: None for p in dst.persons}
    if len(src.persons) >= 3 and len(dst.persons) >= 2:
        z = _fused(dst.persons, src.persons)
        # relative ranking decides; a pair unlike by every cue (someone not in the source cut) stays unmatched
        cost = np.where(_plausible(dst.persons, src.persons), -z, 1e6)
        for i, j in assign(cost, 1e5):
            out[dst.persons[i].id] = src.persons[j].id
        return out
    for i, j in assign(1 - sim, 0.45):
        out[dst.persons[i].id] = src.persons[j].id
    return out


def apply_copy(src: Cut, dst: Cut, mapping: Dict[int, Optional[int]], frame: int, opt: CopyOptions) -> None:
    for dpid, spid in mapping.items():
        if spid is None:
            continue
        sp = src.person(spid)
        dp = dst.person(dpid)
        if sp is None or dp is None:
            continue
        if opt.roles:
            dp.role = sp.role
            if not sp.name.startswith("인물 "):
                dp.name = sp.name
        if opt.amounts or opt.widths:
            s_params = src.params_for(spid).copy_static(frame)
            d_params = dst.params_for(dpid)
            if opt.amounts:
                d_params.amounts = s_params.amounts
                d_params.strength = s_params.strength
                d_params.skin_detail = s_params.skin_detail
                d_params.skin_scope = s_params.skin_scope
                d_params.grain_size = s_params.grain_size
                d_params.blemish_size = s_params.blemish_size
                d_params.cloth = s_params.cloth
                d_params.feather = s_params.feather
                d_params.sculpt = s_params.sculpt
                d_params.body_sculpt = s_params.body_sculpt
                d_params.body_sides = s_params.body_sides
                d_params.skin_basic = s_params.skin_basic
            if opt.widths:
                d_params.widths = dict(s_params.widths)
    if opt.mode:
        dst.mode = src.mode
    if opt.protect:
        dst.protect.others = src.protect.others
        dst.protect.falloff_pct = src.protect.falloff_pct
        dst.protect.border = src.protect.border
        dst.protect.reach = src.protect.reach
    if opt.tracking:
        dst.tracking.smoothing = src.tracking.smoothing
        dst.tracking.conf_threshold = src.tracking.conf_threshold
        dst.tracking.models = list(src.tracking.models)
    if opt.output:
        o = src.output
        dst.output.format = o.format
        dst.output.out_dir = o.out_dir
        dst.output.name_pattern = o.name_pattern
        dst.output.include_audio = o.include_audio
        dst.output.matte_person = o.matte_person
        dst.output.matte_warp = o.matte_warp
        dst.output.output_cs = o.output_cs
