"""The person matte made again where an edit changed the tracking (instruction 2026-11-19 §4).

The analysis makes the matte once. After that, a corrected joint (resolve_local), people merged, split or swapped,
or a longer work range change the tracking that prompted it, and the matte (alpha and who owns which pixel) went
stale until the whole cut was matted again by hand. Here only the frames an edit changed are matted again, in the
background, with the SAM models kept loaded between runs:
- compute(): the new matte and owners for the changed ranges, in memory (nothing written);
- commit(): written into the cut's matte and owner files at once, then the body widths and outlines of everyone in
  those frames measured again (an owner change moves a neighbour's edge too), and recorded in seg_meta.json.
The caller (ui/matte_refresh.py) decides when, drops results an edit made since has outdated, and keeps the old
matte on cancel or failure."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MARGIN = 3          # frames before and after a changed range matted again with it (the edges of a re-solve blend)
MOVE = 0.01         # a prompt point that moved more than this share of the frame height changes the matte
                    # (0.004 marked 79 frames for one wrist moved on a real cut: the re-solve's few-pixel ripples)


# ---------------------------------------------------------------------- which frames an edit changed
def prompt_state(tracker) -> Optional[dict]:
    """What the matte of every frame is prompted with (segment.prompts: who is there, their boxes and visible
    joints), copied before an edit to compare with after."""
    st = getattr(tracker, "store", None)
    if st is None or not hasattr(st, "final"):
        return None
    vis = getattr(st, "vis", None)
    return dict(final=st.final[:, :, :, :2].copy(), vis=None if vis is None else vis.copy(),
                present=st.present.copy(), boxes=st.boxes.copy())


def _people(s: dict, t: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    from .store import VIS_VISIBLE
    out = []
    for p in np.nonzero(s["present"][:, t])[0]:
        kp = s["final"][p, t].astype(np.float64)
        if s["vis"] is not None:
            kp = np.where((s["vis"][p, t] == VIS_VISIBLE)[:, None], kp, np.nan)
        out.append((kp, s["boxes"][p, t].astype(np.float64)))
    return out


def _differs(B, A, thr: float) -> bool:
    """Two frames' prompt sets differ: people paired by where they are (so a swap or a merge of two track ids,
    which moves nobody, is no change)."""
    if len(A) != len(B):
        return True
    left = list(range(len(B)))
    for kp, box in A:
        best, cost = None, np.inf
        for i in left:
            d = np.abs(kp - B[i][0])
            db = np.abs(box - B[i][1])
            c = float(np.nanmean(d)) if np.isfinite(d).any() else float(np.nanmean(db)) if np.isfinite(db).any() \
                else 1e9
            if c < cost:
                best, cost = i, c
        if best is None:
            return True
        bk, bb = B[best]
        left.remove(best)
        if (np.isfinite(kp[:, 0]) != np.isfinite(bk[:, 0])).any():
            return True                         # a joint became visible or hidden: points added or taken away
        d = np.abs(kp - bk)
        if np.isfinite(d).any() and np.nanmax(d) > thr:
            return True
        db = np.abs(box - bb)
        if np.isfinite(box).all() != np.isfinite(bb).all() or (np.isfinite(db).any() and np.nanmax(db) > 2 * thr):
            return True
    return False


def changed(tracker, before: Optional[dict], rng: Optional[Sequence[int]] = None) -> List[List[int]]:
    """[a, b) ranges, inside rng (default the work range), where the matte's prompts are no longer what the matte
    of that frame was made from (matte_rev.load_prompts, recorded with every matte revision since 1.22.3). Small
    moves add up: the third 10 px move of a wrist at 4K is 30 px from what the matte was made of, though each one
    was under the limit against the edit before (which is all 1.22.2 looked at).
    before: the prompts just before the edit, used for frames nothing is recorded for (a cut matted by an older
    version, a frame still marked). Frames an edit left alone are not matted again."""
    st = getattr(tracker, "store", None)
    if st is None:
        return []
    base = None
    try:
        from . import matte_rev as MRV
        base = MRV.load_prompts(tracker)
    except Exception:
        base = None
    if before is None and base is None:
        return []
    now = prompt_state(tracker)
    a0, b0 = tracker.cut.work_range
    a, b = (a0, b0) if rng is None else (max(a0, int(rng[0])), min(b0, int(rng[1])))
    thr = MOVE * float(getattr(tracker.cut.info, "height", 0) or 1080)
    if before is not None and before["final"].shape[1] != now["final"].shape[1]:
        return [[a, b]] if b > a else []
    out: List[List[int]] = []
    for t in range(a, b):
        if base is not None and base["known"][t]:
            ref = _people(base, t)
        elif before is not None:
            ref = _people(before, t)
        else:
            continue
        if _differs(ref, _people(now, t), thr):
            if out and out[-1][1] == t:
                out[-1][1] = t + 1
            else:
                out.append([t, t + 1])
    return merge(out)


def stale(tracker) -> List[List[int]]:
    """The frames of the work range whose matte is not made from the present tracking, told from what is recorded
    with the matte alone (no edit needed to say so): for a cut being opened, after an edit whose marks were
    never saved or a refresh taken back. [] when nothing is recorded."""
    try:
        from . import matte_rev as MRV
        if getattr(tracker, "store", None) is None or not tracker.has_seg() or MRV.load_prompts(tracker) is None:
            return []
    except Exception:
        return []
    return changed(tracker, None)


def merge(ranges: Sequence[Sequence[int]], gap: int = 6) -> List[List[int]]:
    """[a, b) ranges joined where they overlap or nearly touch (two edits close together: one run)."""
    rs = sorted([int(a), int(b)] for a, b in ranges if int(b) > int(a))
    out: List[List[int]] = []
    for a, b in rs:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def subtract(ranges: Sequence[Sequence[int]], done: Sequence[Sequence[int]]) -> List[List[int]]:
    """What is left of ranges once done is taken out."""
    left = [list(r) for r in merge(ranges, 0)]
    for da, db in done:
        nxt = []
        for a, b in left:
            if db <= a or da >= b:
                nxt.append([a, b])
                continue
            if a < da:
                nxt.append([a, da])
            if db < b:
                nxt.append([db, b])
        left = nxt
    return [r for r in left if r[1] > r[0]]


class RangeStore:
    """Stands in for a MatteStore while segment.run writes (open_rw()[t] = frame): keeps the frames in memory."""

    def __init__(self, like):
        self.w, self.h, self.n = like.w, like.h, like.n
        self.frames: Dict[int, np.ndarray] = {}

    def open_rw(self):
        return self

    def exists(self) -> bool:
        return True

    def __setitem__(self, t, v):
        self.frames[int(t)] = np.asarray(v, np.uint8).copy()

    def flush(self):
        pass

    def close(self):
        pass


def clip_ranges(tracker, ranges) -> List[List[int]]:
    a0, b0 = tracker.cut.work_range
    out = []
    for a, b in merge(ranges):
        a, b = max(a0, a - MARGIN), min(b0, b + MARGIN)
        if b > a:
            out.append([a, b])
    return out


def compute(tracker, ranges, cancel: Optional[Callable[[], bool]] = None,
            progress: Optional[Callable[[float, str], None]] = None, models: Optional[dict] = None) -> Optional[dict]:
    """New matte and owners for the ranges, from the current (corrected) tracking. None when the cut has no SAM
    matte to refresh (an RVM matte does not follow the tracking). Raises analyze.Cancelled on cancel."""
    from . import segment as SG
    st = tracker.store
    if st is None or not tracker.has_seg():
        return None
    rs = clip_ranges(tracker, ranges)
    if not rs:
        return dict(matte={}, labels={}, ranges=[], model="")
    models = models if models is not None else {}
    if "matter" not in models:
        try:
            from .infer import Matter
            models["matter"] = Matter()
        except Exception:
            models["matter"] = None
    mbuf, lbuf = RangeStore(tracker.matte), RangeStore(tracker.seglab)
    total = sum(b - a for a, b in rs)
    done = 0
    for a, b in rs:
        sub = None
        if progress:
            sub = (lambda f, m="", d=done, n=b - a: progress((d + f * n) / max(1, total), m))
        SG.run(st, tracker.proxy, mbuf, lbuf, a, b, SG.model_for("max"), True, sub, cancel, models.get("matter"),
               models=models)
        done += b - a
    res = dict(matte=mbuf.frames, labels=lbuf.frames, ranges=rs, model=models.get("used", ""))
    # the rest of the revision, still in memory: widths and outlines measured on the new matte, the clothing of
    # the frames whose owners change (instruction 2026-11-20 §1-2: finished first, put in use together)
    from . import matte_rev as MRV
    res["stage"] = MRV.stage(tracker, mbuf.frames, lbuf.frames, rs, cancel=cancel, models=models)
    return res


def commit(tracker, result: dict) -> List[List[int]]:
    """Puts a computed refresh in use as one revision: alpha, owners, widths, outlines and clothing of its frames
    together (matte_rev.activate). Raises with the revision before still in use when that does not go through."""
    from . import matte_rev as MRV
    return MRV.activate(tracker, result)


def refresh_now(tracker, ranges, progress=None, cancel=None, models: Optional[dict] = None) -> List[List[int]]:
    """compute() and commit() at once, for a job that waits for it (a render that needs the current matte)."""
    res = compute(tracker, ranges, cancel=cancel, progress=progress, models=models)
    if res is None:
        return []
    return commit(tracker, res)


def extension(tracker, old: Sequence[int], new: Sequence[int]) -> Tuple[List[List[int]], List[List[int]]]:
    """Frames a longer work range takes in, as (to matte, not analysed): tracked frames without a matte yet are
    matted from their tracking; frames the analysis never saw (nobody tracked, outside the range the matte was
    made for when that is known) need the analysis itself."""
    st = getattr(tracker, "store", None)
    if st is None:
        return [], []
    a0, b0 = int(old[0]), int(old[1])
    fresh = subtract([[int(new[0]), int(new[1])]], [[a0, b0]])
    made = None
    try:
        made = (tracker.seg_meta() or {}).get("range")
    except Exception:
        pass
    to_matte: List[List[int]] = []
    unseen: List[List[int]] = []

    def add(rs, t):
        if rs and rs[-1][1] == t:
            rs[-1][1] = t + 1
        else:
            rs.append([t, t + 1])
    has = tracker.has_seg()
    for a, b in fresh:
        for t in range(a, min(b, st.n)):
            if st.present[:, t].any():
                if has:
                    m = tracker.matte.get(t)
                    if m is None or not np.any(m):
                        add(to_matte, t)
            elif made is None or not (made[0] <= t < made[1]):
                add(unseen, t)
    return to_matte, unseen
