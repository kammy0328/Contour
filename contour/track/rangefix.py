"""One person's frames made again where the analysis went wrong, compared, then applied (instruction 2026-11-19
§3). The first item is the face and neck masks (the same path as '얼굴·목 마스크 다시 계산', for one person and a
range instead of the whole cut):

- candidate(): the face layer of the person over the range, made again with the face model of the settings from
  the current tracking. The parser reads a few frames before and after (the masks are steadied over neighbouring
  frames) but only the range is kept. Nothing is written; the cut's own tracking is not touched (the landmark
  alignment the parser finds is kept in the candidate);
- compare(): per frame how far the new skin and neck masks are from the ones in use (same frame, same picture),
  for the comparison view;
- apply(): the range of that person replaced in the face layer file (swapped in once written) and the aligned
  face points written into the tracking of those frames. Every other person, every frame outside the range, the
  joints, people, retouch values and keys stay as they are. Cancel or failure: nothing is written.

The preview and the render read the same face layer file (facelayer.load), so what is applied is what renders."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import facelayer as FL
from . import faceparse as FP
from .modelpkg import cut_models as _cut_models

ITEMS = (("face", "얼굴·목"), ("track", "추적"), ("body", "몸 윤곽"))
READY = {"face"}
MARGIN = max(FL.T_R, FL.COLOR_R, FL.REL_R)      # frames read on each side for the steadying (not applied)

PER_PERSON = ("face", "present", "face_vis", "scale", "hand", "final", "vis")


class _StoreSlice:
    """Frames s0..s1 of a track store, for the face layer build. The face points are a copy (the build moves them
    onto the face it finds: the live tracking must not move before the user applies)."""

    def __init__(self, st, s0: int, s1: int):
        self._st, self.s0, self.s1 = st, int(s0), int(s1)
        self.n = self.s1 - self.s0
        self.pids = st.pids
        self.has_face = getattr(st, "has_face", False)
        for k in PER_PERSON:
            v = getattr(st, k, None)
            if v is not None:
                setattr(self, k, v[:, self.s0:self.s1].copy() if k == "face" else v[:, self.s0:self.s1])
        self.jump = np.asarray(st.jump[self.s0:self.s1]) if hasattr(st, "jump") else np.zeros(self.n, bool)

    def connected(self, f0: int, f1: int) -> bool:
        lo, hi = min(f0, f1), max(f0, f1)
        return not self.jump[lo:hi].any()

    def index(self, pid: int) -> int:
        return self._st.index(pid)


class _ProxySlice:
    def __init__(self, px, s0: int, n: int):
        self._px, self.s0, self.n = px, int(s0), int(n)
        self.sx, self.sy, self.w, self.h = px.sx, px.sy, px.w, px.h

    def frame(self, t):
        return self._px.frame(int(t) + self.s0)


class _TrackerSlice:
    def __init__(self, tracker, s0: int, s1: int):
        self._tr = tracker
        self.s0 = int(s0)
        self.cut, self.project = tracker.cut, tracker.project
        self.store = _StoreSlice(tracker.store, s0, s1)
        self.proxy = _ProxySlice(tracker.proxy, s0, s1 - s0)

    def _virtual_face(self, p, t):
        return self._tr._virtual_face(p, int(t) + self.s0)


@_cut_models
def candidate(tracker, pid: int, a: int, b: int, progress: Optional[Callable[[float, str], None]] = None,
              cancel: Optional[Callable[[], bool]] = None) -> Optional[dict]:
    """The face layer of person pid over frames [a, b), made again (nothing written). None when there is no face
    to parse there (or the face model is not available)."""
    st = tracker.store
    if st is None or not st.has(pid) or not getattr(st, "has_face", False):
        return None
    a, b = max(0, int(a)), min(st.n, int(b))
    if b <= a:
        return None
    s0, s1 = max(0, a - MARGIN), min(st.n, b + MARGIN)
    view = _TrackerSlice(tracker, s0, s1)
    p = st.index(pid)
    r = FL.build(view, progress, cancel, persons={p}, keep=False)
    if r is None:
        return None
    i0, i1 = a - s0, b - s0
    out = dict(pid=int(pid), a=a, b=b, model=r["model"], seconds=round(float(r["seconds"]), 2))
    for k in ("M", "skin", "neck", "tex", "rel", "col", "have", "face"):
        out[k] = np.array(r[k][p, i0:i1])
    out["moved"] = int((np.abs(np.nan_to_num(out["face"][..., :2] - st.face[p, a:b, :, :2])).sum((1, 2)) > 1e-3).sum())
    return out


def _image_maps(layer_arrays, k: int, shape):
    """skin, neck on the picture for frame index k of arrays (M, skin, neck, have)."""
    if not layer_arrays["have"][k]:
        return None, None
    M = layer_arrays["M"][k]
    return (FP.to_image(layer_arrays["skin"][k], M, shape) > 0.5,
            FP.to_image(layer_arrays["neck"][k], M, shape) > 0.5)


def current(tracker, pid: int, a: int, b: int) -> Optional[dict]:
    """The face layer in use for person pid over [a, b) (same keys as a candidate), None without one."""
    fl = FL.load(tracker)
    if fl is None:
        return None
    p = fl.index(pid)
    if p is None:
        return None
    b = min(b, fl.have.shape[1])
    return dict(M=fl.M[p, a:b], skin=fl.skin[p, a:b], neck=fl.neck[p, a:b], have=fl.have[p, a:b],
                rel=fl.rel[p, a:b])


def compare(tracker, cand: dict, shape=None) -> dict:
    """Per frame of the range: overlap (IoU) of the skin and of the neck between the masks in use and the new ones."""
    px = tracker.proxy
    shape = shape or (max(2, px.h // 2), max(2, px.w // 2))
    a, b, pid = cand["a"], cand["b"], cand["pid"]
    cur = current(tracker, pid, a, b)
    rows = []
    for k in range(b - a):
        t = a + k
        ns, nn = _image_maps(cand, k, shape)
        os_, on = _image_maps(cur, k, shape) if cur is not None and k < len(cur["have"]) else (None, None)

        def iou(x, y):
            if x is None and y is None:
                return None
            if x is None or y is None:
                return 0.0
            u = (x | y).sum()
            return float((x & y).sum()) / u if u else 1.0
        rows.append(dict(t=t, old=os_ is not None, new=ns is not None, skin=iou(os_, ns), neck=iou(on, nn),
                         rel_old=float(cur["rel"][k]) if cur is not None and k < len(cur["rel"]) else None,
                         rel_new=float(cand["rel"][k])))
    vals = [r["skin"] for r in rows if r["skin"] is not None]
    changed = sum(1 for r in rows if r["skin"] is not None and r["skin"] < 0.95 or r["old"] != r["new"])
    return dict(rows=rows, mean_skin=round(float(np.mean(vals)), 3) if vals else None, changed=changed,
                frames=len(rows), moved=cand.get("moved", 0))


def effects(cut, cand: dict) -> str:
    """What applying changes and what stays (shown before the user applies)."""
    a, b, pid = cand["a"], cand["b"], cand["pid"]
    p = cut.person(pid) if cut is not None else None
    who = p.name if p is not None else f"인물 {pid}"
    moved = cand.get("moved", 0)
    t = (f"적용하면 바뀌는 것: {who}의 {a}~{b - 1} 프레임 얼굴·목 마스크와 피부색 기준"
         + (f", 얼굴 점 위치 보정 {moved}프레임 (얼굴 모양 보정이 따라감)" if moved else "") + ".\n"
         "그대로인 것: 관절·몸 윤곽·인물 매트·의상 마스크, 다른 인물, 구간 밖 프레임, 보정값과 키프레임.")
    return t


def apply(tracker, cand: dict) -> None:
    """The person's range replaced in the face layer file (written beside it, swapped in once complete) and the
    aligned face points written into the tracking of those frames."""
    import time
    st = tracker.store
    pid, a, b = cand["pid"], cand["a"], cand["b"]
    path = FL.path_for(tracker)
    P, N, K = len(st.pids), st.n, FP.KEEP
    if path.exists():
        with np.load(path) as d:
            arr = {k: d[k].copy() for k in d.files}
        lay_pids = [int(x) for x in arr["pids"]]
    else:
        arr = dict(version=np.array(FL.VERSION), pids=np.array(st.pids),
                   M=np.full((P, N, 2, 3), np.nan, np.float32), skin=np.zeros((P, N, K, K), np.float16),
                   neck=np.zeros((P, N, K, K), np.float16), tex=np.ones((P, N, K, K), np.float16),
                   rel=np.zeros((P, N), np.float32), col=np.full((P, N, 3), np.nan, np.float32),
                   have=np.zeros((P, N), bool), model=np.array(cand["model"]), made=np.array(time.time()),
                   seconds=np.array(0.0))
        lay_pids = [int(x) for x in st.pids]
    if pid not in lay_pids:
        # a person the layer was made without (added since): a row of their own
        lay_pids.append(pid)
        for k, fill in (("M", np.nan), ("skin", 0), ("neck", 0), ("tex", 1), ("rel", 0), ("col", np.nan),
                        ("have", False)):
            if k in arr:
                row = np.full((1,) + arr[k].shape[1:], fill, arr[k].dtype)
                arr[k] = np.concatenate([arr[k], row], 0)
        arr["pids"] = np.array(lay_pids)
    p = lay_pids.index(pid)
    b = min(b, arr["have"].shape[1])
    n = b - a
    for k in ("M", "skin", "neck", "tex", "rel", "col", "have"):
        if k in arr:
            arr[k][p, a:b] = cand[k][:n].astype(arr[k].dtype)
    if str(arr.get("model", "")) != cand["model"] and cand["model"]:
        # one model made the whole layer before: the range made with another is told apart
        old = str(arr.get("model", ""))
        arr["model"] = np.array(old if cand["model"] in old.split("+") else (old + "+" + cand["model"]).strip("+"))
    arr["ranges"] = np.concatenate([arr["ranges"].reshape(-1, 4) if "ranges" in arr else np.zeros((0, 4)),
                                    np.array([[pid, a, b, time.time()]])], 0)
    tmp = path.with_name(path.stem + ".tmp.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp, **arr)
    FL._swap_in(tmp, path, tracker)
    sp = st.index(pid)
    st.face[sp, a:b] = cand["face"][:n]
    tracker.__dict__.pop("_steady_lru", None)
    tracker.save()
