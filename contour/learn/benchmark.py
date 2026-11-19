"""Fixed benchmark cuts: the yardstick that decides whether a newly trained model is actually better.

A benchmark stores everything the automatic pipeline needs (raw model outputs, camera motion, frames) plus
the user's finished tracks as ground truth, so every model version can be re-scored in seconds without
running the pose networks again.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from ..core.settings import settings
from ..track import skeleton as S
from ..track import solver as SV
from ..track.store import TrackStore

MAX_SIDE = 1280


def bench_dir() -> Path:
    return settings().benchmark_path


def list_benchmarks() -> List[dict]:
    out = []
    for d in sorted(bench_dir().glob("b_*")):
        f = d / "meta.json"
        if f.exists():
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
                m["path"] = str(d)
                out.append(m)
            except Exception:
                pass
    return out


def register(project, cut, tracker, progress=None) -> str:
    st = tracker.store
    proxy = tracker.proxy
    a, b = cut.work_range
    bid = cut.benchmark_id or f"b_{uuid.uuid4().hex[:8]}"
    d = bench_dir() / bid
    if d.exists():
        shutil.rmtree(d)
    (d / "frames").mkdir(parents=True)
    s = min(1.0, MAX_SIDE / max(proxy.w, proxy.h))
    bw, bh = int(proxy.w * s) // 2 * 2, int(proxy.h * s) // 2 * 2
    for i, t in enumerate(range(a, b)):
        img = cv2.resize(np.asarray(proxy.frame(t)), (bw, bh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(d / "frames" / f"{i:06d}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
        if progress and i % 10 == 0:
            progress(i / max(1, b - a), "벤치마크 프레임 저장")
    # ground truth: persons with any correction or confirmation, else the targets
    gt_p = [st.index(pid) for pid in st.pids
            if cut.joint_keys.get(pid) or cut.confirmed.get(pid) or (cut.person(pid) and cut.person(pid).role == "target")]
    if not gt_p:
        gt_p = list(range(len(st.pids)))
    raw = {k: v[gt_p, a:b] for k, v in st.raw.items()}
    # which answers a person gave or confirmed (keyframes, confirmed ranges): the accuracy is read on these; the
    # rest of the solved track is only the automatic answer
    human = np.zeros((len(gt_p), b - a, S.K), bool)
    for i, p in enumerate(gt_p):
        pid = int(st.pids[p])
        for j, fr in (cut.joint_keys.get(pid) or {}).items():
            for f in fr:
                if a <= int(f) < b:
                    human[i, int(f) - a, int(j)] = True
        for r0, r1 in (cut.confirmed.get(pid) or []):
            lo, hi = max(a, int(r0)), min(b - 1, int(r1))
            if hi >= lo:
                human[i, lo - a:hi - a + 1, :] = True
    np.savez_compressed(d / "data.npz", present=st.present[gt_p, a:b], boxes=st.boxes[gt_p, a:b],
                        blur=st.blur[gt_p, a:b], homog=st.homog[a:b], homog_ok=st.homog_ok[a:b],
                        gt=st.final[gt_p, a:b], gt_vis=st.vis[gt_p, a:b], gt_human=human,
                        **{"raw__" + k: v for k, v in raw.items()})
    meta = dict(id=bid, project=project.name, cut=cut.name, frames=b - a, persons=len(gt_p), created=time.time(),
                width=cut.info.width, height=cut.info.height, frame_w=bw, frame_h=bh,
                keys=int(sum(len(fr) for pid in st.pids for fr in cut.joint_keys.get(pid, {}).values())),
                smoothing=cut.tracking.smoothing)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    cut.benchmark_id = bid
    return bid


def remove(bid: str) -> None:
    shutil.rmtree(bench_dir() / bid, ignore_errors=True)


def _load(bid: str):
    d = bench_dir() / bid
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    n = int(meta["frames"])
    P = int(meta["persons"])
    st = TrackStore(n, list(range(P)))
    with np.load(d / "data.npz") as data:
        st.present = data["present"].copy()
        st.boxes = data["boxes"].copy()
        st.blur = data["blur"].copy()
        st.homog = data["homog"].copy()
        st.homog_ok = data["homog_ok"].copy()
        for k in data.files:
            if k.startswith("raw__"):
                st.raw[k[5:]] = data[k].copy()
        gt = data["gt"].copy()
        meta["_gt_vis"] = data["gt_vis"].copy() if "gt_vis" in data.files else None
        meta["_gt_human"] = data["gt_human"].copy() if "gt_human" in data.files else None
    return d, meta, st, gt


METRIC_VERSION = 2       # 2 (1.22.2): pck5 over every answer that can be scored; a joint left out counts as missed
PCK = 0.05
CAP = 0.20               # a missed joint costs this much (of the person's size) in the score; errors are capped there


def score(trk: np.ndarray, gt: np.ndarray, present: np.ndarray, scale: np.ndarray,
          vis: Optional[np.ndarray] = None, only: Optional[np.ndarray] = None) -> dict:
    """Joint accuracy of a track (P,N,K,>=2) against answers (P,N,K,>=2).
    Scored joints: a finite answer, the person present, a usable scale, visible (when visibility is known) and,
    with `only`, one of those joints. pck5 = hits / scored (a joint without a prediction is not a hit, so leaving
    a wrong one out never raises it); missing = scored joints without a prediction; err_pct = mean error of the
    predicted ones (% of the person's size); score_pct = mean of min(error, CAP) with a missed joint at CAP (what a
    better model must lower). Nothing scored: every figure is None (not measured)."""
    sc = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)[..., None]
    can = np.isfinite(gt[..., :2]).all(-1) & present[..., None] & np.isfinite(sc)
    if vis is not None:
        can &= np.asarray(vis) == 1
    if only is not None:
        can &= np.asarray(only, bool)
    n = int(can.sum())
    if n == 0:
        return dict(metric_version=METRIC_VERSION, n_eval=0, n=0, pck5=None, missing=None, err_pct=None,
                    score_pct=None)
    have = np.isfinite(trk[..., :2]).all(-1)
    err = np.linalg.norm(trk[..., :2] - gt[..., :2], axis=-1) / sc
    pred = can & have & np.isfinite(err)
    e = err[pred]
    hits = int((e < PCK).sum())
    capped = np.full(n, CAP, np.float64)
    capped[:e.size] = np.minimum(e, CAP)
    return dict(metric_version=METRIC_VERSION, n_eval=n, n=int(e.size), pck5=hits / n,
                missing=1.0 - e.size / n, err_pct=float(np.mean(e) * 100) if e.size else None,
                score_pct=float(np.mean(capped) * 100))


def evaluate(version: int, progress=None) -> Dict[str, dict]:
    """Score the automatic pipeline (no user input) with refiner `version` (0 = none) on every benchmark."""
    from .refiner import RefinerRuntime, refine_measurements, registry_dir
    rt = None
    if version > 0:
        p = registry_dir() / f"v{version:03d}" / "refiner.onnx"
        if p.exists():
            rt = RefinerRuntime(p, version)
    results = {}
    benches = list_benchmarks()
    for bi, m in enumerate(benches):
        d, meta, st, gt = _load(m["id"])
        frames_dir = d / "frames"
        sx = meta["frame_w"] / meta["width"]
        sy = meta["frame_h"] / meta["height"]
        cache: Dict[int, np.ndarray] = {}

        def rgb(t):
            if t not in cache:
                im = cv2.imread(str(frames_dir / f"{t:06d}.jpg"))
                cache[t] = im[..., ::-1] if im is not None else np.zeros((meta["frame_h"], meta["frame_w"], 3), np.uint8)
                if len(cache) > 40:
                    cache.pop(next(iter(cache)))
            return cache[t]

        def gray(t):
            return cv2.cvtColor(rgb(t), cv2.COLOR_RGB2GRAY)

        SV.fuse_measurements(st)
        extra = None
        if rt is not None:

            class _P:
                pass
            px = _P()
            px.sx, px.sy = sx, sy
            extra = [refine_measurements(st, px, rt, frame_getter=rgb, sx=sx, sy=sy)]
        fs = SV.FrameSource(gray, sx, sy)
        trk, _, _ = SV.solve(st, fs, range(len(st.pids)), (0, st.n), extra=extra,
                             smoothing=float(meta.get("smoothing", 0.35)))
        r = score(trk, gt, st.present, st.scale, meta.get("_gt_vis"))
        hm = meta.get("_gt_human")
        hr = score(trk, gt, st.present, st.scale, meta.get("_gt_vis"), hm) if hm is not None else None
        results[m["id"]] = dict(cut=m["cut"], project=m["project"], **r,
                                visibility_known=meta.get("_gt_vis") is not None,
                                human=({k: hr[k] for k in ("n_eval", "pck5", "missing", "err_pct", "score_pct")}
                                       if hr is not None else None))
        if progress:
            progress((bi + 1) / max(1, len(benches)), f"벤치마크 {m['cut']}")
    log = bench_dir() / "results.jsonl"
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time=time.time(), version=version, metric_version=METRIC_VERSION, results=results),
                           ensure_ascii=False) + "\n")
    return results


def summary(results: Dict[str, dict]) -> Optional[float]:
    """One number to compare models (lower is better). Metric 2: the capped score, where a missed joint costs as
    much as a bad one (so dropping predictions cannot win). Older results (metric 1): their mean error, comparable
    only with each other (metric_of tells which)."""
    if metric_of(results) >= 2:
        vals = [r["score_pct"] for r in results.values() if r.get("score_pct") is not None]
    else:
        vals = [r["err_pct"] for r in results.values() if r.get("err_pct") is not None]
    return float(np.mean(vals)) if vals else None


def metric_of(results: Dict[str, dict]) -> int:
    v = [int(r.get("metric_version", 1)) for r in results.values()]
    return min(v) if v else METRIC_VERSION


def history() -> List[dict]:
    f = bench_dir() / "results.jsonl"
    out = []
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out
