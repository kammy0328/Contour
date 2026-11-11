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
    np.savez_compressed(d / "data.npz", present=st.present[gt_p, a:b], boxes=st.boxes[gt_p, a:b],
                        blur=st.blur[gt_p, a:b], homog=st.homog[a:b], homog_ok=st.homog_ok[a:b],
                        gt=st.final[gt_p, a:b], **{"raw__" + k: v for k, v in raw.items()})
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
    return d, meta, st, gt


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
        ok = np.isfinite(gt[..., :2]).all(-1) & np.isfinite(trk[..., :2]).all(-1) & st.present[..., None]
        sc = np.where(np.isfinite(st.scale), st.scale, np.nan)[..., None]
        err = np.linalg.norm(trk[..., :2] - gt[..., :2], axis=-1) / sc
        e = err[ok & np.isfinite(err)]
        results[m["id"]] = dict(cut=m["cut"], project=m["project"], err_pct=float(np.mean(e) * 100) if e.size else None,
                                pck5=float(np.mean(e < 0.05)) if e.size else None, n=int(e.size))
        if progress:
            progress((bi + 1) / max(1, len(benches)), f"벤치마크 {m['cut']}")
    log = bench_dir() / "results.jsonl"
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time=time.time(), version=version, results=results), ensure_ascii=False) + "\n")
    return results


def summary(results: Dict[str, dict]) -> Optional[float]:
    vals = [r["err_pct"] for r in results.values() if r.get("err_pct") is not None]
    return float(np.mean(vals)) if vals else None


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
