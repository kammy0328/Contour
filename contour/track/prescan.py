"""A quick look for people before the full analysis (instruction 2026-11-19 §5).

The full analysis opens several large models per cut; a cut with nobody in it needs none of them. One light person
detector (YOLOX-m) is opened once for all the cuts of a batch, and each cut is looked at in the picture the analysis
itself would see (the same colour view: a log original is looked at in its display view, not as flat log):
- a short cut every second frame first and, before it is called empty (or when those leave it open), every frame;
  a long one at its start, middle and end and at even steps, then about every half second (or when the first
  frames leave it open), and before it is called empty every frame of it too; two frames with someone plainly
  there end the look at once;
- three answers: someone is there (analyse), nobody is there (skip, with the reason), or not sure (analyse: a
  small person, someone seen in one frame only, a weak detection, a dark or flashing picture, frames that could not
  be read, too few frames). Only a clear 'nobody' skips a cut.
The cuts of one source are read in their order on it, so a baked timeline is read forward.

Why a long cut is read whole before it is called empty (PC 2026-11-21, j2785b; instruction 2026-11-20 §1-1 "긴 컷의
속도 최적화는 역광·작은 인물·짧은 등장 표본에서 오검출을 비교한 뒤 적용"): frames of real cuts with a person (as
shot, backlit, dark, a third of the size) were put into 581 frames of the projects' empty cuts for 1, 2, 3, 6 and 12
frames, and each way of sampling was asked for its answer. The detector saw the person on 98-100 % of those frames
in every condition; what missed them was the sampling: someone there for one frame was called 'nobody' in 3.6 % of
the cuts by the two rounds above (11 % of the 96-frame cuts), in 2.1 % by every second frame, in 7.8 % by every
twelfth; for two frames 2.8 % / 0 % / 7.7 %; only reading every frame missed none. Reading and looking at a frame
cost 0.04-0.1 s there (decode 0.013-0.066 s, detector 0.025-0.043 s), so an empty cut of 600 frames costs about
half a minute more; the projects' own 27 empty cuts were 7-72 frames long. Reading the frames smaller (960x540
straight from the decoder) gave the same answers on 75 frames x 3 conditions and is 3.5 times faster for frames far
apart (0.12-0.15 s against 0.38-0.52 s) but no faster for frames in a row, which is what an empty cut now is."""
from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

PRESENT, NONE, UNSURE = "present", "none", "unsure"
SURE_SCORE = 0.50          # a detection this sure ...
SURE_H = 0.08              # ... at least this tall (of the frame height): someone to retouch
SEEN_SCORE = 0.25          # anything above this is somebody maybe (small, half hidden, dark)
SMALL_H = 0.03
DARK = 18.0                # mean picture level (0..255) under which a frame is too dark to tell
DENSE = 48                 # cuts up to this many frames are looked at every second frame
WIDTH = 960                # detector input width (the detector itself resizes to 640)


def plan(n: int, more: bool = False) -> List[int]:
    """Frames to look at in a cut of n frames (more: the second round of a long cut)."""
    n = int(n)
    if n <= 0:
        return []
    if n <= DENSE:
        return list(range(0, n, 2)) + ([n - 1] if (n - 1) % 2 else [])
    # second round: about every half second (a person seen for less is not retouched), at most 160 frames
    k = 13 if not more else int(np.clip(np.ceil(n / 12.0), 37, 160))
    ts = np.unique(np.linspace(0, n - 1, k).round().astype(int)).tolist()
    if more:
        first = set(plan(n))
        ts = [t for t in ts if t not in first]
    return ts


def judge(found: List[Optional[List[tuple]]], lumas: List[float], planned: int) -> dict:
    """found: per frame looked at, the detections [(score, height share)] or None when it could not be read.
    The answer, why, and the figures it came from."""
    read = [f for f in found if f is not None]
    failed = len(found) - len(read)
    sure = [any(s >= SURE_SCORE and h >= SURE_H for s, h in f) for f in read]
    maybe = [any(s >= SEEN_SCORE for s, h in f) for f in read]
    small = [any(s >= SEEN_SCORE and SMALL_H <= h < SURE_H for s, h in f) for f in read]
    best = max([s for f in read for s, _ in f], default=0.0)
    tall = max([h for f in read for s, h in f if s >= SEEN_SCORE], default=0.0)
    dark = sum(1 for v in lumas if v < DARK)
    flash = 0
    if len(lumas) >= 3:
        L = np.asarray(lumas, np.float64)
        med = float(np.median(L))
        flash = int(np.sum(np.abs(L - med) > max(40.0, 0.6 * med)))
    figs = dict(looked=len(found), read=len(read), failed=failed, sure_frames=int(sum(sure)),
                maybe_frames=int(sum(maybe)), small_frames=int(sum(small)), best_score=round(float(best), 3),
                tallest=round(float(tall), 3), dark_frames=dark, flash_frames=flash, planned=planned)
    if sum(sure) >= 2 or (sum(sure) == 1 and len(read) <= 3):
        return dict(verdict=PRESENT, reason="인물 있음", **figs)
    if sum(sure) == 1:
        return dict(verdict=UNSURE, reason="한 프레임에서만 보인 인물", **figs)
    if not read or len(read) < max(3, min(planned, 6)) or failed > 0.25 * max(1, len(found)):
        return dict(verdict=UNSURE, reason="프레임을 충분히 읽지 못함", **figs)
    if any(small):
        return dict(verdict=UNSURE, reason="작은 인물", **figs)
    if any(maybe):
        return dict(verdict=UNSURE, reason="낮은 검출 신뢰도", **figs)
    if dark > 0.3 * len(read):
        return dict(verdict=UNSURE, reason="어두운 화면", **figs)
    if flash:
        return dict(verdict=UNSURE, reason="플래시·조명 변화", **figs)
    return dict(verdict=NONE, reason=f"인물 없음 (빠른 검사: {len(read)}프레임 확인)", **figs)


class Prescan:
    """One detector for every cut of a batch."""

    def __init__(self, detector=None):
        if detector is None:
            from .infer import Detector
            detector = Detector("det_yolox_m", score_thr=0.15, nms_thr=0.6)
        self.det = detector

    def look(self, frame_of: Callable[[int], Optional[np.ndarray]], n: int) -> dict:
        """frame_of(t): the cut's frame t as the analysis sees it (RGB uint8), None when it cannot be read."""
        t0 = time.time()
        found: List[Optional[List[tuple]]] = []
        lumas: List[float] = []

        def sure_frames() -> int:
            return sum(1 for f in found if f and any(s_ >= SURE_SCORE and h >= SURE_H for s_, h in f))

        def run(ts) -> int:
            """Looks at ts in order; stops once two frames show someone plainly (the rest is not needed: most
            cuts have people, and reading every planned frame of a 4K original cost ~2.8 s a cut where ~0.5 s
            will do). The number of frames looked at."""
            k = 0
            for t in ts:
                self._one(frame_of, t, found, lumas)
                k += 1
                if sure_frames() >= 2:
                    break
            return k
        looked = run(plan(n))
        r = judge(found, lumas, looked)
        open_ = ("낮은 검출 신뢰도", "작은 인물", "프레임을 충분히 읽지 못함", "한 프레임에서만 보인 인물")
        if r["verdict"] == NONE or r["reason"] in open_:
            if n > DENSE:
                # a long cut: never called empty from its first frames alone (someone may come in between them),
                # and one left open is looked at closer before the large models are opened for it
                more = plan(n, more=True)
            else:
                # a short cut: every frame is looked at before it is called empty (someone there on one frame
                # between the samples was missed: 20 frames, a person on frame 5 only)
                seen = set(plan(n))
                more = [t for t in range(int(n)) if t not in seen]
            if more:
                looked += run(more)
                r = judge(found, lumas, looked)
            if n > DENSE and r["verdict"] == NONE:
                # still nobody: the frames between the samples, every one, before the cut is skipped (someone
                # there for a frame or two between them was called 'nobody'; see the module's notes)
                seen = set(plan(n)) | set(plan(n, more=True))
                rest = [t for t in range(int(n)) if t not in seen]
                if rest:
                    looked += run(rest)
                    r = judge(found, lumas, looked)
        r["seconds"] = round(time.time() - t0, 2)
        return r

    def _one(self, frame_of, t, found, lumas):
        try:
            img = frame_of(int(t))
        except Exception:
            img = None
        if img is None:
            found.append(None)
            return
        img = np.asarray(img)
        H, W = img.shape[:2]
        if W > WIDTH:
            import cv2
            s = WIDTH / W
            img = cv2.resize(img, (WIDTH, max(2, int(round(H * s)))), interpolation=cv2.INTER_AREA)
            H, W = img.shape[:2]
        lumas.append(float(img[::4, ::4].mean()))
        try:
            boxes, scores = self.det(np.ascontiguousarray(img))
        except Exception:
            found.append(None)
            return
        dets = []
        for b, s in zip(boxes, scores):
            h = (float(b[3]) - float(b[1])) / max(1.0, H)
            dets.append((float(s), h))
        found.append(dets)


def scan(project, cuts: Iterable, progress: Optional[Callable[[float, str], None]] = None,
         cancel: Optional[Callable[[], bool]] = None, prescan: Optional[Prescan] = None,
         frames_for: Optional[Callable] = None) -> Dict[str, dict]:
    """{cut id: answer} for the cuts, read in their order on each source (a baked timeline forward)."""
    from ..core.frames import frames_for as _ff
    ff = frames_for or _ff
    cuts = sorted(list(cuts), key=lambda c: (str(c.source or c.info.path), int(getattr(c.info, "offset", 0) or 0)))
    ps = prescan or Prescan()
    out: Dict[str, dict] = {}
    for i, c in enumerate(cuts):
        if cancel and cancel():
            break
        if progress:
            progress(i / max(1, len(cuts)), f"인물 빠른 확인 {i + 1}/{len(cuts)}")
        try:
            fr = ff(project, c)
        except Exception:
            fr = None
        if fr is None:
            out[c.id] = dict(verdict=UNSURE, reason="원본을 읽지 못함", looked=0, read=0, failed=0)
            continue
        a, b = c.work_range
        try:
            out[c.id] = ps.look(lambda t, a=a, fr=fr: fr.frame(a + t), max(0, b - a))
        finally:
            try:
                fr.close()
            except Exception:
                pass
    if progress:
        progress(1.0, "인물 빠른 확인 끝")
    return out


SKIP_KEYS = ("prescan", "prescan_info")


def clear_skip(cut, cut_dir=None) -> bool:
    """The user asked for this cut to be analysed: what an earlier quick look decided ('nobody, skipped') is taken
    off the cut (and its prescan.json removed), so nothing downstream skips it again. True when there was one."""
    st = dict(getattr(cut, "status", None) or {})
    had = bool(st.get("prescan")) or (st.get("auto") == "none" and "빠른 검사" in str(st.get("reason", "")))
    if had:
        for k in SKIP_KEYS + ("auto", "reason"):
            st.pop(k, None)
        cut.status = st
    if cut_dir is not None:
        try:
            f = cut_dir / "prescan.json"
            if f.exists():
                f.unlink()
        except OSError:
            pass
    return had
