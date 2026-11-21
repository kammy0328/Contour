"""Clothing areas per person, for 'keep the clothes as they are' in the body reshaping.

A human parsing model (SegFormer-B2 fine-tuned for clothes, 18 classes) labels every pixel of each tracked
person's crop: skin parts (face, arms, legs), hair, and each kind of clothing. The class map is stored per frame
at matte resolution (class + 1, 0 = nobody) next to the matte; the person owning a pixel comes from the person
labels, so overlapping people never lend each other their clothes."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from . import skeleton as S
from .exposure import analysis_view
from .modelpkg import cut_models as _cut_models

CLASSES = ["배경", "모자", "머리카락", "선글라스", "상의", "치마", "바지", "원피스", "벨트", "왼 신발", "오른 신발",
           "얼굴", "왼 다리", "오른 다리", "왼 팔", "오른 팔", "가방", "스카프"]
SKIRT, DRESS = 5, 7
ALL_CLOTHES = {1, 4, 5, 6, 7, 8, 9, 10, 16, 17}
# "옷 그대로": which garments stay as they are. Only garments found on the person are offered.
GARMENTS = [("top", "상의", {4, 17}), ("bottom", "바지", {6, 8}), ("skirt", "치마", {5}), ("dress", "원피스", {7})]
ACCESSORIES = {1, 9, 10, 16}           # hat, shoes, bag: kept whenever clothes are kept
MODES = ("include", "skirt", "all")    # older projects; now "keep:top,skirt,..."


def keep_set(mode: str) -> set:
    """Garment keys kept by a cloth mode ('include' -> none)."""
    mode = str(mode or "include")
    if mode == "include":
        return set()
    if mode == "all":
        return {k for k, _, _ in GARMENTS}
    if mode == "skirt":
        return {"skirt", "dress_low"}
    if mode.startswith("keep:"):
        return {k for k in mode[5:].split(",") if k}
    return set()


def detected(tracker, p: int, n: int = 12) -> list:
    """Garment keys seen on this person (more than 1 % of the person's pixels in the sampled frames)."""
    if not tracker.cloth.exists():
        return []
    st = tracker.store
    ts = [t for t in np.linspace(0, st.n - 1, n).astype(int) if st.present[p, t]]
    tot = 0
    cnt = {k: 0 for k, _, _ in GARMENTS}
    for t in ts:
        cm = tracker.cloth.get(int(t))
        lab = tracker.labels(int(t))
        if cm is None or lab is None:
            continue
        own = lab == p
        tot += int(own.sum())
        cls = cm.astype(np.int16) - 1
        for k, _, ids in GARMENTS:
            cnt[k] += int((own & np.isin(cls, list(ids))).sum())
    if tot <= 0:
        return []
    return [k for k, _, _ in GARMENTS if cnt[k] > 0.01 * tot]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_MODEL = "sam31"          # commercial use (2026-11-19, j2758); parse_clothes: non-commercial, on request
GARMENT_FILE = Path(__file__).resolve().parent / "data" / "sam31_text_garments.npz"
# SAM 3.1 phrases -> the clothes model's classes (so the stored map, keep_mask and detected stay as they are)
PHRASES = [("shirt", 4), ("jacket", 4), ("coat", 4), ("sweater", 4), ("hoodie", 4), ("top", 4), ("pants", 6),
           ("shorts", 6), ("skirt", 5), ("dress", 7), ("belt", 8), ("hat", 1), ("shoe", 9), ("bag", 16)]
# anything else worn (2026-11-19: a denim jacket none of the garment words found): counted as a top, only where no
# garment word claimed the pixel
CATCH_ALL = ("clothing", 4)
GARMENT_SCORE = 0.4


def cloth_model() -> str:
    """Which model finds clothing: 'sam31' (SAM 3.1's finder with garment phrases: SAM License, commercial use
    allowed; the phrases' encodings are shipped in data/) or 'parse_clothes' (SegFormer-B2: NVIDIA licence,
    non-commercial). settings.cloth_model, DIAP_CLOTH_MODEL overrides."""
    import os
    v = os.environ.get("DIAP_CLOTH_MODEL", "")
    if not v:
        try:
            from ..core.settings import settings
            v = getattr(settings(), "cloth_model", "") or ""
        except Exception:
            v = ""
    # a missing garment-word file never brings parse_clothes in by itself: run() says so and leaves the clothing
    # undecided
    return v if v in ("sam31", "parse_clothes") else DEFAULT_MODEL


class GarmentFinder:
    """Every garment in a frame with SAM 3.1: the picture encoded once, its finder run with each garment phrase.
    Returns a class map in the clothes model's classes (0: none). Where two phrases claim a pixel, the surer one
    (higher mask logit times score) takes it."""

    def __init__(self):
        from .infer import Finder31, Segmenter31
        self.seg = Segmenter31()
        self.finder = Finder31()
        d = np.load(str(GARMENT_FILE))
        self.phr = []
        for ph, cls in PHRASES + [CATCH_ALL]:
            k = ph.replace(" ", "_")
            if f"{k}__text" in d.files:
                self.phr.append((cls, d[f"{k}__text"].astype(np.float32), d[f"{k}__text_pad"].astype(bool),
                                 ph == CATCH_ALL[0]))

    stats: dict = None
    also: tuple = ()        # more phrases for the same picture: (class, text, text_pad). What they find takes only the
                            # pixels no garment word claims (track.skincloth: 'hair', class 2)

    def classes(self, rgb: np.ndarray, size: Tuple[int, int], only: Optional[set] = None,
                catch_all: str = "always") -> np.ndarray:
        """rgb uint8 (H, W, 3) -> (h, w) uint8 class map at size (w, h).
        only: search just the phrases of these classes (an experiment for 'keep the skirt' style modes);
        catch_all: 'always', 'when_empty' (only when no garment word found anything) or 'never'."""
        if self.stats is None:
            self.stats = dict(frames=0, encode_s=0.0, find_s=0.0, find_calls=0)
        w, h = size
        t0 = time.time()
        enc = self.seg.encode(rgb)
        self.stats["encode_s"] += time.time() - t0
        self.stats["frames"] += 1
        best = np.full((h, w), 0.0, np.float32)
        out = np.zeros((h, w), np.uint8)
        base = dict(self.finder.const)
        try:
            for cls, text, pad, is_catch in self.phr:
                if only is not None and cls not in only and not is_catch:
                    continue
                if is_catch and (catch_all == "never" or (catch_all == "when_empty" and out.any())):
                    continue
                self.finder.const = dict(base, text=text, text_pad=pad)
                t1 = time.time()
                found = self.finder.find(enc, GARMENT_SCORE)
                self.stats["find_s"] += time.time() - t1
                self.stats["find_calls"] += 1
                for d in found:
                    lg = cv2.resize(d["lg"], (w, h), interpolation=cv2.INTER_LINEAR)
                    sc = np.where(lg > 0, d["score"] * (1.0 / (1.0 + np.exp(-lg))), 0.0).astype(np.float32)
                    win = (sc > 0) & (out == 0) if is_catch else sc > best
                    out[win] = cls
                    best[win] = sc[win]
            for cls, text, pad in (self.also or ()):
                self.finder.const = dict(base, text=text, text_pad=pad)
                t1 = time.time()
                found = self.finder.find(enc, GARMENT_SCORE)
                self.stats["find_s"] += time.time() - t1
                self.stats["find_calls"] += 1
                for d in found:
                    lg = cv2.resize(d["lg"], (w, h), interpolation=cv2.INTER_LINEAR)
                    out[(lg > 0) & (out == 0)] = cls
        finally:
            self.finder.const = base
        return out


class UnifiedFailed(RuntimeError):
    """The combined model could not make the clothing masks (the run goes on with SAM 3.1's garment words)."""


class Parser:
    SIZE = 512

    def __init__(self, key: str = "parse_clothes", unified_only: bool = False):
        """unified_only: the combined model or nothing (UnifiedFailed): with SAM 3.1 garments as the setting, a
        failing combined model must never bring parse_clothes (non-commercial) in."""
        from . import unified as U
        self.key = key
        self.unified_only = unified_only
        self.used: set = set()
        self.unified = U.runtime() if (key == "parse_clothes" and U.active("cloth")) else None
        self.sess = None
        if self.unified is None:
            if unified_only:
                raise UnifiedFailed("통합 모델을 불러오지 못했습니다")
            self._load_own()

    def _load_own(self):
        from . import models as M
        p = M.ensure_model(self.key)
        self.sess = M.session(p)
        self.inp = self.sess.get_inputs()[0].name
        from .faceparse import file_fp
        self._own_key = f"{self.key}@{file_fp(p)}"

    def provider(self) -> str:
        return "+".join(sorted(self.used))

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        """RGB uint8 crop -> class map (h, w) uint8."""
        h, w = crop.shape[:2]
        c512 = cv2.resize(crop, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        lg = None
        if self.unified is not None:
            try:
                lg = self.unified.cloth_logits(c512)
                from . import unified as U
                from .faceparse import file_fp
                self.used.add("unified@" + file_fp(U.path()))
            except Exception as e:
                from . import unified as U
                U.fail("cloth", e)
                self.unified = None
                if self.unified_only:
                    raise UnifiedFailed(str(e))
                self._load_own()
        if lg is None:
            x = ((c512.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
            lg = self.sess.run(None, {self.inp: x})[0][0]             # (18, 128, 128)
            self.used.add(self._own_key)
        up = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in lg])
        return up.argmax(0).astype(np.uint8)


def _crop_box(st, p: int, t: int, sx: float, sy: float, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    b = st.boxes[p, t]
    kp = st.final[p, t]
    pts = []
    if np.isfinite(b).all():
        pts += [b[:2], b[2:]]
    ok = np.isfinite(kp[:, :2]).all(1)
    if ok.any():
        pts += list(kp[ok, :2])
    if len(pts) < 2:
        return None
    P = np.array(pts, np.float64) * [sx, sy]
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, int(x0 - 0.10 * w)), min(W, int(np.ceil(x1 + 0.10 * w)))
    y0, y1 = max(0, int(y0 - 0.12 * h)), min(H, int(np.ceil(y1 + 0.06 * h)))
    if x1 - x0 < 16 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def _parse_frame(parser, st, proxy, pres, t: int, frame: np.ndarray, lab, rgb, mw: int, mh: int) -> np.ndarray:
    """One frame of the clothing store with the parser: each present person's crop parsed and written into the
    pixels that person owns (lab: the owners, store index per matte pixel)."""
    W, H = proxy.w, proxy.h
    kx, ky = mw / W, mh / H
    for p in pres:
        if lab is not None:
            frame[lab == p] = 0
        box = _crop_box(st, p, t, proxy.sx, proxy.sy, W, H)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        cm = parser(rgb[y0:y1, x0:x1])
        mx0, my0 = int(x0 * kx), int(y0 * ky)
        mx1, my1 = max(mx0 + 1, int(round(x1 * kx))), max(my0 + 1, int(round(y1 * ky)))
        small = cv2.resize(cm, (mx1 - mx0, my1 - my0), interpolation=cv2.INTER_NEAREST)
        own = (lab[my0:my1, mx0:mx1] == p) if lab is not None else (small > 0)
        reg = frame[my0:my1, mx0:mx1]
        reg[own] = small[own] + 1
    return frame


def _garment_frame(gf, st, proxy, pres, t: int, frame: np.ndarray, lab, rgb, mw: int, mh: int) -> np.ndarray:
    """One frame of the clothing store with SAM 3.1's garment words: one class map for the picture, each present
    person's own pixels taken from it."""
    cmap = gf.classes(rgb, (mw, mh))
    for p in pres:
        if lab is not None:
            own = lab == p
        else:
            box = _crop_box(st, p, t, proxy.sx, proxy.sy, proxy.w, proxy.h)
            if box is None:
                continue
            own = np.zeros((mh, mw), bool)
            x0, y0, x1, y1 = box
            own[int(y0 * mh / proxy.h):int(np.ceil(y1 * mh / proxy.h)),
                int(x0 * mw / proxy.w):int(np.ceil(x1 * mw / proxy.w))] = True
        frame[own] = 1                                    # the person's own pixels: nothing kept ...
        g = own & (cmap > 0)
        frame[g] = cmap[g] + 1                            # ... but the garments found on them
    return frame


@_cut_models
def run(tracker, persons: List[int], progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None) -> bool:
    """Parse the given people (store indices) on every frame of the work range into the clothing store. The masks
    are written to a copy and swapped in only when every frame is done (cancelled or failed: the masks in use stay,
    and nothing is marked ready). Which model made them is recorded with them (info())."""
    from . import unified as U
    t0 = time.time()
    if cloth_model() == "sam31":
        if U.active("cloth"):
            try:
                return _run_parser(tracker, persons, progress, cancel, Parser(unified_only=True), t0)
            except UnifiedFailed as e:
                U.fail("cloth", e)          # SAM 3.1 garment words from the start, never parse_clothes
                tracker.cloth.drop_tmp()
        return _run_garments(tracker, persons, progress, cancel, t0)
    return _run_parser(tracker, persons, progress, cancel, Parser(), t0)


def _run_parser(tracker, persons, progress, cancel, parser, t0) -> bool:
    st = tracker.store
    proxy = tracker.proxy
    a, b = tracker.cut.work_range
    mw, mh = tracker.matte.w, tracker.matte.h
    mm = tracker.cloth.open_tmp()
    ok = False
    try:
        for t in range(a, b):
            if cancel and cancel():
                return False
            pres = [p for p in persons if st.present[p, t]]
            if not pres:
                continue
            rgb = analysis_view(np.asarray(proxy.frame(t)))
            lab = tracker.labels(t)
            mm[t] = _parse_frame(parser, st, proxy, pres, t, np.array(mm[t]), lab, rgb, mw, mh)
            if progress:
                progress((t - a + 1) / max(1, b - a), f"옷 영역 분석 {t - a + 1}/{b - a}")
        ok = True
    finally:
        mm.flush()
        del mm
        if ok:
            tracker.cloth.commit_tmp()
        else:
            tracker.cloth.drop_tmp()
        tracker.cloth.close()
    _write_state(tracker, persons, parser.provider(), time.time() - t0)
    return True


def _run_garments(tracker, persons: List[int], progress=None, cancel=None, t0=None) -> bool:
    """run() with SAM 3.1's garment phrases: one class map per frame, each person's own pixels taken from it."""
    st = tracker.store
    proxy = tracker.proxy
    a, b = tracker.cut.work_range
    t0 = time.time() if t0 is None else t0
    if not GARMENT_FILE.is_file():
        # never parse_clothes in its place: the clothing stays undecided and the reason is said
        raise RuntimeError(f"옷 단어 파일이 없어 옷 영역을 분석하지 못했습니다: {GARMENT_FILE.name} (앱을 다시 설치하세요)")
    gf = GarmentFinder()
    mw, mh = tracker.matte.w, tracker.matte.h
    mm = tracker.cloth.open_tmp()
    ok = False
    try:
        for t in range(a, b):
            if cancel and cancel():
                return False
            pres = [p for p in persons if st.present[p, t]]
            if not pres:
                continue
            rgb = analysis_view(np.asarray(proxy.frame(t)))
            lab = tracker.labels(t)
            mm[t] = _garment_frame(gf, st, proxy, pres, t, np.array(mm[t]), lab, rgb, mw, mh)
            if progress:
                progress((t - a + 1) / max(1, b - a), f"옷 영역 분석 {t - a + 1}/{b - a}")
        ok = True
    finally:
        mm.flush()
        del mm
        if ok:
            tracker.cloth.commit_tmp()
        else:
            tracker.cloth.drop_tmp()
        tracker.cloth.close()
    from .faceparse import file_fp
    _write_state(tracker, persons, "sam31@" + file_fp(GARMENT_FILE), time.time() - t0, gf.stats)
    return True


def _state_path(tracker) -> Path:
    return Path(getattr(tracker.cloth, "live_path", tracker.cloth.path)).with_suffix(".json")


def _write_state(tracker, persons: List[int], model: str = "", seconds: Optional[float] = None,
                 stats: Optional[dict] = None) -> None:
    """After a whole run: who was parsed, for which analysis and range, by what. Every frame was cut with the
    owners in use now, so no range is stale."""
    st = tracker.store
    old = _state(tracker)
    same = old.get("analysis") == tracker.cut.analysis_version and old.get("model", "") == model
    pids = sorted(set(old.get("pids", [])) | {int(st.pids[p]) for p in persons}) \
        if same else sorted(int(st.pids[p]) for p in persons)
    d = dict(analysis=tracker.cut.analysis_version, range=list(tracker.cut.work_range), pids=pids, model=model,
             made=time.time(), seconds=None if seconds is None else round(float(seconds), 2), stale=[])
    if stats:
        d["stats"] = stats
    _put_state(tracker, d)


def _put_state(tracker, d: dict) -> None:
    import os
    tmp = _state_path(tracker).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    os.replace(tmp, _state_path(tracker))
    tracker.__dict__.pop("_cloth_state", None)


def state_text(tracker) -> Optional[str]:
    """The state file as it is (None: there is none): kept by a matte revision to put back when it is undone."""
    try:
        return _state_path(tracker).read_text(encoding="utf-8")
    except OSError:
        return None


def put_state_text(tracker, text: Optional[str]) -> None:
    import os
    f = _state_path(tracker)
    if text is None:
        try:
            f.unlink()
        except OSError:
            pass
    else:
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, f)
    tracker.__dict__.pop("_cloth_state", None)


def _state(tracker) -> dict:
    """The one reader of the clothing state, for the preview, a render, the inspector and a matte revision alike
    (1.22.2 had two: one kept per tracker by the file's time in seconds, one reading the file each time). Kept on
    the tracker by the file's time (ns) and size; every writer here drops that copy."""
    f = _state_path(tracker)
    try:
        fs = f.stat()
    except OSError:
        return {}
    key = (fs.st_mtime_ns, fs.st_size)
    c = tracker.__dict__.get("_cloth_state")
    if c is not None and c[0] == key:
        return c[1]
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    tracker.__dict__["_cloth_state"] = (key, d)
    return d


def stale_ranges(tracker) -> List[List[int]]:
    """[a, b) ranges whose clothing was cut with owners that are no longer the ones in use (the person matte was
    made again there since): not used until it is made again there. Written with the matte revision that changed
    the owners (set_stale). A state from 1.22.2 or older has no such list: the ranges the matte was refreshed in
    after the clothes were analysed are taken, as its inspector line said."""
    s = _state(tracker)
    if not s:
        return []
    if "stale" in s:
        return [[int(a), int(b)] for a, b in (s.get("stale") or [])]
    try:
        made = float(s.get("made") or 0)
        from .matte_refresh import merge
        return merge([r[:2] for r in (tracker.seg_meta().get("refreshed") or []) if len(r) > 2 and r[2] > made], 0)
    except Exception:
        return []


def stale_of_cut(project, cut) -> List[List[int]]:
    """stale_ranges() of a cut that is not open, from its files alone (a render asking before it starts). []
    when nobody who is retouched keeps clothes in the cut: stale clothing changes nothing then."""
    try:
        keeps = any(p.role == "target" and getattr(cut.params.get(p.id), "cloth", "include") != "include"
                    for p in cut.persons)
        if not keeps:
            return []
        cache = Path(project.cache_dir(cut))
        fs = sorted(cache.glob("cloth_*.json"))
        if not fs:
            return []
        s = json.loads(fs[-1].read_text(encoding="utf-8"))
        if s.get("analysis") != cut.analysis_version or list(s.get("range", [])) != list(cut.work_range):
            return []
        if "stale" in s:
            return [[int(a), int(b)] for a, b in (s.get("stale") or [])]
        from .matte_refresh import merge
        m = json.loads((cache / "seg_meta.json").read_text(encoding="utf-8"))
        made = float(s.get("made") or 0)
        return merge([r[:2] for r in (m.get("refreshed") or []) if len(r) > 2 and r[2] > made], 0)
    except Exception:
        return []


def stale_at(tracker, t: int) -> bool:
    return any(a <= t < b for a, b in stale_ranges(tracker))


def set_stale(tracker, add=None, done=None) -> List[List[int]]:
    """Ranges whose owners changed are added, ranges made again with the present owners taken off; written into
    the state at once (the preview and a render read the same file)."""
    from .matte_refresh import merge, subtract
    s = dict(_state(tracker))
    if not s:
        return []
    cur = stale_ranges(tracker)
    if add:
        cur = merge(cur + [[int(a), int(b)] for a, b in add], 0)
    if done:
        cur = subtract(cur, [[int(a), int(b)] for a, b in done])
    s["stale"] = cur
    _put_state(tracker, s)
    return cur


def info(tracker) -> dict:
    """What the inspector shows about this cut's clothing masks: model (provider key, '' = not recorded), its
    words, state ('none' / 'unknown' / 'fallback' / 'current' / 'other'), seconds the last run took."""
    from .facelayer import provider_state
    from .faceparse import label
    from . import unified as U
    if not tracker.cloth.exists():
        return dict(model="", label="없음", state="none", seconds=None)
    s = _state(tracker)
    model = s.get("model", "")
    want = {cloth_model()} | ({"unified"} if U.active("cloth") else set())
    out = dict(model=model, label=label(model), state=provider_state(model, want), seconds=s.get("seconds"),
               stats=s.get("stats"))
    # the owners the clothes were cut with changed in these frames (a matte revision since): not used there until
    # made again
    later = stale_ranges(tracker)
    if later:
        out["stale_ranges"] = later
    return out


def _ready_pid(tracker, p: int) -> bool:
    s = _state(tracker)
    st = tracker.store
    return bool(s) and s.get("analysis") == tracker.cut.analysis_version \
        and list(s.get("range", [])) == list(tracker.cut.work_range) and int(st.pids[p]) in s.get("pids", [])


def ready_for(tracker, pid: int) -> bool:
    """Has this person's clothing been parsed for the current analysis and work range?"""
    if not tracker.cloth.exists():
        return False
    s = _state(tracker)
    return bool(s) and s.get("analysis") == tracker.cut.analysis_version \
        and list(s.get("range", [])) == list(tracker.cut.work_range) and int(pid) in s.get("pids", [])


def _parsed_persons(tracker) -> List[int]:
    """Store indices of the people whose clothing is parsed for the present analysis and range."""
    st = getattr(tracker, "store", None)
    if st is None or not hasattr(tracker, "cloth") or not tracker.cloth.exists():
        return []
    s = _state(tracker)
    if not s or s.get("analysis") != tracker.cut.analysis_version \
            or list(s.get("range", [])) != list(tracker.cut.work_range):
        return []
    return [st.index(int(q)) for q in s.get("pids", []) if st.has(int(q))]


def _frame_maker(tracker, models: Optional[dict] = None):
    """make(pres, t, frame, lab, rgb) -> frame, with the clothing model of the settings (kept in models between
    runs). Raises when the masks in use were made by another model than the settings give now: frames of two
    models are never mixed in one cut (the whole cut is made again by hand then)."""
    from . import unified as U
    if info(tracker).get("state") != "current":
        raise RuntimeError("옷 영역을 만든 모델이 지금 설정과 달라 일부 구간만 다시 만들지 않았습니다 (옷 마스크 다시 만들기)")
    models = models if models is not None else {}
    st, proxy = tracker.store, tracker.proxy
    mw, mh = tracker.matte.w, tracker.matte.h
    if cloth_model() == "sam31" and not U.active("cloth"):
        if not GARMENT_FILE.is_file():
            raise RuntimeError(f"옷 단어 파일이 없어 옷 영역을 분석하지 못했습니다: {GARMENT_FILE.name}")
        gf = models.get("garments")
        if gf is None:
            gf = models["garments"] = GarmentFinder()
        return (lambda pres, t, frame, lab, rgb: _garment_frame(gf, st, proxy, pres, t, frame, lab, rgb, mw, mh))
    key = "cloth_parser"
    parser = models.get(key)
    if parser is None:
        parser = models[key] = Parser(unified_only=cloth_model() == "sam31")
    return (lambda pres, t, frame, lab, rgb: _parse_frame(parser, st, proxy, pres, t, frame, lab, rgb, mw, mh))


def restage(tracker, frames, staged: dict, cancel=None, models: Optional[dict] = None) -> Optional[dict]:
    """For a matte revision that is not in use yet (matte_rev.stage): the clothing of the frames whose owners it
    changes, cut with the new owners, for the people whose clothing is parsed. {'frames': {t: frame}, 'stale':
    [t, ...]} (stale: frames that could not be made again and are not used until they are), or None when the cut
    has no clothing in use (nothing to do). Nothing is written."""
    persons = _parsed_persons(tracker)
    if not persons:
        return None
    st, proxy = tracker.store, tracker.proxy
    frames = [int(t) for t in frames if any(st.present[p, int(t)] for p in persons)]
    if not frames:
        return dict(frames={}, stale=[])
    try:
        make = _frame_maker(tracker, models)
    except Exception as e:
        return dict(frames={}, stale=frames, error=str(e)[:200])
    out: dict = {}
    for i, t in enumerate(frames):
        if cancel and cancel():
            from .analyze import Cancelled
            raise Cancelled()
        pres = [p for p in persons if st.present[p, t]]
        seg = staged["labels"].get(t) if t in staged.get("labels", {}) else (
            tracker.seglab.get(t) if tracker.seglab.exists() else None)
        al = staged["matte"].get(t) if t in staged.get("matte", {}) else tracker.matte.get(t)
        lab = tracker.labels_from(t, seg, al)
        cur = tracker.cloth.get(t)
        frame = np.zeros((tracker.matte.h, tracker.matte.w), np.uint8) if cur is None else np.array(cur)
        rgb = analysis_view(np.asarray(proxy.frame(t)))
        out[t] = make(pres, t, frame, lab, rgb)
    return dict(frames=out, stale=[])


@_cut_models
def refresh_stale(tracker, progress=None, cancel=None, models: Optional[dict] = None) -> List[List[int]]:
    """The stale ranges made again with the owners in use (a refresh whose clothing could not be made with it, a
    cut from 1.22.2): frame by frame, each range taken off the list once all of it is written. The ranges done."""
    rs = stale_ranges(tracker)
    persons = _parsed_persons(tracker)
    if not rs or not persons:
        return []
    st, proxy = tracker.store, tracker.proxy
    make = _frame_maker(tracker, models)        # raises with the reason when it cannot be done
    total = sum(b - a for a, b in rs)
    n = 0
    done: List[List[int]] = []
    for a, b in rs:
        mm = tracker.cloth.open_rw()
        try:
            for t in range(a, b):
                if cancel and cancel():
                    return done
                pres = [p for p in persons if st.present[p, t]]
                if pres:
                    rgb = analysis_view(np.asarray(proxy.frame(t)))
                    mm[t] = make(pres, t, np.array(mm[t]), tracker.labels(t), rgb)
                n += 1
                if progress:
                    progress(n / max(1, total), f"옷 영역 갱신 {n}/{total}")
            mm.flush()
        finally:
            del mm
            tracker.cloth.close()
        set_stale(tracker, done=[[a, b]])
        done.append([a, b])
    return done


def keep_mask(tracker, t: int, p: int, mode: str) -> Optional[np.ndarray]:
    """Bool mask (matte resolution) of this person's clothing that stays as it is, or None."""
    keep = keep_set(mode)
    if not keep or not tracker.cloth.exists():
        return None
    if not _ready_pid(tracker, p):
        return None                       # never masks of a run that did not finish (or of another analysis)
    if stale_at(tracker, t):
        return None                       # cut with owners that are not the ones in use any more (1.22.3)
    cm = tracker.cloth.get(t)
    lab = tracker.labels(t)
    if cm is None or lab is None:
        return None
    cls = cm.astype(np.int16) - 1
    own = lab == p
    ids = set(ACCESSORIES)
    for k, _, cl in GARMENTS:
        if k in keep:
            ids |= cl
    m = own & np.isin(cls, list(ids))
    if "dress_low" in keep:
        dress = own & (cls == DRESS)
        if dress.any():
            # a dress is kept from the waist down (the top of it follows the body like any top)
            st = tracker.store
            kp = st.final[p, t]
            sh = [kp[j, :2] for j in (S.L_SH, S.R_SH) if np.isfinite(kp[j, :2]).all()]
            hp = [kp[j, :2] for j in (S.L_HIP, S.R_HIP) if np.isfinite(kp[j, :2]).all()]
            if sh and hp:
                y_waist = (0.38 * np.mean(sh, 0)[1] + 0.62 * np.mean(hp, 0)[1]) * tracker.cloth.h / tracker.cut.info.height
                rows = np.arange(m.shape[0])[:, None]
                m |= dress & (rows >= y_waist)
            else:
                m |= dress
    if not m.any():
        return None
    # pin holes and ragged edges of the parsing
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=2).astype(bool)
    return m
