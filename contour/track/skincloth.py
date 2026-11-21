"""Clothing the body parser takes for skin, found for the body skin retouch (1.23, instruction 2026-11-20 §2).

The body skin's area is the person's own pixels the body parser (MediaPipe Selfie Multiclass) calls skin, where the
skin's colour agrees. A fur collar, a top or tights of the skin's colour are skin to that parser, and the colour
agrees because it was learnt from them (LNGSHOT BANANAS 056 and 118: a fur collar and cuffs were retouched and the
gaps between the strands removed as blemishes: j2790). SAM 3.1's garment words (clothing.GarmentFinder) find such
clothes (j2794: 53..96 % of that area taken away, nothing on a cut without them), at about 1.6 s a frame: too slow
for every frame of every cut with a body value on, and most cuts have no such garment.

So the cut is looked at: every LOOK_STEP-th frame gets its garment map (and one frame of each stay of a person on the
picture that falls between two of them: look_frames), and on it, for each person, how much of what the parser is
sure is that person's bare skin the garment words call clothing. Where that is more than the two agree
on at a garment's edge, the frames from the look before to the look after get their garment maps too and the body
skin's area leaves the clothing out there (bodyskin.garments). Elsewhere nothing more is made and the parser stands
as it is: what a look costs a cut with nothing to find is a twelfth of the frames.

Hair is looked for with the garments (the word 'hair', 1.23 after j2801): fair hair in a bun seen from behind is skin
to the parser as well (2611 025: three quarters of that person's area lay on the hair). It is written into the same
maps (the clothing store's hair class) and left out of the area like a garment; of the skin the parser is sure of
only what lies outside the head the joints place counts for it (hair on the head is not the body's anyway).

Kept apart from the clothing masks of 옷 유지 (clothing.py: every frame of the cut, staged and made again with the
matte's revisions): cache/skincloth/<frame>.png (the clothing store's values at the matte's size, only the frames
that have one) and cache/skincloth.json. Where the clothing masks are in use for a person, they are read instead.
A look is of one analysis, one work range, one garment model and the owners of the time: frames whose person matte
was made again since are not used until the cut is looked at again."""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

LOOK_STEP = 12              # frames between two looks
HAIR = 2                    # the clothing store's class for hair (clothing.CLASSES)
LOOK_FILE = Path(__file__).resolve().parent / "data" / "sam31_text_skinlook.npz"      # 'hair' (tools/sam31_phrases.py)
SURE = 0.80                 # the parser's skin share from which a pixel is skin it is sure of
DIRTY_SHARE = 0.10          # of the skin the parser is sure of on a person: called clothing by the garment words ...
DIRTY_PX = 0.0004           # ... and that much of the picture, from which the frames round the look get their maps
BODY_KEYS = ("body_skin", "body_blemish")
_LRU = 24                   # garment maps kept in memory


def _dir(tracker) -> Path:
    return Path(getattr(tracker.cloth, "live_path", tracker.cloth.path)).parent / "skincloth"


def _state_file(tracker) -> Path:
    return _dir(tracker).with_suffix(".json")


def state(tracker) -> dict:
    """The look's record ({} when the cut was not looked at): kept on the tracker by the file's time and size."""
    f = _state_file(tracker)
    try:
        fs = f.stat()
    except OSError:
        return {}
    key = (fs.st_mtime_ns, fs.st_size)
    c = tracker.__dict__.get("_skincloth_state")
    if c is not None and c[0] == key:
        return c[1]
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    tracker.__dict__["_skincloth_state"] = (key, d)
    return d


def model_key() -> str:
    from . import clothing as CL
    from .faceparse import file_fp
    return "sam31@" + file_fp(CL.GARMENT_FILE) + ("+" + file_fp(LOOK_FILE) if LOOK_FILE.is_file() else "")


def more_phrases() -> tuple:
    """The look's own words beside the garments': ((class, text, text_pad), ...). () when their file is not there."""
    try:
        d = np.load(str(LOOK_FILE))
        return ((HAIR, d["hair__text"].astype(np.float32), d["hair__text_pad"].astype(bool)),)
    except Exception:
        return ()


def _fresh(tracker, s: dict) -> bool:
    """The look is of this analysis and garment model and covers the work range (in and out moved inwards since:
    its frames are the same frames)."""
    if not s or s.get("analysis") != tracker.cut.analysis_version or s.get("model") != model_key():
        return False
    r = list(s.get("range", []))
    a, b = tracker.cut.work_range
    return len(r) == 2 and r[0] <= a and b <= r[1]


def stale_ranges(tracker) -> List[List[int]]:
    """[a, b) ranges whose owners are no longer the ones the look saw (the person matte made again there since)."""
    s = state(tracker)
    if not s:
        return []
    try:
        made = float(s.get("made") or 0)
        from .matte_refresh import merge
        return merge([r[:2] for r in (tracker.seg_meta().get("refreshed") or []) if len(r) > 2 and r[2] > made], 0)
    except Exception:
        return []


MODELS = ("seg_sam31", "seg_sam31_find")       # what a look opens (SAM 3.1's encoder and its finder)


def models_missing() -> List[str]:
    """The model keys a look needs that are not on this computer (a look never downloads one: the app's model
    list does, and says so)."""
    from . import models as M
    out = []
    for k in MODELS:
        try:
            if not M.is_downloaded(k):
                out.append(k)
        except Exception:
            out.append(k)
    return out


def body_pids(cut) -> List[int]:
    """The retouched people of a cut who have a body value on (now or on any key)."""
    out = []
    for person in cut.targets():
        pp = cut.params_for(person.id)
        for k in BODY_KEYS:
            c = pp.amounts.get(k)
            if c is not None and (abs(c.base) > 1e-4 or any(abs(v) > 1e-4 for v in c.keys.values())):
                out.append(int(person.id))
                break
    return out


def needed(tracker, cut=None) -> List[int]:
    """Store indices of the people with a body value on for whom the cut is still to be looked at: nobody when the
    look is there for all of them (this analysis, range, garment model, owners) or their clothing masks are in use."""
    from . import clothing as CL
    cut = cut if cut is not None else tracker.cut
    st = getattr(tracker, "store", None)
    if st is None or not tracker.has_seg():
        return []
    want = [q for q in body_pids(cut) if st.has(q)]
    if not want:
        return []
    s = state(tracker)
    ok = _fresh(tracker, s) and not stale_ranges(tracker)
    have = set(int(q) for q in s.get("pids", [])) if ok else set()
    out = []
    for q in want:
        if q in have:
            continue
        try:
            if CL.ready_for(tracker, q) and not CL.stale_ranges(tracker):
                continue                    # every frame of this person's clothing is there already
        except Exception:
            pass
        out.append(st.index(q))
    return out


def get(tracker, t: int) -> Optional[np.ndarray]:
    """Frame t's garment map (the clothing store's values, the matte's size) where the look found the body parser's
    skin among clothing; None where it found none, where the cut was not looked at, or the owners changed since."""
    s = state(tracker)
    if not _fresh(tracker, s):
        return None
    t = int(t)
    if not any(a <= t < b for a, b in s.get("dirty", [])):
        return None
    if any(a <= t < b for a, b in stale_ranges(tracker)):
        return None
    lru = tracker.__dict__.get("_skincloth_lru")
    sig = (s.get("made"), s.get("analysis"))
    if lru is None or tracker.__dict__.get("_skincloth_sig") != sig:
        lru = tracker.__dict__["_skincloth_lru"] = OrderedDict()
        tracker.__dict__["_skincloth_sig"] = sig
    if t in lru:
        lru.move_to_end(t)
        return lru[t]
    try:
        raw = np.fromfile(str(_dir(tracker) / f"{t:06d}.png"), np.uint8)
        im = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    except Exception:
        im = None
    if im is not None and im.shape != (tracker.matte.h, tracker.matte.w):
        im = None
    lru[t] = im
    while len(lru) > _LRU:
        lru.popitem(last=False)
    return im


def token(tracker) -> str:
    """What get() depends on besides the frame, short, for cache keys."""
    s = state(tracker)
    if not s:
        return "s0"
    import zlib
    return "s%x" % (zlib.crc32(repr((s.get("made"), s.get("analysis"), s.get("range"), s.get("dirty"),
                                     stale_ranges(tracker))).encode()) & 0xffffff)


def _write(d: Path, t: int, m: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", m)
    if not ok:
        raise OSError("skincloth: png")
    tmp = d / f"{t:06d}.png.tmp"
    buf.tofile(str(tmp))
    os.replace(tmp, d / f"{t:06d}.png")


def _dirty(row: Dict[int, list], picture_px: int) -> bool:
    """row: person -> [skin the parser is sure of, of it called clothing, of it (off the head) called hair] px."""
    for v in row.values():
        n = v[0]
        if any(k >= DIRTY_PX * picture_px and k >= DIRTY_SHARE * max(1, n) for k in v[1:3]):
            return True
    return False


def look_frames(present, persons: List[int], a: int, b: int, boxes=None) -> List[int]:
    """The frames of the work range [a, b) a look looks at: every LOOK_STEP-th and the last, and for a person who is
    on the picture for a run of frames none of those falls on, one frame of that run: the one the person is largest
    on (boxes: the store's (people, frames, 4); of the frames within a tenth of the largest, the one nearest the
    run's middle), the middle one without boxes. (A person who comes and goes between two looks was never looked at:
    BANANAS 155 P9, on frames 3..11 of 21 in a top the parser takes for skin, kept all of it in the area: j2803.
    Coming into the picture, the run's middle showed 645 px of it and two frames on 13 363: j2804.)
    present: (people, frames) of the store; persons: store indices."""
    looks = set(range(a, b, LOOK_STEP)) | {b - 1}
    for p in persons:
        row = np.asarray(present[p, a:b], bool)
        if not row.any():
            continue
        edge = np.flatnonzero(np.diff(np.concatenate([[False], row, [False]]).astype(np.int8)))
        for s, e in zip(edge[::2], edge[1::2]):                 # on the picture over [a + s, a + e)
            if any(a + s <= t < a + e for t in looks):
                continue
            mid = (s + e - 1) // 2
            best = mid
            if boxes is not None:
                bx = np.nan_to_num(np.asarray(boxes[p, a + s:a + e], np.float64))
                size = np.clip(bx[:, 2] - bx[:, 0], 0, None) * np.clip(bx[:, 3] - bx[:, 1], 0, None)
                if size.size and float(size.max()) > 0:
                    near = np.flatnonzero(size >= 0.9 * float(size.max())) + s
                    best = int(near[np.argmin(np.abs(near - mid))])
            looks.add(int(a + best))
    return sorted(looks)


def ranges_of(looks: List[int], dirty: List[bool]) -> List[List[int]]:
    """The [a, b) frame ranges that get garment maps: from the look before a dirty look to the look after it."""
    from .matte_refresh import merge
    out = []
    for i, t in enumerate(looks):
        if not dirty[i]:
            continue
        a = looks[i - 1] if i > 0 else t
        b = looks[i + 1] if i + 1 < len(looks) else t
        out.append([a, b + 1])
    return merge(out, 0)


def run(tracker, persons: List[int], progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None, models: Optional[dict] = None) -> bool:
    """Looks at the cut for these people (store indices; the ones looked at before with this analysis stay) and
    makes the garment maps where the look calls for them. False: cancelled (the look in use stays as it was).
    Raises when the garment words or SAM 3.1 are not on this computer."""
    from . import clothing as CL
    from .exposure import analysis_view
    from ..retouch import bodyskin as BS
    st, proxy = tracker.store, tracker.proxy
    a, b = tracker.cut.work_range
    if b <= a:
        return True
    if not CL.GARMENT_FILE.is_file():
        raise RuntimeError(f"옷 단어 파일이 없어 옷을 확인하지 못했습니다: {CL.GARMENT_FILE.name} (앱을 다시 설치하세요)")
    models = models if models is not None else {}
    if models.get("garments") is None and models_missing():
        raise RuntimeError("피부색 옷 확인에 필요한 SAM 3.1 모델이 이 컴퓨터에 없습니다 (설정 > 모델 관리)")
    t0 = time.time()
    old = state(tracker)
    persons = sorted(set(int(p) for p in persons))
    if _fresh(tracker, old) and not stale_ranges(tracker):
        persons = sorted(set(persons) | {st.index(int(q)) for q in old.get("pids", []) if st.has(int(q))})
    gf = models.get("garments")
    if gf is None:
        gf = models["garments"] = CL.GarmentFinder()
    also_was = getattr(gf, "also", ())
    gf.also = more_phrases()
    try:
        return _run(tracker, persons, progress, cancel, gf, old, t0)
    finally:
        gf.also = also_was


def _run(tracker, persons, progress, cancel, gf, old, t0) -> bool:
    from . import clothing as CL
    from .exposure import analysis_view
    from ..retouch import bodyskin as BS
    st, proxy = tracker.store, tracker.proxy
    a, b = tracker.cut.work_range
    mw, mh = tracker.matte.w, tracker.matte.h
    W, H = int(proxy.w), int(proxy.h)
    d = _dir(tracker)
    d.mkdir(parents=True, exist_ok=True)
    looks = look_frames(st.present, persons, a, b, getattr(st, "boxes", None))
    clothes = list(CL.ALL_CLOTHES)

    def frame_map(t):
        pres = [p for p in persons if st.present[p, t]]
        if not pres:
            return None, pres, None
        rgb = np.asarray(proxy.frame(t))
        m = CL._garment_frame(gf, st, proxy, pres, t, np.zeros((mh, mw), np.uint8), tracker.labels(t),
                              analysis_view(rgb), mw, mh)
        return m, pres, rgb

    looked: Dict[int, dict] = {}
    made: set = set()
    for i, t in enumerate(looks):
        if cancel and cancel():
            return False
        m, pres, rgb = frame_map(t)
        row = {}
        if m is not None:
            g = cv2.resize(np.isin(m.astype(np.int16) - 1, clothes).astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST) > 0
            gh = cv2.resize((m.astype(np.int16) - 1 == HAIR).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            heads = None
            for p in pres:
                own = BS.ownership(tracker, t, p, (W, H), others=False)[0]
                pr = BS._parse_cached(tracker, p, t, rgb, own) if own is not None else None
                if pr is None:
                    continue
                sk = BS._place(pr, "skin", (H, W))
                if "face" in pr:
                    # the skin the area itself takes for the body's: the face class is a face where a head is, skin
                    # like the body's elsewhere (bodyskin.FACE_NEAR: a trunk the parser calls face, 2611 085, 087)
                    fc = BS._place(pr, "face", (H, W))
                    if BS.FACE_NEAR is not None:
                        if heads is None:
                            heads = BS.heads_map(BS.head_spots(st, t), (0, 0, W, H), proxy.sx, proxy.sy, 2.0)
                        fc = fc * heads
                    sk = np.clip(sk - fc, 0, 1)
                sure = (sk > SURE) & (own > 0.8)
                n_hair = 0
                if gh.any():
                    hit = sure & gh
                    if hit.any():
                        head = BS.region_maps(BS.body_info(st, p, t), (H, W), proxy.sx, proxy.sy)["head"]
                        n_hair = int((hit & (head < 0.5)).sum())
                row[int(st.pids[p])] = [int(sure.sum()), int((sure & g).sum()), n_hair]
            _write(d, t, m)
            made.add(t)
        looked[t] = row
        if progress:
            progress(0.5 * (i + 1) / len(looks), f"피부색 옷 확인 {i + 1}/{len(looks)}")
    dirty = [_dirty(looked[t], W * H) for t in looks]
    ranges = ranges_of(looks, dirty)
    rest = [t for ra, rb in ranges for t in range(ra, rb) if t not in looked]
    for i, t in enumerate(rest):
        if cancel and cancel():
            return False
        m, _pres, _rgb = frame_map(t)
        _write(d, t, m if m is not None else np.zeros((mh, mw), np.uint8))
        made.add(t)
        if progress:
            progress(0.5 + 0.5 * (i + 1) / len(rest), f"피부색 옷 영역 {i + 1}/{len(rest)}")
    # a dirty range's own looks with nobody present have no file: an empty map says so
    for ra, rb in ranges:
        for t in range(ra, rb):
            if t not in made:
                _write(d, t, np.zeros((mh, mw), np.uint8))
                made.add(t)
    s = dict(analysis=tracker.cut.analysis_version, range=[int(a), int(b)], model=model_key(), made=time.time(),
             step=LOOK_STEP, pids=sorted(int(st.pids[p]) for p in persons), dirty=ranges,
             looked={str(t): {str(q): v for q, v in looked[t].items()} for t in looks},
             seconds=round(time.time() - t0, 2), frames=len(made), stats=getattr(gf, "stats", None))
    tmp = _state_file(tracker).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s), encoding="utf-8")
    os.replace(tmp, _state_file(tracker))
    tracker.__dict__.pop("_skincloth_state", None)
    tracker.__dict__.pop("_skincloth_lru", None)
    for f in d.glob("*.png*"):                       # what an earlier look left that this one does not use
        try:
            if f.suffix == ".tmp" or int(f.name.split(".")[0]) not in made:
                f.unlink()
        except (OSError, ValueError):
            pass
    if progress:
        progress(1.0, "피부색 옷 확인 끝")
    return True


def info(tracker) -> dict:
    """For the inspector: state 'none' (not looked at), 'stale' (looked at with other owners, range or model),
    'current'; and with a current look how many frames have garment maps and in which ranges."""
    s = state(tracker)
    if not s:
        return dict(state="none")
    if not _fresh(tracker, s) or stale_ranges(tracker):
        return dict(state="stale")
    return dict(state="current", pids=list(s.get("pids", [])), ranges=[list(r) for r in s.get("dirty", [])],
                frames=int(s.get("frames", 0)), looks=len(s.get("looked", {})), seconds=s.get("seconds"))
