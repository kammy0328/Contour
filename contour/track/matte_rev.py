"""The person matte as one revision (instruction 2026-11-20 §1-2).

What belongs to a revision: the alpha of every frame (matte_*.u8), who owns each pixel (seglab_*.u8), the body
widths and outline points measured on them (in tracks.npz), the clothing classes cut with those owners where a
person keeps clothes (cloth_*.u8 and its state) and seg_meta.json. seg_meta's 'revision' number is written last:
it is the point from which the new revision is the one in use.

Until 1.22.2 a refresh wrote the new alpha and owners straight into the files in use, then measured the widths,
then saved; an analysis or a matte made again overwrote the matte in use frame by frame, long before its tracks
were saved. A failure, a cancel or the app closing in between left a matte of one revision with widths (or a whole
tracking) of another. Now everything of a revision is finished first and put in use together, and what was in use
comes back when that does not go through:

- some frames (a refresh after an edit; activate): the new alpha, owners, widths, outlines and clothing are
  complete in memory (matte_refresh.compute and stage here) before anything is written. What they replace is
  written to an undo file first (matte_undo.npz); then the new data goes in; then seg_meta. An error on the way
  puts the old data back at once; an undo file found when the cut is opened (the app ended in between) does the
  same then.
- the whole matte (an analysis, '인물 매트 강제 갱신'; begin / commit / drop): the new matte and owners are written to
  files of their own beside the ones in use (MatteStore.begin_stage), the new tracks and the face layer as well;
  commit() notes what it is about to do (matte_swap.json), moves the old files aside and the new ones in, then
  writes seg_meta. Cancelled or failed before: the stage files are deleted and nothing in use was touched. An
  error or the app ending during the move: the old files are moved back (at once, or when the cut is opened).

With a revision goes what the matte was made from (matte_prompts.npz: per frame the people, their boxes and visible
joints the matte was prompted with). An edit is compared with that, not with the tracking just before it: three
moves of 10 px are 30 px from what the matte was made of (matte_refresh.changed)."""
from __future__ import annotations

import inspect
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

UNDO = "matte_undo.npz"          # what a refresh of some frames replaces (deleted once the refresh is in use)
SWAP = "matte_swap.json"         # a whole new matte being moved in
MARK = "matte_stage.json"        # who is writing stage files (a running analysis): they are not leftovers
PROMPTS = "matte_prompts.npz"
OLD = ".old"
STALE_S = 600.0                  # a journal this old belongs to nothing that still runs

_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_ACTIVE: set = set()             # cache folders a whole new matte is being written for, in this process


class ActivationError(RuntimeError):
    """The new revision could not be put in use; the one before is in use."""


def cache_dir(tracker) -> Path:
    return Path(getattr(tracker.matte, "live_path", tracker.matte.path)).parent


def _key(folder) -> str:
    return os.path.normcase(os.path.abspath(str(folder)))


def lock_for(folder) -> threading.RLock:
    k = _key(folder)
    with _LOCKS_GUARD:
        lk = _LOCKS.get(k)
        if lk is None:
            lk = _LOCKS[k] = threading.RLock()
        return lk


def revision(tracker) -> int:
    try:
        return int((tracker.seg_meta() or {}).get("revision", 0) or 0)
    except Exception:
        return 0


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))     # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return bool(ok) and code.value == 259                                # STILL_ACTIVE
        except Exception:
            return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _write_json(path: Path, d: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(d))
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _unlink(path: Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


def _meta(tracker, model: str, ranges, full: bool, rev: int) -> None:
    f = tracker.write_seg_meta
    try:
        has = "revision" in inspect.signature(f).parameters
    except (TypeError, ValueError):
        has = False
    if has:
        f(model=model, ranges=ranges, full=full, revision=rev)
    else:
        f(model=model, ranges=ranges, full=full) if full else f(model=model, ranges=ranges)


# ---------------------------------------------------------------------- what the matte was made from
def prompts_path(tracker) -> Path:
    return cache_dir(tracker) / PROMPTS


def _prompt_rows(tracker) -> Optional[dict]:
    """The prompts of every frame as they are now: joints that are not visible, and people who are not there, as
    NaN (matte_refresh._people reads it back the same way as a live prompt_state)."""
    from . import matte_refresh as MR
    s = MR.prompt_state(tracker)
    if s is None:
        return None
    kp = s["final"].astype(np.float32).copy()
    if s["vis"] is not None:
        from .store import VIS_VISIBLE
        kp[s["vis"] != VIS_VISIBLE] = np.nan
    return dict(kp=kp, present=s["present"].astype(bool).copy(), boxes=s["boxes"].astype(np.float32).copy())


def load_prompts(tracker) -> Optional[dict]:
    """{'final', 'vis' (None), 'present', 'boxes', 'known'}: what each frame's matte was made from ('known' False:
    not recorded for that frame). Cached on the tracker by the file's time and size. None: nothing recorded."""
    f = prompts_path(tracker)
    try:
        st = f.stat()
    except OSError:
        tracker.__dict__.pop("_matte_prompts", None)
        return None
    key = (st.st_mtime_ns, st.st_size)
    c = tracker.__dict__.get("_matte_prompts")
    if c is not None and c[0] == key:
        return c[1]
    try:
        with np.load(f, allow_pickle=False) as d:
            out = dict(final=d["kp"].copy(), vis=None, present=d["present"].copy(), boxes=d["boxes"].copy(),
                       known=d["known"].copy())
    except Exception:
        return None
    n = getattr(getattr(tracker, "store", None), "n", None)
    if n is not None and out["final"].shape[1] != n:
        return None
    tracker.__dict__["_matte_prompts"] = (key, out)
    return out


def save_prompts(tracker, frames: Optional[Sequence[int]] = None, unknown: Optional[Sequence[int]] = None) -> bool:
    """Records the present prompts as what the matte of these frames (all frames: None) was made from. Rows of
    other frames stay as they were recorded. unknown: frames whose matte is known not to match (still marked)."""
    now = _prompt_rows(tracker)
    if now is None:
        return False
    P, n = now["present"].shape
    old = load_prompts(tracker)
    if frames is None or old is None:
        kp, pres, boxes = now["kp"], now["present"], now["boxes"]
        known = np.zeros(n, bool)
        if frames is None:
            known[:] = True
        else:
            known[np.asarray(sorted(frames), int)] = True
    else:
        Po = old["present"].shape[0]
        Pm = max(P, Po)
        kp = np.full((Pm, n) + now["kp"].shape[2:], np.nan, np.float32)
        pres = np.zeros((Pm, n), bool)
        boxes = np.full((Pm, n, 4), np.nan, np.float32)
        kp[:Po], pres[:Po], boxes[:Po] = old["final"], old["present"], old["boxes"]
        known = old["known"].copy()
        ts = np.asarray(sorted(frames), int)
        kp[:, ts], pres[:, ts], boxes[:, ts] = np.nan, False, np.nan
        kp[:P, ts], pres[:P, ts], boxes[:P, ts] = now["kp"][:, ts], now["present"][:, ts], now["boxes"][:, ts]
        known[ts] = True
    if unknown is not None and len(unknown):
        known[np.asarray(sorted(unknown), int)] = False
    return _write_prompts(tracker, kp, pres, boxes, known)


def _write_prompts(tracker, kp, pres, boxes, known) -> bool:
    f = prompts_path(tracker)
    tmp = f.with_name("matte_prompts.tmp.npz")
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, kp=kp, present=pres, boxes=boxes, known=known)
        os.replace(tmp, f)
    except OSError:
        _unlink(tmp)
        return False
    tracker.__dict__.pop("_matte_prompts", None)
    return True


def forget_person(tracker, p: int) -> None:
    """A person (store index) is about to be deleted: taken out of what the matte was made from as well, frame by
    frame the recorded person nearest to where this one is. A deletion does not make the matte again (as
    before), and the frames it was in must not count as changed for ever after."""
    rec = load_prompts(tracker)
    st = getattr(tracker, "store", None)
    if rec is None or st is None:
        return
    kp, pres, boxes = rec["final"].copy(), rec["present"].copy(), rec["boxes"].copy()
    hit = False
    for t in np.nonzero(st.present[p])[0]:
        cand = np.nonzero(pres[:, t])[0]
        if not len(cand):
            continue
        b = st.boxes[p, t].astype(np.float64)
        if np.isfinite(b).all():
            c = np.array([b[0] + b[2], b[1] + b[3]]) * 0.5
            rb = boxes[cand, t].astype(np.float64)
            d = np.hypot((rb[:, 0] + rb[:, 2]) * 0.5 - c[0], (rb[:, 1] + rb[:, 3]) * 0.5 - c[1])
        else:
            c = np.nanmean(st.final[p, t, :, :2].astype(np.float64), 0)
            d = np.hypot(*(np.nanmean(kp[cand, t].astype(np.float64), 1) - c).T)
        d = np.where(np.isfinite(d), d, np.inf)
        if not np.isfinite(d).any():
            continue
        k = cand[int(np.argmin(d))]
        pres[k, t] = False
        kp[k, t] = np.nan
        boxes[k, t] = np.nan
        hit = True
    if hit:
        _write_prompts(tracker, kp, pres, boxes, rec["known"])


def ensure_prompts(tracker) -> bool:
    """A cut whose matte was made before this was recorded (1.22.2 and older): the tracking as it is now is taken
    as what the matte was made from, except in the frames an edit has marked for a refresh. True: there is a
    record now."""
    if getattr(tracker, "store", None) is None or not tracker.has_seg():
        return False
    if load_prompts(tracker) is not None:
        return True
    marks = (getattr(tracker.cut, "status", None) or {}).get("matte_dirty") or []
    unknown = sorted({t for a, b in marks for t in range(int(a), int(b))})
    return save_prompts(tracker, None, [t for t in unknown if 0 <= t < tracker.store.n])


# ---------------------------------------------------------------------- some frames: a refresh after an edit
def stage(tracker, matte: Dict[int, np.ndarray], labels: Dict[int, np.ndarray], ranges,
          cancel: Optional[Callable[[], bool]] = None, models: Optional[dict] = None) -> dict:
    """The rest of a refresh's revision, finished before anything is written: the widths and outline points
    measured on the new alpha and owners, the frames whose owners really change, and the clothing of those frames
    cut with the new owners (for the people who keep clothes). Nothing in use is read for writing or changed."""
    out: dict = {}
    measure = getattr(tracker, "measure_widths", None)
    staged = dict(matte=matte, labels=labels)
    if measure is not None and getattr(tracker, "store", None) is not None:
        w = None
        for a, b in ranges:
            w = measure(None, (int(a), int(b)), staged=staged, base=w)
        if w is not None:
            out["widths"] = w
    changed = []
    for t in sorted(labels):
        cur = tracker.seglab.get(t)
        if cur is None or not np.array_equal(np.asarray(cur), labels[t]):
            changed.append(int(t))
    out["owners_changed"] = changed
    if changed:
        try:
            from . import clothing as CL
            cl = CL.restage(tracker, changed, staged, cancel=cancel, models=models)
        except Exception as e:
            if type(e).__name__ == "Cancelled":
                raise
            import traceback
            traceback.print_exc()
            cl = dict(frames={}, stale=changed, error=str(e)[:200])
        if cl is not None:
            out["cloth"] = cl
    return out


def _ranges(ts: Sequence[int]) -> List[List[int]]:
    out: List[List[int]] = []
    for t in sorted(set(int(x) for x in ts)):
        if out and out[-1][1] == t:
            out[-1][1] = t + 1
        else:
            out.append([t, t + 1])
    return out


def _write_frames(store, frames: Dict[int, np.ndarray]) -> None:
    if not frames:
        return
    mm = store.open_rw()
    try:
        for t in sorted(frames):
            mm[t] = frames[t]
        mm.flush()
    finally:
        del mm
        store.close()


def _read_frames(store, ts: Sequence[int]) -> Optional[np.ndarray]:
    if not ts or not store.exists():
        return None
    out = np.zeros((len(ts), store.h, store.w), np.uint8)
    for i, t in enumerate(ts):
        f = store.get(t)
        if f is not None:
            out[i] = f
    store.close()
    return out


def activate(tracker, result: dict) -> List[List[int]]:
    """Puts a finished refresh (matte_refresh.compute) in use as the next revision: alpha, owners, widths, outlines
    and clothing of its frames together, then seg_meta. On any error everything it replaced is put back and the
    error goes on to the caller (the revision before is the one in use)."""
    if not result or not result.get("ranges"):
        return []
    folder = cache_dir(tracker)
    stg = result.get("stage") or {}
    with lock_for(folder):
        st = getattr(tracker, "store", None)
        ts = sorted(set(result["matte"]) | set(result["labels"]))
        cloth = stg.get("cloth") or {}
        cts = sorted(cloth.get("frames") or {})
        base = revision(tracker)
        rev = base + 1
        undo: dict = dict(rev=np.array(rev), base=np.array(base), pid=np.array(os.getpid()),
                          ts=np.array(ts, np.int64), cts=np.array(cts, np.int64))
        om, ol = _read_frames(tracker.matte, ts), _read_frames(tracker.seglab, ts)
        if om is not None:
            undo["matte"] = om
        if ol is not None:
            undo["labels"] = ol
        has_cloth = hasattr(tracker, "cloth")
        if has_cloth and cts:
            oc = _read_frames(tracker.cloth, cts)
            if oc is not None:
                undo["cloth"] = oc
        cloth_state = None
        if has_cloth and (cts or cloth.get("stale")):
            from . import clothing as CL
            cloth_state = CL.state_text(tracker)
            undo["cloth_state"] = np.array(cloth_state if cloth_state is not None else "")
            undo["cloth_state_had"] = np.array(cloth_state is not None)
        w = stg.get("widths")
        keep_arrays = st is not None and all(hasattr(st, k) for k in ("widths", "contour_raw", "contour"))
        if keep_arrays:
            undo["widths"], undo["contour_raw"], undo["contour"] = st.widths.copy(), st.contour_raw.copy(), \
                st.contour.copy()
            undo["has_contour"] = np.array(bool(getattr(st, "has_contour", False)))
        undo["store_rev"] = np.array(int(getattr(st, "matte_rev", 0) or 0))
        jf = folder / UNDO
        tmp = folder / "matte_undo.tmp.npz"
        try:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **undo)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp, jf)
        except BaseException:
            _unlink(tmp)
            raise
        state = dict(saved=False)
        try:
            # alpha and owners of a frame one right after the other
            mm = tracker.matte.open_rw()
            lm = tracker.seglab.open_rw()
            try:
                for t in ts:
                    if t in result["matte"]:
                        mm[t] = result["matte"][t]
                    if t in result["labels"]:
                        lm[t] = result["labels"][t]
                mm.flush()
                lm.flush()
            finally:
                del mm, lm
                tracker.matte.close()
                tracker.seglab.close()
            _forget_derived(tracker, result["ranges"])
            if w is not None and keep_arrays:
                for p in w["persons"]:
                    st.widths[p], st.contour_raw[p], st.contour[p] = w["widths"][p], w["contour_raw"][p], \
                        w["contour"][p]
                st.has_contour = True
            else:
                for a, b in result["ranges"]:
                    tracker.update_widths(None, (a, b))
            if has_cloth and (cts or cloth.get("stale")):
                from . import clothing as CL
                _write_frames(tracker.cloth, cloth.get("frames") or {})
                CL.set_stale(tracker, add=_ranges(cloth.get("stale") or []), done=_ranges(cts))
            if st is not None and hasattr(st, "matte_rev"):
                st.matte_rev = rev
            tracker.save()
            state["saved"] = True            # (a save that raised replaced nothing: the tracks file is written
            #                                   whole beside the one in use and moved over it)
            _meta(tracker, result.get("model", ""), result["ranges"], False, rev)      # from here it is in use
        except BaseException as e:
            try:
                _undo_frames(tracker, dict(undo), in_memory=True, saved=state["saved"])
            except Exception as e2:
                # what was replaced could not be put back now either (the disk is still failing): the undo file
                # stays, and the cut is put back on the revision before when it is opened again
                raise ActivationError(f"인물 매트를 적용하지도 되돌리지도 못했습니다. 컷을 다시 열면 이전 매트로 "
                                      f"복구합니다: {e2}") from e
            raise
        try:
            save_prompts(tracker, [t for a, b in result["ranges"] for t in range(a, b)])
        except Exception:
            import traceback
            traceback.print_exc()
        _unlink(jf)
    return [list(r) for r in result["ranges"]]


def _forget_derived(tracker, ranges) -> None:
    """What was cut out of the old owners and is kept in memory: label maps, heads (hair), outlines, skin models."""
    try:
        for a, b in ranges:
            tracker.invalidate_labels(range(int(a), int(b)))
    except Exception:
        pass
    tracker._geo_cache = {}
    tracker._head_raw = {}
    tracker._skin_models = {}


def _undo_frames(tracker, undo: dict, in_memory: bool, saved: bool) -> None:
    """Puts back what a refresh replaced (from the undo data), and removes the undo file once that is done."""
    folder = cache_dir(tracker)
    ts = [int(t) for t in undo["ts"]]
    cts = [int(t) for t in undo["cts"]] if "cts" in undo else []
    rev = int(undo["rev"])
    was = int(undo["store_rev"]) if "store_rev" in undo else int(undo["base"])
    if "matte" in undo:
        _write_frames(tracker.matte, {t: undo["matte"][i] for i, t in enumerate(ts)})
    if "labels" in undo:
        _write_frames(tracker.seglab, {t: undo["labels"][i] for i, t in enumerate(ts)})
    if "cloth" in undo and hasattr(tracker, "cloth"):
        _write_frames(tracker.cloth, {t: undo["cloth"][i] for i, t in enumerate(cts)})
    if "cloth_state" in undo and hasattr(tracker, "cloth"):
        from . import clothing as CL
        CL.put_state_text(tracker, str(undo["cloth_state"]) if bool(undo["cloth_state_had"]) else None)
    st = getattr(tracker, "store", None)
    # the widths and outlines: in memory when this run changed them; in the file only when the file is the one this
    # run saved (the app ended after the save: the file says which revision it was measured on)
    mine = in_memory or (st is not None and getattr(st, "matte_rev", None) == rev)
    if st is not None and "widths" in undo and mine and undo["widths"].shape == st.widths.shape:
        st.widths, st.contour_raw, st.contour = undo["widths"].copy(), undo["contour_raw"].copy(), \
            undo["contour"].copy()
        st.has_contour = bool(undo["has_contour"]) if "has_contour" in undo else st.has_contour
        st.matte_rev = was
        if saved or not in_memory:
            tracker.save()
    elif st is not None and in_memory and hasattr(st, "matte_rev"):
        st.matte_rev = was
        if saved:
            tracker.save()
    if ts:
        _forget_derived(tracker, _ranges(ts))
    tracker.__dict__.pop("_matte_prompts", None)
    _unlink(folder / UNDO)


def recover(tracker) -> str:
    """Called when a cut's tracker is made (its tracks are loaded): a refresh that was being put in use when the
    app ended is taken back, so the cut is on its last complete revision. '' (nothing to do), 'cleaned' (the
    refresh had gone through, only its undo file was left) or 'rolled_back'."""
    folder = cache_dir(tracker)
    jf = folder / UNDO
    _unlink(folder / "matte_undo.tmp.npz")
    if not jf.exists():
        return ""
    with lock_for(folder):
        if not jf.exists():
            return ""
        try:
            with np.load(jf, allow_pickle=False) as d:
                undo = {k: d[k] for k in d.files}
            pid = int(undo["pid"])
            rev = int(undo["rev"])
        except Exception:
            _unlink(jf)                     # never complete (it is renamed into place only when it is)
            return "cleaned"
        if pid != os.getpid() and _alive(pid) and time.time() - jf.stat().st_mtime < STALE_S:
            return ""                       # another running DIAP is putting it in use right now
        if revision(tracker) >= rev:
            _unlink(jf)
            return "cleaned"
        _undo_frames(tracker, undo, in_memory=False, saved=False)
        return "rolled_back"


# ---------------------------------------------------------------------- the whole matte: analysis, forced rebuild
def _stores(tracker) -> list:
    return [s for s in (tracker.matte, tracker.seglab) if hasattr(s, "begin_stage")]


def staging(tracker) -> bool:
    return getattr(tracker, "_stage_extra", None) is not None


def begin(tracker, copy: bool = True) -> None:
    """A whole new matte is about to be made: from here the tracker's matte and owner stores (and the face layer
    made on them) are written beside the ones in use. commit() puts them in use, drop() throws them away."""
    folder = cache_dir(tracker)
    folder.mkdir(parents=True, exist_ok=True)
    with lock_for(folder):
        _ACTIVE.add(_key(folder))
        _write_json(folder / MARK, dict(pid=os.getpid(), time=time.time()))
        for s in _stores(tracker):
            s.begin_stage(copy=copy)
        tracker._stage_extra = {}
        tracker._stage_meta = None


def stage_file(tracker, live: Path, write: bool = False) -> Path:
    """Where a file that belongs to the revision being made is (the face layer): written beside the one in use and
    moved in by commit(). Read from there once it has been written; until then (and when nothing is being
    staged) from the one in use."""
    live = Path(live)
    extra = getattr(tracker, "_stage_extra", None)
    if extra is None:
        return live
    sp = live.with_name(live.stem + ".stage" + live.suffix)
    if write:
        extra[str(live)] = sp
        return sp
    return sp if (str(live) in extra and sp.exists()) else live


def drop(tracker) -> None:
    """The new matte is not used (cancelled, failed): its files are deleted; nothing in use was touched."""
    folder = cache_dir(tracker)
    with lock_for(folder):
        for s in _stores(tracker):
            s.end_stage(drop=True)
        for live, sp in (getattr(tracker, "_stage_extra", None) or {}).items():
            _unlink(sp)
            _unlink(Path(sp).with_name(Path(sp).stem + ".tmp" + Path(sp).suffix))
        tracker._stage_extra = None
        tracker._stage_meta = None
        tracker.__dict__.pop("_facelayer", None)
        if hasattr(tracker, "path"):
            _unlink(Path(tracker.path).with_name("tracks.stage.npz"))
        _unlink(folder / MARK)
        _ACTIVE.discard(_key(folder))


def _retry(fn, tries: int = 40, wait: float = 0.05):
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def _copy_into(src: Path, dst: Path) -> None:
    """dst's content replaced by src's without replacing the file (Windows: a file someone has mapped cannot be
    renamed over; it can be written)."""
    size = src.stat().st_size
    with open(src, "rb") as fi, open(dst, "r+b") as fo:
        if dst.stat().st_size != size:
            fo.truncate(size)               # (never for a mapped file: the stores of one cut have one size)
        shutil.copyfileobj(fi, fo, 8 << 20)
        fo.flush()
        try:
            os.fsync(fo.fileno())
        except OSError:
            pass


def _move_in(live: Path, new: Optional[Path]) -> None:
    """The file in use goes to <name>.old and new takes its place (new None: the revision has no such file)."""
    old = live.with_name(live.name + OLD)
    _unlink(old)
    if live.exists():
        try:
            _retry(lambda: os.replace(live, old))
        except PermissionError:
            # someone else has it mapped: its content is kept beside it and replaced in place
            shutil.copyfile(live, old)
            if new is not None:
                _copy_into(new, live)
                _unlink(new)
            else:
                with open(live, "r+b") as fo:          # cannot be removed while mapped: emptied of owners
                    fo.write(b"\0" * live.stat().st_size)
            return
    if new is not None:
        _retry(lambda: os.replace(new, live))


def _move_back(live: Path, had: bool, new_was: bool, stage_path: Optional[Path]) -> None:
    old = live.with_name(live.name + OLD)
    if old.exists():
        try:
            _retry(lambda: os.replace(old, live))
        except PermissionError:
            if live.exists():
                _copy_into(old, live)
                _unlink(old)
            else:
                raise
    elif not had and new_was and live.exists() and (stage_path is None or not stage_path.exists()):
        try:
            _retry(lambda: live.unlink())           # the new one was moved in where there was none
        except OSError:
            pass
    if stage_path is not None:
        _unlink(stage_path)


def _roll_back_swap(folder: Path, j: dict) -> None:
    for e in j.get("files", []):
        live = Path(e["dir"]) / e["name"] if e.get("dir") else folder / e["name"]
        sp = Path(e["stage"]) if e.get("stage") else None
        _move_back(live, bool(e.get("had")), bool(e.get("new")), sp)
    _unlink(folder / SWAP)
    _unlink(folder / MARK)


def _clean_swap(folder: Path, j: dict) -> None:
    for e in j.get("files", []):
        live = Path(e["dir"]) / e["name"] if e.get("dir") else folder / e["name"]
        _unlink(live.with_name(live.name + OLD))
        if e.get("stage"):
            _unlink(Path(e["stage"]))
    _unlink(folder / SWAP)
    _unlink(folder / MARK)


def commit(tracker, rng: Sequence[int], marks_done: bool = True) -> int:
    """Puts the staged matte, owners and face layer in use together with the tracks (their widths and outlines) as
    the next revision. Raises (ActivationError) with everything as it was before when a file cannot be moved."""
    folder = cache_dir(tracker)
    a, b = int(rng[0]), int(rng[1])
    with lock_for(folder):
        base = revision(tracker)
        rev = base + 1
        st = tracker.store
        if st is not None and hasattr(st, "matte_rev"):
            st.matte_rev = rev
        tpath = Path(tracker.path)
        tstage = tpath.with_name("tracks.stage.npz")
        files = []
        try:
            if st is not None:
                st.save(tstage)
            for s in _stores(tracker):
                s.close()
                files.append(dict(name=s.live_path.name, had=s.live_path.exists(), new=s.stage_path.exists(),
                                  stage=str(s.stage_path)))
            for live, sp in (getattr(tracker, "_stage_extra", None) or {}).items():
                if Path(sp).exists():
                    files.append(dict(name=Path(live).name, dir=str(Path(live).parent), had=Path(live).exists(),
                                      new=True, stage=str(sp)))
            if st is not None:
                files.append(dict(name=tpath.name, dir=str(tpath.parent), had=tpath.exists(), new=True,
                                  stage=str(tstage)))
            j = dict(rev=rev, base=base, pid=os.getpid(), time=time.time(), files=files)
            _write_json(folder / SWAP, j)
        except BaseException:
            _unlink(tstage)
            if st is not None and hasattr(st, "matte_rev"):
                st.matte_rev = base
            raise
        try:
            for e in files:
                live = Path(e["dir"]) / e["name"] if e.get("dir") else folder / e["name"]
                _move_in(live, Path(e["stage"]) if e["new"] else None)
            for s in _stores(tracker):
                s.end_stage(drop=False)
            tracker._stage_extra = None
            tracker.__dict__.pop("_facelayer", None)
            meta = getattr(tracker, "_stage_meta", None) or {}
            _meta(tracker, meta.get("model", ""), [[a, b]], True, rev)                 # from here it is in use
        except BaseException as e:
            try:
                _roll_back_swap(folder, j)
            finally:
                for s in _stores(tracker):
                    s.end_stage(drop=True)
                tracker._stage_extra = None
                tracker.__dict__.pop("_facelayer", None)
                _ACTIVE.discard(_key(folder))
                if st is not None and hasattr(st, "matte_rev"):
                    st.matte_rev = base
            if isinstance(e, Exception):
                raise ActivationError(f"새 인물 매트를 적용하지 못해 이전 것을 그대로 씁니다: {e}") from e
            raise
        _clean_swap(folder, j)
        _ACTIVE.discard(_key(folder))
        tracker._stage_meta = None
        try:
            tracker._label_cache.clear()
        except Exception:
            pass
        if marks_done:
            try:
                tracker._matte_marks_done(a, b)
            except Exception:
                pass
        try:
            # what this matte was made from; frames outside the range keep what was recorded for them
            save_prompts(tracker, None if (a <= 0 and b >= st.n) else range(a, b)) if st is not None else None
        except Exception:
            import traceback
            traceback.print_exc()
        return rev


def recover_files(folder: Path, tracks_path: Path, read_meta: Callable[[], dict]) -> str:
    """Before a cut's tracks are loaded: a whole new matte that was being moved in when the app ended is moved
    back out (or, when it had gone through, its leftovers are removed); stage files nobody is writing any more
    are deleted. '' / 'cleaned' / 'rolled_back'."""
    folder = Path(folder)
    jf, mk = folder / SWAP, folder / MARK
    if not jf.exists() and not mk.exists():
        return ""
    with lock_for(folder):
        out = ""
        j = _read_json(jf) if jf.exists() else None
        if jf.exists() and j is None:
            _unlink(jf)
            out = "cleaned"
        if j is not None:
            pid = int(j.get("pid", 0) or 0)
            if pid != os.getpid() and _alive(pid) and time.time() - float(j.get("time", 0) or 0) < STALE_S:
                return ""
            try:
                cur = int((read_meta() or {}).get("revision", 0) or 0)
            except Exception:
                cur = 0
            if cur >= int(j.get("rev", 0)):
                _clean_swap(folder, j)
                return "cleaned"
            _roll_back_swap(folder, j)
            return "rolled_back"
        m = _read_json(mk) if mk.exists() else None
        if mk.exists():
            pid = int((m or {}).get("pid", 0) or 0)
            if m is None or not _alive(pid) or (pid == os.getpid() and _key(folder) not in _ACTIVE):
                # stage files of an analysis that ended without using them
                for f in list(folder.glob("*.u8.stage")) + list(folder.glob("*.stage.npz")):
                    _unlink(f)
                _unlink(Path(tracks_path).with_name("tracks.stage.npz"))
                _unlink(mk)
                out = out or "cleaned"
        return out
