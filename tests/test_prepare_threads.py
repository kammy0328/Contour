"""The people of a frame made side by side (engine.PREP_THREADS, j2815): what comes out is what one after the other gives, bit for bit.

A render's masks of one picture were made one person after the other on one core, the one thing the render did on a single core. Each person's masks are
made from the picture, the tracks and caches kept under that person's own keys, so they can be made on several threads; these tests hold that the
numbers do not change: the prepared jobs (every mask, every map) and the planes a render hands to the encoder are the same with 1, 2 and 4 threads, and
that the few things the people share (the frame cache's book-keeping, the colour models kept on the tracker, the owners' small cache) stay whole when
many threads use them at once."""
import copy
import threading
from pathlib import Path

import numpy as np
import pytest

import baseline_util as BU

FIX = Path(__file__).resolve().parent / "data" / "baseline_1222"
FIX_BODY = Path(__file__).resolve().parent / "data" / "baseline_1222_body"


def two_targets(prj):
    """The fixture retouches person 0 only (person 1 is protected): the second person gets the first one's values too, so that a frame has two jobs."""
    cut = prj.cuts[0]
    cut.persons[1].role = "target"
    p0, p1 = cut.params_for(0), cut.params_for(1)
    p1.amounts = {k: copy.deepcopy(v) for k, v in p0.amounts.items()}
    p1.skin_basic = copy.deepcopy(p0.skin_basic)
    p1.skin_scope = p0.skin_scope
    assert len(cut.targets()) == 2
    return cut


def differences(a, b, path="", out=None, depth=0):
    """Where two prepared things differ (paths of the first few): arrays by value (NaN equal), containers and objects by their parts."""
    out = [] if out is None else out
    if len(out) >= 8 or depth > 12:
        return out
    if type(a) is not type(b):
        out.append(f"{path}: {type(a).__name__} / {type(b).__name__}")
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or a.dtype != b.dtype:
            out.append(f"{path}: {a.shape}{a.dtype} / {b.shape}{b.dtype}")
        elif a.dtype.kind in "fc":
            if not np.array_equal(a, b, equal_nan=True):
                out.append(f"{path}: {int((a != b).sum())} values differ, max {float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))):.3g}")
        elif not np.array_equal(a, b):
            out.append(f"{path}: {int((a != b).sum())} values differ")
    elif isinstance(a, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(map(str, set(a) ^ set(b)))}")
        for k in a:
            if k in b:
                differences(a[k], b[k], f"{path}[{k!r}]", out, depth + 1)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} / {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                differences(x, y, f"{path}[{i}]", out, depth + 1)
    elif isinstance(a, float):
        if not (a == b or (a != a and b != b)):
            out.append(f"{path}: {a!r} / {b!r}")
    elif hasattr(a, "__dict__") and not callable(a):
        differences(vars(a), vars(b), path, out, depth + 1)
    elif a != b:
        out.append(f"{path}: {a!r} / {b!r}")
    return out


def prepared(pdir, threads, monkeypatch, with_two=True):
    """Every frame of the fixture prepared (_prepare_raw, an empty cache each time) with this many threads, on a tracker of its own."""
    from contour.core.frames import frames_for
    from contour.core.project import Project
    from contour.retouch import engine as RE
    from contour.retouch import skin as SK
    from contour.track.analyze import CutTracker
    monkeypatch.setattr(RE, "PREP_THREADS", threads)
    prj = Project.load(str(pdir))
    cut = two_targets(prj) if with_two else prj.cuts[0]
    BU.steady_rng()
    fr = frames_for(prj, cut)
    tr = CutTracker(prj, cut, fr)
    try:
        with BU.NoModels():
            out = []
            for t in range(BU.N):
                # (the chroma planes a thread keeps for the frame it is on are found by the frame's address and a sample of its pixels: on these
                # tiny frames, which hardly change from one to the next, one thread's planes of frame 2 can pass for frame 3's. A thread that
                # starts clean on every frame (what side by side does) gets frame 3's own: so does the reference here)
                SK._PLANE_TLS.__dict__.clear()
                out.append(RE._prepare_raw(cut, tr, t, {}))
            return out
    finally:
        fr.close()


@pytest.mark.parametrize("fixture", [FIX, FIX_BODY], ids=["face", "body regions"])
@pytest.mark.parametrize("threads", [2, 4])
def test_the_masks_of_a_frame_are_the_same_side_by_side(tmp_path, monkeypatch, fixture, threads):
    pdir, _src = BU.copy_fixture(fixture, tmp_path)
    one = prepared(pdir, 1, monkeypatch)
    many = prepared(pdir, threads, monkeypatch)
    assert any(rf is not None and len(rf.jobs) == 2 for rf in one), "the scene gives frames with two people's jobs"
    for t, (a, b) in enumerate(zip(one, many)):
        assert (a is None) == (b is None), f"frame {t}"
        if a is not None:
            assert [j.pid for j in a.jobs] == [j.pid for j in b.jobs], f"frame {t}: the people come in the same order"
            d = differences(a, b, f"frame {t}")
            assert not d, "; ".join(d)


def test_the_threads_are_used_for_two_people_and_not_for_one(tmp_path, monkeypatch):
    from contour.retouch import engine as RE
    pdir, _src = BU.copy_fixture(FIX, tmp_path)
    calls = []
    real = RE._prime
    monkeypatch.setattr(RE, "_prime", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    prepared(pdir, 4, monkeypatch, with_two=True)
    assert len(calls) == BU.N, "two people: every frame made side by side"
    calls.clear()
    prepared(pdir, 4, monkeypatch, with_two=False)
    assert not calls, "one person: nothing to make side by side"
    prepared(pdir, 1, monkeypatch, with_two=True)
    assert not calls, "1 thread: one after the other, as ever"


def test_a_failure_of_one_person_stops_the_frame_as_it_did(tmp_path, monkeypatch):
    from contour.retouch import engine as RE
    from contour.core.frames import frames_for
    from contour.core.project import Project
    from contour.track.analyze import CutTracker
    pdir, _src = BU.copy_fixture(FIX, tmp_path)
    monkeypatch.setattr(RE, "PREP_THREADS", 3)
    prj = Project.load(str(pdir))
    cut = two_targets(prj)
    real = RE.SK.SkinJob

    class Boom(Exception):
        pass

    def bad(p, *a, **k):
        if p == 1:
            raise Boom("person 1")
        return real(p, *a, **k)
    monkeypatch.setattr(RE.SK, "SkinJob", bad)
    fr = frames_for(prj, cut)
    tr = CutTracker(prj, cut, fr)
    try:
        with BU.NoModels(), pytest.raises(Boom):
            RE._prepare_raw(cut, tr, 2, {})
    finally:
        fr.close()


def test_the_planes_of_a_render_are_the_same_side_by_side(tmp_path, monkeypatch):
    from contour.core.project import Project
    from contour.retouch import engine as RE
    pdir, _src = BU.copy_fixture(FIX, tmp_path)
    got = {}
    for threads in (1, 2, 4):
        monkeypatch.setattr(RE, "PREP_THREADS", threads)
        prj = Project.load(str(pdir))
        two_targets(prj)
        with BU.NoModels():
            got[threads] = BU.render_planes(prj, prj.cuts[0])
    for threads in (2, 4):
        d = np.abs(got[threads].astype(np.int32) - got[1].astype(np.int32))
        assert d.max() == 0, f"{threads} threads: {int((d > 0).sum())} values differ, max {int(d.max())}"
    base = Project.load(str(pdir))
    with BU.NoModels():
        single = BU.render_planes(base, base.cuts[0])
    assert not np.array_equal(single, got[1]), "the second person's retouch is in the picture (the test would not see a difference otherwise)"


def test_the_frame_cache_keeps_its_count_with_many_threads(monkeypatch):
    from contour.retouch import engine as RE
    monkeypatch.setattr(RE, "CACHE_BYTES", 6 * 2 ** 20)
    cache: dict = {}
    errors = []

    def work(k):
        try:
            for i in range(400):
                key = (i, k, "m", "")
                RE._cache_put(cache, key, np.zeros((256, 1024), np.uint8), tok=(k, i % 3))        # 256 kB each
                RE._cache_look(cache, (i - 2, k, "m", ""), (k, (i - 2) % 3))
        except BaseException as e:                       # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:3]
    real = sum(int(v.nbytes) for k, v in cache.items() if k not in RE._CACHE_OWN)
    assert cache["__bytes__"] == real <= RE.CACHE_BYTES
    assert set(cache["__tok__"]) <= {k for k in cache if k not in RE._CACHE_OWN}


def test_the_colour_models_on_the_tracker_stay_whole_with_many_threads(tmp_path, monkeypatch):
    from contour.core.frames import frames_for
    from contour.core.project import Project
    from contour.retouch import skin as SK
    from contour.track.analyze import CutTracker
    pdir, _src = BU.copy_fixture(FIX, tmp_path)
    prj = Project.load(str(pdir))
    cut = prj.cuts[0]
    fr = frames_for(prj, cut)
    tr = CutTracker(prj, cut, fr)
    px = tr.proxy
    rgb = np.asarray(px.frame(0))
    lm = tr.store.face[0, 0].copy()
    lm[:, 0] *= px.sx
    lm[:, 1] *= px.sy
    errors = []
    got = {}

    def work(k):
        try:
            for i in range(300):
                t = k * 1000 + i                                       # (more than the 600 the cache keeps)
                got[(k, i)] = SK.frame_model_for(tr, k % 2, t, rgb, lm)
            SK.model_for(tr, 0)
        except BaseException as e:                                     # noqa: BLE001
            errors.append(repr(e))
    try:
        ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    finally:
        fr.close()
    assert not errors, errors[:3]
    assert len(tr._skin_fmodels) <= 600
    one = [got[(0, i)][0] for i in range(300)]
    assert all(np.array_equal(one[0], x) for x in one), "the same face, the same colour model"
