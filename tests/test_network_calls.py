"""1.27: fewer file-system calls where the project is on a network drive (the app's freeze log of 11/26 and j2890:
opening one cut made 46 folders that were there, read its tracks in 160 pieces; each a round trip, seconds when the
drive is busy)."""
import builtins
import io
from pathlib import Path

import numpy as np
import pytest


def test_a_cut_folder_is_made_once_and_made_again_when_it_is_gone(project, monkeypatch):
    from contour.core import project as PJ
    PJ.forget_dirs()
    made = []
    mk0 = Path.mkdir
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: (made.append(str(self)), mk0(self, *a, **k))[1])
    cut = project.cuts[0]
    d = project.cut_dir(cut)
    c = project.cache_dir(cut)
    assert d.is_dir() and c.is_dir() and c == d / "cache"
    assert str(d) in made and str(c) in made
    n = len(made)
    for _ in range(20):                                      # what opening a cut asks, again and again
        assert project.cut_dir(cut) == d and project.cache_dir(cut) == c
    assert len(made) == n
    # another cut: its own folders
    project.cache_dir(project.cuts[1])
    assert len(made) == n + 2
    # removed from outside: made again once the memory is old (or forgotten)
    import shutil
    shutil.rmtree(d)
    monkeypatch.setattr(PJ, "DIR_MEMO_S", 0.0)
    assert project.cache_dir(cut).is_dir()
    monkeypatch.setattr(PJ, "DIR_MEMO_S", 20.0)
    shutil.rmtree(d)
    PJ.forget_dirs()
    assert project.cut_dir(cut).is_dir()


def test_the_tracks_are_read_in_one_piece_and_are_what_was_saved(tmp_path, monkeypatch):
    from contour.track.store import TrackStore
    st = TrackStore(12, [3, 7])
    rng = np.random.default_rng(0)
    st.final[:] = rng.normal(size=st.final.shape).astype(st.final.dtype)
    st.present[:] = rng.random(st.present.shape) > 0.3
    st.raw["face_yunet"] = rng.normal(size=(2, 12, 5, 3)).astype(np.float32)
    st.removed[9] = np.arange(6, dtype=np.float32)
    st.has_face, st.matte_rev = True, 4
    p = tmp_path / "tracks.npz"
    st.save(p)
    opens = []
    open0 = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opens.append(str(f)), open0(f, *a, **k))[1])
    monkeypatch.setattr(io, "open", builtins.open)
    got = TrackStore.load(p)
    assert opens.count(str(p)) == 1                          # one open, one read: not one for every array
    assert got is not None and got.pids == [3, 7] and got.n == 12 and got.has_face and got.matte_rev == 4
    assert np.array_equal(got.final, st.final, equal_nan=True) and np.array_equal(got.present, st.present)
    assert np.array_equal(got.raw["face_yunet"], st.raw["face_yunet"]) and np.array_equal(got.removed[9], st.removed[9])
    assert got.final.flags.writeable and got.final.base is None or got.final.flags.owndata or got.final.flags.writeable
    got.final[0, 0] = 1.0                                    # the arrays are the store's own
    # no file, a file that is not one: None, as before
    assert TrackStore.load(tmp_path / "nothing.npz") is None
    (tmp_path / "bad.npz").write_bytes(b"not an archive")
    assert TrackStore.load(tmp_path / "bad.npz") is None
