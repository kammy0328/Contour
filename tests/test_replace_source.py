"""(1.27, LNGSHOT BEEP BEEP 11/27) 소스 교체: a cut's pictures from another file of the same shot (a cleaned plate, a new export). The user:
"video2로 올라가 있는 17개의 컷은 추후에 배경에 클리닝이 진행된 컷으로 Replace 해야해. Replace한다고 해서 보정 값이 달라지면 안돼."
What was analysed and every value stay with the cut; only where its pictures are read from changes (core/replace.py)."""
import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import make_video


def _info(path="film.mov", frames=4609, tc="01:05:35:00", w=4096, h=2304, offset=0, fps=(24000, 1001)):
    from contour.core import media as MD
    return MD.MediaInfo(path=str(path), width=w, height=h, fps_num=fps[0], fps_den=fps[1], frames=frames, timecode=tc, offset=offset,
                        codec="prores", pix_fmt="yuv444p12le", duration=frames * fps[1] / fps[0])


def _cut(start=849, n=22, film=None):
    """A timeline cut: a stretch of the graded film (as core/tlimport makes them), with a person and values."""
    from contour.core.project import Cut, Curve, Person
    film = film or _info()
    ci = copy.deepcopy(film)
    ci.offset, ci.frames = start, n
    c = Cut(id="c018abcdef", name="c018", source=film.path, info=ci)
    c.tl = dict(index=17, kind="clip", rec_in=start, rec_out=start + n, base=start, name="A_0005C012", note="V2", src_in=3, speed=100.0)
    c.persons = [Person(id=0, name="인물 1", color="#fff", role="target"), Person(id=1, name="인물 2", color="#000", role="protect")]
    pp = c.params_for(0)
    pp.amounts["skin"] = Curve(base=70.0)
    pp.amounts["neck_lines"] = Curve(base=70.0, keys={3: 40.0})
    pp.skin_detail = 15.0
    c.analyzed, c.analysis_version, c.version = True, 1, 2
    return c


def _probe_of(table):
    def probe(p):
        k = Path(str(p)).name
        if k not in table:
            raise OSError("no such file " + k)
        return copy.deepcopy(table[k])
    return probe


def test_a_cleaned_cut_with_the_timelines_timecode_is_taken_from_its_first_frame():
    from contour.core import replace as RPL
    c = _cut(849, 22)                                                # the film's 01:05:35:00 + 849 frames = 01:06:10:09
    probe = _probe_of({"film.mov": _info(), "BEEP_c001.mov": _info("BEEP_c001.mov", 22, "01:06:10:09")})
    r = RPL.check(c, "BEEP_c001.mov", probe)
    assert r["ok"] and r["how"] == "타임코드" and r["offset"] == 0
    assert r["info"].path == "BEEP_c001.mov" and r["info"].frames == 22 and r["info"].offset == 0
    assert r["info"].tc_at(0) == c.info.tc_at(0) == "01:06:10:09", "the cut's timecode stays the same"


def test_a_new_export_of_the_whole_film_is_taken_at_the_cuts_place():
    from contour.core import replace as RPL
    c = _cut(849, 22)
    probe = _probe_of({"film.mov": _info(), "film_v2.mov": _info("film_v2.mov", 4609, "01:05:35:00"),
                       "film_notc.mov": _info("film_notc.mov", 4609, ""), "longer.mov": _info("longer.mov", 4700, "")})
    r = RPL.check(c, "film_v2.mov", probe)
    assert r["ok"] and r["how"] == "타임코드" and r["offset"] == 849 and r["info"].frames == 22
    r = RPL.check(c, "film_notc.mov", probe)                         # no timecode: as long as the file the cut reads now -> the same place
    assert r["ok"] and r["how"] == "같은 위치" and r["offset"] == 849
    r = RPL.check(c, "longer.mov", probe)
    assert not r["ok"] and "찾을 수 없습니다" in r["why"]


def test_a_file_as_long_as_the_cut_without_a_matching_timecode_is_taken_from_its_first_frame():
    from contour.core import replace as RPL
    c = _cut(849, 22)
    probe = _probe_of({"film.mov": _info(), "clean.mov": _info("clean.mov", 22, ""), "other_tc.mov": _info("other_tc.mov", 22, "00:00:00:00")})
    r = RPL.check(c, "clean.mov", probe)
    assert r["ok"] and r["how"] == "같은 길이" and r["offset"] == 0
    r = RPL.check(c, "other_tc.mov", probe)
    assert r["ok"] and r["how"] == "같은 길이" and "타임코드" in r["note"], "said: the timecode does not fit, taken by its length"


@pytest.mark.parametrize("name,new,word", [("small.mov", dict(w=3840, h=2160, frames=22), "크기"), ("fps.mov", dict(fps=(25, 1), frames=22), "프레임 속도"),
                                           ("short.mov", dict(frames=21, tc=""), "찾을 수 없습니다"), ("gone.mov", None, "읽지 못했습니다")])
def test_what_cannot_stand_for_the_cut_is_not_taken_and_said(name, new, word):
    from contour.core import replace as RPL
    c = _cut(849, 22)
    table = {"film.mov": _info()}
    if new is not None:
        table[name] = _info(name, **new)
    r = RPL.check(c, name, _probe_of(table))
    assert not r["ok"] and word in r["why"], r


def test_replacing_changes_where_the_pictures_come_from_and_nothing_else(tmp_path):
    from contour.core import replace as RPL
    from contour.core.project import Project
    from contour.core.proxy import SHARED
    prj = Project.create(str(tmp_path), "T")
    c = _cut(849, 22)
    prj.cuts.append(c)
    # the analysis files of the cut and the link to the timeline's shared proxy
    d = prj.cut_dir(c)
    (d / "tracks.npz").write_bytes(b"tracks")
    (prj.cache_dir(c) / "matte_1280x720.u8").write_bytes(b"m" * 64)
    (prj.cache_dir(c) / "facelayer.npz").write_bytes(b"face")
    pd = prj.proxy_dir(c)
    (pd / SHARED).write_text(json.dumps(dict(path="x", meta="y", offset=849)), encoding="utf-8")
    was = c.to_dict()
    r = RPL.check(c, "BEEP_c001.mov", _probe_of({"film.mov": _info(), "BEEP_c001.mov": _info("BEEP_c001.mov", 22, "01:06:10:09")}))
    RPL.apply(prj, c, r["info"], r["how"])
    now = c.to_dict()
    assert now["source"] == "BEEP_c001.mov" and now["info"]["path"] == "BEEP_c001.mov" and now["info"]["offset"] == 0 and now["info"]["frames"] == 22
    for k in was:
        if k not in ("source", "info"):
            assert now[k] == was[k], f"{k} changed with the source"
    assert now["params"] == was["params"] and now["persons"] == was["persons"], "the values and the people are the cut's"
    assert (d / "tracks.npz").read_bytes() == b"tracks" and (prj.cache_dir(c) / "facelayer.npz").exists() and (prj.cache_dir(c) / "matte_1280x720.u8").exists()
    assert not (pd / SHARED).exists(), "the link to the old film's proxy is gone"
    assert c.analyzed and c.version == 2 and c.tl["rec_in"] == 849
    # saved and read again
    prj.save()
    c2 = Project.load(prj.path).cut(c.id)
    assert c2.source == "BEEP_c001.mov" and c2.replaced["source"] == "film.mov" and c2.replaced["info"]["offset"] == 849
    assert c2.to_dict()["params"] == was["params"]
    # and back
    assert RPL.can_undo(c2)
    assert RPL.undo(prj, c2)
    assert c2.source == "film.mov" and c2.info.offset == 849 and c2.info.frames == 22 and c2.to_dict()["params"] == was["params"]
    assert RPL.can_undo(c2) and c2.replaced["source"] == "BEEP_c001.mov", "taking it back can be taken back"


def test_a_cut_never_replaced_has_nothing_to_take_back_and_its_file_says_nothing_of_it(tmp_path):
    from contour.core import replace as RPL
    from contour.core.project import Project
    c = _cut()
    assert not RPL.can_undo(c) and "replaced" not in c.to_dict()
    prj = Project.create(str(tmp_path), "T")
    assert not RPL.undo(prj, c)


def test_a_folder_gives_each_cut_the_file_with_its_timecode(tmp_path):
    from contour.core import replace as RPL
    film = _info(str(tmp_path / "film.mov"))
    cuts = [_cut(849, 22, film), _cut(871, 52, film), _cut(0, 70, film)]
    cuts[1].id, cuts[1].name = "c019", "c019"
    cuts[2].id, cuts[2].name = "c001", "c001"
    table = {"film.mov": film, "BEEP_c001_cln_v1.mov": _info("BEEP_c001_cln_v1.mov", 22, "01:06:10:09"),
             "BEEP_c001_cln_v2.mov": _info("BEEP_c001_cln_v2.mov", 22, "01:06:10:09"), "BEEP_c002.mov": _info("BEEP_c002.mov", 52, "01:06:11:07"),
             "whatever.mov": _info("whatever.mov", 22, ""), "small.mov": _info("small.mov", 22, "01:06:10:09", w=1920, h=1080)}
    folder = tmp_path / "fromCLN"
    folder.mkdir()
    for k, nm in enumerate(n for n in table if n != "film.mov"):
        (folder / nm).write_bytes(b"x")
        os.utime(folder / nm, (1000 + 10 * k, 1000 + 10 * k))        # (v2 is newer than v1)
    (folder / "notes.txt").write_text("x")
    res = RPL.find_in_folder(cuts, str(folder), _probe_of(table))
    by = {r["cut"].name: r for r in res}
    assert by["c018"]["ok"] and Path(by["c018"]["file"]).name == "BEEP_c001_cln_v2.mov", "the newest of the files of the same stretch"
    assert [Path(p).name for p in by["c018"]["others"]] == ["BEEP_c001_cln_v1.mov"]
    assert by["c019"]["ok"] and Path(by["c019"]["file"]).name == "BEEP_c002.mov" and by["c019"]["offset"] == 0
    assert not by["c001"]["ok"] and "없습니다" in by["c001"]["why"], "a cut no file of the folder holds stays as it is"
    # a cut without a timecode cannot be told apart in a folder
    c = _cut(849, 22, _info(str(tmp_path / "film.mov"), tc=""))
    r = RPL.find_in_folder([c], str(folder), _probe_of(table))[0]
    assert not r["ok"] and "타임코드" in r["why"]


def test_the_same_shot_cleaned_is_alike_and_another_shot_is_not():
    from contour.core import replace as RPL
    rng = np.random.default_rng(1)

    def pic(seed, clean=False):
        r = np.random.default_rng(seed)
        import cv2
        img = cv2.GaussianBlur(r.random((180, 320, 3)).astype(np.float32), (0, 0), 6) * 3.0
        img = np.clip(img - img.min(), 0, 1)
        if clean:
            img = img.copy()
            img[20:50, 30:90] = cv2.GaussianBlur(img, (0, 0), 9)[20:50, 30:90]       # a rig painted out of the background
            img = np.clip(img * 0.97 + 0.01, 0, 1)                                   # and a new encode
        return img
    c = _cut(0, 12)
    new = copy.deepcopy(c.info)
    new.path = "new.mov"
    same = RPL.similarity(c, new, read=lambda info, t: pic(100 + t, clean=info.path == "new.mov"))
    other = RPL.similarity(c, new, read=lambda info, t: pic((100 if info.path != "new.mov" else 500) + t))
    assert same is not None and same > 0.8, same
    assert other is not None and other < RPL.SIM_WARN, other

    def gone(info, t):
        raise OSError("the old film is not there")
    assert RPL.similarity(c, new, read=gone) is None, "cannot be told: not a reason to refuse"


def test_a_replaced_timeline_cut_no_longer_shares_the_films_proxy(tmp_path):
    from contour.core import replace as RPL
    from contour.core import tlimport as TLI
    from contour.core.project import Project
    film = make_video(tmp_path / "film.mp4", n=12)
    clean = make_video(tmp_path / "clean.mp4", n=4)
    from contour.core import media as MD
    fi = MD.probe(str(film))
    prj = Project.create(str(tmp_path), "T")
    cuts = []
    for k, (a, n) in enumerate(((0, 4), (4, 4), (8, 4))):
        ci = copy.deepcopy(fi)
        ci.offset, ci.frames = a, n
        c = prj.add_cut(ci, f"c{k + 1:03d}")
        c.tl = dict(index=k, kind="clip", rec_in=a, rec_out=a + n, base=a, name="", note="", src_in=-1, speed=100.0)
        cuts.append(c)
    prj.timeline = dict(baked=str(film), frames=12)
    assert all(TLI.reads_bake(c, str(film)) for c in cuts) and TLI.shares_bake(prj, cuts[1])
    r = RPL.check(cuts[1], str(clean))                               # (the real probe: a file as long as the cut)
    assert r["ok"] and r["how"] == "같은 길이", r
    sim = RPL.similarity(cuts[1], r["info"])                         # (the real reader)
    assert sim is not None
    RPL.apply(prj, cuts[1], r["info"], r["how"])
    assert not TLI.reads_bake(cuts[1], str(film)) and not TLI.shares_bake(prj, cuts[1])
    assert TLI.shares_bake(prj, cuts[0]) and TLI.shares_bake(prj, cuts[2])
    # its frames are the new file's
    from contour.core.frames import frames_for
    from contour.warp.render import _vs_for
    px = frames_for(prj, cuts[1], _vs_for(cuts[1]))
    assert px is not None and px.n == 4
    px.close()


def test_the_cut_menu_offers_it_and_the_window_replaces(tmp_path, monkeypatch):
    """Through the window's own handler (no dialogs: the path is given): the cut reads the new file, its values are the same, the project is saved."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    from contour.core import media as MD
    from contour.core.project import Project, Curve
    from contour.ui.main_window import MainWindow
    a = make_video(tmp_path / "a.mp4", n=6)
    b = make_video(tmp_path / "b.mp4", n=6)
    short = make_video(tmp_path / "short.mp4", n=5)
    prj = Project.create(str(tmp_path), "T")
    c = prj.add_cut(MD.probe(str(a)), "a")
    c.params_for(0).amounts["skin"] = Curve(base=70.0)
    c.params_for(0).skin_detail = 15.0
    prj.save()
    was = c.to_dict()["params"]
    toasts = []
    w = NS(session=NS(project=prj, cut=None, dirty=False), job=None, batch=NS(busy=False), thumbcache=NS(refresh=Mock()), cutlist=NS(update_rows=Mock()),
           tlstrip=NS(refresh=Mock(), set_current=Mock()), toast=lambda text, kind="": toasts.append((kind, text)), _reopen_cut=Mock(return_value=True))
    for name in ("_busy_with", "_replace_ready", "_replace_done"):
        setattr(w, name, getattr(MainWindow, name).__get__(w))
    assert not MainWindow.replace_cut_source(w, c.id, str(short), ask=False), "a file of another length is not taken"
    assert c.source == str(a) and toasts and toasts[-1][0] == "warn"
    assert MainWindow.replace_cut_source(w, c.id, str(b), ask=False)
    assert c.source == str(b) and c.info.path == str(b) and c.to_dict()["params"] == was
    assert Project.load(prj.path).cut(c.id).source == str(b), "saved"
    assert toasts[-1][0] == "ok" and "그대로" in toasts[-1][1]
    assert MainWindow.replace_cut_source_back(w, c.id)
    assert c.source == str(a) and Project.load(prj.path).cut(c.id).source == str(a)
    # while a job runs nothing is replaced
    w.job = object()
    assert not MainWindow.replace_cut_source(w, c.id, str(b), ask=False) and c.source == str(a)
