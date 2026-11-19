"""Instruction 2026-11-19 §6: 'AI 검수 자료 만들기' (and python -m contour.review) reads an analysed project as it is
saved and writes report.md, review.json and pictures under <project>/ai_review/<time>/: per cut its id, name,
timecodes, people, analysis state and saved QC, and findings (low confidence, missing joints, spikes, an id jump,
a person missing from the matte, a matte refresh still due) each with frames, timecodes, evidence and a
suggestion, told apart by where they come from (measured here, saved QC, a person's decision). No model runs,
nothing in the project changes; a copy of the project, a missing original or missing stores still give what can
be read. Before 1.22.2 there was nothing to hand an analysed project to for review."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import make_video

N, W, H = 40, 64, 48
MW, MH = 32, 24


def _digest(folder: Path, skip="ai_review"):
    out = {}
    for f in sorted(folder.rglob("*")):
        if f.is_file() and skip not in f.relative_to(folder).parts:
            out[f.relative_to(folder).as_posix()] = hashlib.md5(f.read_bytes()).hexdigest()
    return out


def _analysed(tmp_path, name="rv"):
    from contour.core import media as M
    from contour.core.project import Person, Project, QCIssue
    from contour.track.store import TrackStore
    from contour.track import skeleton as S
    v = tmp_path / f"{name}.mp4"
    make_video(v, n=N, w=W, h=H)
    prj = Project.create(str(tmp_path), name)
    cut = prj.add_cut(M.probe(str(v)), "c1")
    cut.info.timecode = "01:00:00:00"
    cut.analyzed = True
    cut.analysis_version = 9
    cut.persons = [Person(id=0, name="A", color="#ff8800", role="target"),
                   Person(id=1, name="B", color="#00aaff", role="protect")]
    cut.params_for(0).amounts["waist"].base = 20.0
    cut.qc = [QCIssue("warp", 5, 8, "error", "변형량 과다", person=0),
              QCIssue("jitter", 20, 22, "warn", "떨림", person=0, status="ignored")]
    cut.confirmed = {1: [[30, 35]]}
    cut.joint_keys = {0: {S.L_WR: {12: [20.0, 20.0]}}}
    cut.status = dict(auto="good", user="good", reviewed=True, matte_dirty=[[2, 4]])
    st = TrackStore(N, [0, 1])
    st.has_vis = st.has_face = True
    st.present[:] = True
    base = np.array([[32, 6], [33, 5], [31, 5], [34, 6], [30, 6], [36, 12], [28, 12], [38, 20], [26, 20],
                     [39, 27], [25, 27], [35, 28], [29, 28], [35, 37], [29, 37], [35, 45], [29, 45]], np.float32)
    for p, dx in ((0, -14.0), (1, 14.0)):
        for t in range(N):
            st.final[p, t, :, :2] = base + [dx, 0]
            st.final[p, t, :, 2] = 0.9
            x0, x1 = 24 + dx, 41 + dx
            st.boxes[p, t] = [x0, 3, x1, 46]
    st.scale[:] = 44.0
    st.conf[:] = 0.9
    st.conf[0, 10:16] = 0.2                                   # unsure tracking (and A is retouched there)
    st.final[0, 25, S.R_SH, :2] += [9.0, 0.0]                 # a one-frame spike on a shoulder
    st.final[1, 18:21, S.L_HIP, :] = np.nan                   # a hip without any estimate
    st.boxes[1, 33:] += [-26, 0, -26, 0]                      # B's box jumps (inside a range B was confirmed)
    st.final[1, 33:, :, 0] -= 26
    d = prj.cut_dir(cut)
    st.save(d / "tracks.npz")
    cache = prj.cache_dir(cut)
    mm = np.memmap(cache / f"matte_{MW}x{MH}.u8", np.uint8, "w+", shape=(N, MH, MW))
    lm = np.memmap(cache / f"seglab_{MW}x{MH}.u8", np.uint8, "w+", shape=(N, MH, MW))
    mm[:, 2:23, 4:13] = 255
    lm[:, 2:23, 4:13] = 1                                     # A owns its pixels everywhere
    mm[:, 2:23, 19:28] = 255
    lm[:, 2:23, 19:28] = 2
    lm[26:30, 2:23, 19:28] = 0                                # B missing from the owners on 26..29
    mm.flush()
    lm.flush()
    del mm, lm
    (cache / "seg_meta.json").write_text(json.dumps(dict(model="SAM 3.1", made=1.0, refreshed=[])), encoding="utf-8")
    prj.save()
    return prj, cut, v


@pytest.fixture
def no_models(monkeypatch):
    """Any model being opened fails the test."""
    import contour.track.infer as IN
    import contour.track.models as MOD

    def boom(*a, **k):
        raise AssertionError("a model was opened")
    for name in ("Detector", "Segmenter", "Segmenter31", "Finder31", "Matter", "make_segmenter"):
        if hasattr(IN, name):
            monkeypatch.setattr(IN, name, boom)
    monkeypatch.setattr(MOD, "session", boom, raising=False)


def _kinds(c):
    return {f["kind"] for f in c["findings"]}


def test_report_json_and_pictures_without_models_or_changes(tmp_path, no_models):
    from contour import review as RV
    prj, cut, v = _analysed(tmp_path)
    before = _digest(Path(prj.path))
    out = RV.build(prj.path)
    assert out.parent == Path(prj.path) / "ai_review"
    assert _digest(Path(prj.path)) == before, "nothing in the project changed"
    rev = json.loads((out / "review.json").read_text(encoding="utf-8"))
    assert rev["format"] == "diap-ai-review" and rev["models_run"] is False
    c = rev["cuts"][0]
    assert c["id"] == cut.id and c["name"] == "c1" and c["work"]["tc_in"] == "01:00:00:00"
    assert c["work"]["frame_out"] == N and c["work"]["tc_out"] == cut.info.tc_at(N - 1)
    assert [p["id"] for p in c["persons"]] == [0, 1] and c["persons"][0]["retouch"] is True
    by = {}
    for f in c["findings"]:
        by.setdefault(f["kind"], []).append(f)
        assert f["source"] in ("auto", "qc", "user") and f["evidence"] and f["suggest"]
    low = by["lowconf"][0]
    assert (low["person"], low["frame_in"], low["frame_out"]) == (0, 10, 16) and low["render_risk"] is True
    assert low["tc_in"] == cut.info.tc_at(10)
    assert any(f["person"] == 1 and f["frame_in"] == 18 and f["frame_out"] == 21 for f in by["missing_joints"])
    assert any(f["person"] == 0 and f["frame_in"] <= 25 < f["frame_out"] for f in by["jitter"])
    jump = by["id_jump"][0]
    assert jump["person"] == 1 and jump["frame_in"] == 33 and jump.get("inside_user_confirmed") is True
    lost = by["matte_lost"][0]
    assert lost["person"] == 1 and (lost["frame_in"], lost["frame_out"]) == (26, 30)
    assert by["matte_pending"][0]["frame_in"] == 2
    assert c["caches"]["matte_model"] == "SAM 3.1"
    # saved QC and what a person decided, told apart
    qc = {q["kind"]: q for q in c["qc"]}
    assert qc["qc_warp"]["source"] == "qc" and qc["qc_warp"]["status"] == "open"
    assert qc["qc_jitter"]["source"] == "user"
    us = {u["kind"] for u in c["user"]}
    assert {"status", "reviewed", "confirmed", "joint_edits"} <= us
    # pictures linked relative to the review, drawn on the original
    imgs = [f for f in c["findings"] if f.get("image")]
    assert imgs and all((out / f["image"]).exists() for f in imgs) and not any(f.get("image_base") for f in imgs)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "c1" in md and "추적 신뢰도 낮음" in md and "사람이 확인한 내용" in md and imgs[0]["image"] in md


def test_copy_missing_original_and_missing_stores(tmp_path, no_models):
    from contour import review as RV
    prj, cut, v = _analysed(tmp_path)
    cp = tmp_path / "copy" / Path(prj.path).name
    shutil.copytree(prj.path, cp)
    v.unlink()                                                 # the original is gone
    out = RV.build(str(cp), out_dir=str(tmp_path / "rv_out"))
    c = json.loads((out / "review.json").read_text(encoding="utf-8"))["cuts"][0]
    assert c["id"] == cut.id and c["source"]["found"] is False
    assert "source_missing" in _kinds(c) and "lowconf" in _kinds(c)
    imgs = [f for f in c["findings"] if f.get("image")]
    assert imgs and all(f.get("image_base") == "saved_matte" for f in imgs)
    # owners gone: the RVM case; then no tracks at all
    for f in (cp / "cuts" / cut.id / "cache").glob("seglab_*"):
        f.unlink()
    c = json.loads((RV.build(str(cp), images=False) / "review.json").read_text(encoding="utf-8"))["cuts"][0]
    assert "matte_rvm" in _kinds(c) and "matte_lost" not in _kinds(c)
    (cp / "cuts" / cut.id / "tracks.npz").unlink()
    c = json.loads((RV.build(str(cp), images=False) / "review.json").read_text(encoding="utf-8"))["cuts"][0]
    assert "tracks_missing" in _kinds(c) and c["counts"]["error"] >= 1 and c["qc"]
    assert not (cp / "cuts" / "nothing").exists()


def test_cli(tmp_path, no_models, capsys):
    from contour import review as RV
    prj, cut, v = _analysed(tmp_path)
    assert RV.main([prj.path, "--out", str(tmp_path / "o"), "--no-images"]) == 0
    assert "OUT" in capsys.readouterr().out
    assert (tmp_path / "o" / "report.md").exists() and not (tmp_path / "o" / "img").exists()


def test_menu_item_makes_it_from_the_saved_project(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from contour.core.settings import settings
    from contour.ui import main_window as MW
    import time
    settings().ask_proxy = False
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    prj, cut, v = _analysed(tmp_path)
    w = MW.MainWindow()
    try:
        monkeypatch.setattr(w, "show_error", lambda title, text: None)
        w.open_project(prj.path, wait=True)
        w.make_ai_review(open_folder=False)
        t0 = time.time()
        while w.job is not None and time.time() - t0 < 30:
            qapp.processEvents()
            time.sleep(0.01)
        out = Path(w.last_review)
        assert (out / "review.json").exists() and out.parent.name == "ai_review"
    finally:
        w.proxy_builder.cancel()
        w.thumbcache.stop()
        w.session.dirty = False
        w._discard_on_close = True
        w.close()
