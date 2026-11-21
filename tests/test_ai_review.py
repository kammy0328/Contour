"""Instruction 2026-11-19 §6: 'AI 검수 자료 만들기' (and python -m contour.review) reads an analysed project as it is
saved and writes report.md, review.json and pictures under <project>/ai_review/<time>/: per cut its id, name,
timecodes, people, analysis state and saved QC, and findings (low confidence, missing joints, spikes, an id jump,
a person missing from the matte, a matte refresh still due) each with frames, timecodes, evidence and a
suggestion, told apart by where they come from (measured here, saved QC, a person's decision). No model runs,
nothing in the project changes; a copy of the project, a missing original or missing stores still give what can
be read. Before 1.22.2 there was nothing to hand an analysed project to for review."""
import hashlib
import json
import re
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
    out = RV.build(prj.path, images=True)                     # (pictures are asked for: see the default report below)
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
    out = RV.build(str(cp), out_dir=str(tmp_path / "rv_out"), images=True)
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


def test_a_timeline_index_saved_as_a_float(tmp_path, no_models):
    """PC check on the Ballin project: a cut whose timeline place was saved as 3.0 stopped the whole review."""
    from contour import review as RV
    from contour.core.project import Project
    prj, cut, v = _analysed(tmp_path)
    p = Project.load(prj.path)
    p.cuts[0].tl = dict(index=3.0, kind="clip", rec_in=10, rec_out=50, base=0)
    p.timeline = dict(seq_start=0, fps=24.0, drop=False, name="t")
    p.save()
    c = json.loads((RV.build(prj.path, images=False) / "review.json").read_text(encoding="utf-8"))["cuts"][0]
    assert c["number"] == "004" and c["timeline"]["index"] == 3 and c["work"]["rec_tc_in"] == "00:00:00:10"


# ------------------------------------------------------------------ the default report (instruction 2026-11-20 §4-4)
def _strings(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def test_the_default_report_has_no_footage_and_no_media_path(tmp_path, no_models):
    """No picture of a frame, no crop, and no absolute path of a media file: a source by its name, the project by
    its folder name. (1.22.2 wrote img/ with frames, source.path and project.folder in full.)"""
    from contour import review as RV
    from contour.core.project import Project
    media = tmp_path / "미디어 원본" / "reel 01"
    media.mkdir(parents=True)
    prj, cut, v = _analysed(tmp_path)
    moved = media / "A001 클립.mp4"
    shutil.move(str(v), moved)
    p = Project.load(prj.path)                                 # the cut now points at a file outside the project
    p.cuts[0].source = str(moved)
    p.cuts[0].info.path = str(moved)
    p.cuts[0].retouch_ref = str(media / "A001 클립_after.mov")
    p.cuts[0].stats.last_output = str(tmp_path / "out" / "A001_reshape.mov")
    p.save()
    out = RV.build(prj.path)
    names = sorted(f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file())
    assert names == ["report.md", "review.json"], "nothing but the two text files (no img/, no crop)"
    rev = json.loads((out / "review.json").read_text(encoding="utf-8"))
    assert rev["contains"] == {"pictures": False, "crops": False, "media_paths": "name"}
    c = rev["cuts"][0]
    assert c["source"] == {"name": "A001 클립.mp4", "found": True} and rev["project"]["folder"] == Path(prj.path).name
    assert not any(f.get("image") for f in c["findings"])
    text = (out / "review.json").read_text(encoding="utf-8") + (out / "report.md").read_text(encoding="utf-8")
    for leak in (str(tmp_path), str(media), "미디어 원본", "reel 01", str(Path.home())):
        assert leak not in text, f"a path of this computer is in the report: {leak}"
    absolute = [x for x in _strings(rev) if re.match(r"^[A-Za-z]:[\\/]", x) or x.startswith("\\\\")
                or (x.startswith("/") and "/" in x[1:])]
    assert not absolute, absolute
    assert "영상·크롭이 없고" in (out / "report.md").read_text(encoding="utf-8")
    # a source inside the project folder is named by its place there
    inside = Path(prj.path) / "media" / "b.mp4"
    inside.parent.mkdir()
    shutil.copy(moved, inside)
    p = Project.load(prj.path)
    p.cuts[0].source = str(inside)
    p.cuts[0].info.path = str(inside)
    p.save()
    c = json.loads((RV.build(prj.path) / "review.json").read_text(encoding="utf-8"))["cuts"][0]
    assert c["source"]["name"] == "media/b.mp4"
    # the original gone: said with the name, still without a path
    moved.unlink()
    inside.unlink()
    out = RV.build(prj.path)
    c = json.loads((out / "review.json").read_text(encoding="utf-8"))["cuts"][0]
    f = next(f for f in c["findings"] if f["kind"] == "source_missing")
    assert f["evidence"] == "원본을 찾을 수 없음: media/b.mp4" and str(tmp_path) not in json.dumps(c, ensure_ascii=False)
    assert not (out / "img").exists(), "no picture drawn on the matte either unless asked for"


def test_the_report_says_state_failure_times_settings_and_summary(tmp_path, no_models, monkeypatch):
    from contour import review as RV
    from contour.core import media as M
    from contour.core.project import Project
    prj, cut, v = _analysed(tmp_path)
    p = Project.load(prj.path)
    a = p.cuts[0]
    a.stats.analysis_seconds, a.stats.analysis_runs, a.stats.active_seconds = 83.4, 2, 410.0
    a.stats.auto_ok_rate, a.stats.renders, a.stats.render_seconds, a.stats.corrections = 0.91, 1, 55.0, 3
    a.models_used = {"matte_rvm": "0123456789abcdef"}
    a.model_versions = {"matte_rvm": "2026.11.1"}
    make_video(tmp_path / "f.mp4")
    b = p.add_cut(M.probe(str(tmp_path / "f.mp4")), "c2")                      # an analysis that failed
    b.status = dict(auto="bad", reason="분석 실패: GPU 메모리가 부족합니다", failed=True)
    make_video(tmp_path / "e.mp4")
    e = p.add_cut(M.probe(str(tmp_path / "e.mp4")), "c3")                      # left out by the quick look
    e.status = dict(auto="none", reason="빠른 검사: 인물 없음", prescan=True, prescan_info=dict(verdict="none", looked=6))
    p.set_learn(True)
    p.save()
    (Path(p.cache_dir(a)) / "proxy_960x720.mp4").write_bytes(b"p" * 5000)
    out = RV.build(prj.path)
    rev = json.loads((out / "review.json").read_text(encoding="utf-8"))
    ca, cb, ce = rev["cuts"]
    # state of the analysis and of what is saved
    assert ca["analysis"]["analyzed"] and ca["analysis"]["models"] == {"matte_rvm": "0123456789abcdef"}
    assert ca["analysis"]["model_versions"] == {"matte_rvm": "2026.11.1"} and not ca["analysis"]["failed"]
    cs = ca["caches"]
    assert cs["tracks"] and cs["tracks_bytes"] > 0 and cs["matte"]["bytes"] == N * MW * MH and cs["matte"]["made"]
    assert cs["matte"]["file"] == f"cuts/{a.id}/cache/matte_{MW}x{MH}.u8" and cs["cloth"] is None
    assert cs["proxy"] == {"files": 1, "bytes": 5000}
    # why it failed, and the cut the quick look left out
    assert cb["analysis"]["failed"] and "GPU 메모리" in cb["analysis"]["failure"] and not cb["analysis"]["analyzed"]
    assert ce["analysis"]["prescan"]["verdict"] == "none" and not ce["analysis"]["failed"]
    # times
    t = ca["times"]
    assert (t["analysis_seconds"], t["analysis_runs"], t["worked_seconds"], t["renders"], t["render_seconds"],
            t["corrections"]) == (83.4, 2, 410.0, 1, 55.0, 3)
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$", t["analysed"]) and cb["times"]["analysed"] == ""
    assert ca["metrics"] == {"auto_ok_rate": 0.91, "persons": 2, "targets": 1, "work_frames": N}
    # settings and where it ran, without a path
    s = rev["settings"]
    assert s["accuracy"] and s["proxy_long_side"] and "system" in s and s["cache_folder"] == "기본"
    assert s["learning"] == {"project": True, "chosen": True, "collect": True, "self_learning": True}
    # summary figures
    m = rev["summary"]
    assert (m["cuts"], m["analysed"], m["failed"], m["skipped_prescan"]) == (3, 1, 1, 1)
    assert m["stores"]["matte"] == 1 and m["stores"]["cloth"] == 0 and m["model_versions"] == {"matte_rvm": {"2026.11.1": 1}}
    assert m["auto_ok_rate_mean"] == 0.91 and m["analysis_seconds"] == 83.4 and m["persons"] == 2 and m["cache_bytes"] > 0
    assert m["by_kind"]["lowconf"] >= 1 and sum(m["by_state"].values()) == 3
    md = (out / "report.md").read_text(encoding="utf-8")
    for word in ("## 요약 지표", "## 설정", "분석 실패 원인: 분석 실패: GPU 메모리가 부족합니다", "모델 패키지 버전: matte_rvm 2026.11.1",
                 "분석 실패 1컷", "프록시 파일 1개", "학습 데이터 수집·자동 학습: 이 프로젝트 켬"):
        assert word in md, word


def test_cli_and_menu_make_the_report_without_pictures_unless_asked(qapp, tmp_path, no_models, monkeypatch, capsys):
    from PySide6.QtWidgets import QMessageBox
    from contour import review as RV
    from contour.ui import main_window as MW_
    import time
    prj, cut, v = _analysed(tmp_path)
    assert RV.main([prj.path, "--out", str(tmp_path / "plain")]) == 0
    assert not (tmp_path / "plain" / "img").exists()
    assert RV.main([prj.path, "--out", str(tmp_path / "pics"), "--images"]) == 0
    assert list((tmp_path / "pics" / "img").glob("*.jpg"))
    rev = json.loads((tmp_path / "pics" / "review.json").read_text(encoding="utf-8"))
    assert rev["contains"]["pictures"] is True and "영상 프레임 그림" in (tmp_path / "pics" / "report.md").read_text(encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW_.MainWindow()
    try:
        monkeypatch.setattr(w, "show_error", lambda title, text: None)
        w.open_project(prj.path, wait=True)
        for images in (False, True):
            w.make_ai_review(open_folder=False, images=images)
            t0 = time.time()
            while w.job is not None and time.time() - t0 < 30:
                qapp.processEvents()
                time.sleep(0.01)
            assert (Path(w.last_review) / "img").exists() == images
    finally:
        w.proxy_builder.cancel()
        w.thumbcache.stop()
        w.session.dirty = False
        w._discard_on_close = True
        w.close()
