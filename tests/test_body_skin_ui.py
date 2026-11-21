"""1.23 body skin, the operator's side (instruction 2026-11-20 §2 조작): the switches, strengths, regions and hand marks
of 몸 피부 보정 and 몸 잡티 제거 in the panel and the viewer; that they are off for a new person and leave an unused
project's file as it was; that they are saved and opened, undone and redone, kept in presets and copied between cuts
and people."""
import json
import shutil

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QMessageBox

import body_util as B


# ------------------------------------------------------------------------------------------------- the values
def test_new_person_has_both_effects_off_and_nothing_of_them_in_the_file():
    from contour.core.project import PartParams
    pp = PartParams()
    assert not pp.body_on("body_skin") and not pp.body_on("body_blemish")
    assert pp.body_scope == "all" and pp.body_detail == 60.0 and pp.body_blemish_size == 40.0 and pp.body_marks == []
    d = pp.to_dict()
    assert not [k for k in d if k.startswith("body_") and k not in ("body_sculpt", "body_sides")]
    assert d["amounts"]["body_skin"]["base"] == 0 and d["amounts"]["body_blemish"]["base"] == 0


def test_switch_off_keeps_the_values_and_keys_to_come_back_to():
    from contour.core.params import BODY_START
    from contour.core.project import Curve, PartParams
    pp = PartParams()
    pp.set_body_on("body_skin", True)
    assert pp.amounts["body_skin"].base == BODY_START["body_skin"] and pp.body_on("body_skin")
    pp.amounts["body_skin"] = Curve(base=70.0, keys={3: 20.0, 9: 90.0})
    pp.set_body_on("body_skin", False)
    assert not pp.body_on("body_skin") and pp.amount("body_skin", 5) == 0.0
    again = PartParams.from_dict(json.loads(json.dumps(pp.to_dict())))          # (saved while it is off)
    again.set_body_on("body_skin", True)
    c = again.amounts["body_skin"]
    assert c.base == 70.0 and {int(k): v for k, v in c.keys.items()} == {3: 20.0, 9: 90.0} and not again.body_memo
    # switched on with nothing to come back to (a memo of zeros): the start value, not an effect that is on at 0
    z = PartParams()
    z.body_memo = {"body_blemish": Curve(base=0.0).to_dict()}
    z.set_body_on("body_blemish", True)
    assert z.amounts["body_blemish"].base == BODY_START["body_blemish"]


def test_saved_and_opened_again(tmp_path):
    from conftest import make_video
    from contour.core import media as M
    from contour.core.project import Curve, Person, Project
    prj = Project.create(str(tmp_path), "p")
    cut = prj.add_cut(M.probe(str(make_video(tmp_path / "a.mp4"))), "c1")
    cut.persons = [Person(id=1, name="A", color="#ff8800", role="target"), Person(id=2, name="B", color="#00aaff", role="target")]
    pp = cut.params_for(1)
    pp.amounts["body_skin"] = Curve(base=65.0, keys={2: 30.0})
    pp.amounts["body_blemish"] = Curve(base=40.0)
    pp.body_detail, pp.body_blemish_size, pp.body_scope = 25.0, 80.0, "leg,arm,arm,nose"
    pp.body_marks = [dict(id="m1", kind="remove", t=2, pts=[[10.0, 12.0], [14.0, 12.5]], r=6.0, part="forearm_l",
                          uv=[[0.4, 0.02], [0.45, 0.02]], ru=0.03),
                     dict(id="m2", kind="exclude", t=1, pts=[[30.0, 20.0]], r=4.0),
                     dict(id="m3", kind="paint", t=1, pts=[[1.0, 1.0]], r=4.0)]            # not a kind: dropped
    cut.params_for(2)
    prj.save()
    cut2 = Project.load(prj.path).cuts[0]
    q = cut2.params_for(1)
    assert q.amounts["body_skin"].base == 65.0 and q.amount("body_skin", 2) == 30.0 and q.amounts["body_blemish"].base == 40.0
    assert (q.body_detail, q.body_blemish_size, q.body_scope) == (25.0, 80.0, "arm,leg")
    assert [m["id"] for m in q.body_marks] == ["m1", "m2"]
    m1 = q.body_marks[0]
    assert m1["kind"] == "remove" and m1["part"] == "forearm_l" and m1["pts"] == [[10.0, 12.0], [14.0, 12.5]]
    assert m1["uv"] == [[0.4, 0.02], [0.45, 0.02]] and abs(m1["ru"] - 0.03) < 1e-9 and m1["r"] == 6.0 and m1["t"] == 2
    assert not q.body_marks[1].get("part")
    # the person who never used them: not a word of them in the file
    raw = json.loads((tmp_path / "p.dpprj" / "project.json").read_text(encoding="utf-8")) if (tmp_path / "p.dpprj").exists() \
        else json.loads(next(tmp_path.glob("*/project.json")).read_text(encoding="utf-8"))
    other = raw["cuts"][0]["params"]["2"]
    assert not [k for k in other if k.startswith("body_") and k not in ("body_sculpt", "body_sides")]


def test_preset_keeps_strengths_texture_size_and_regions(home):
    from contour.core import presets as PS
    from contour.core.project import Curve, PartParams
    src = PartParams()
    src.amounts["body_skin"] = Curve(base=10.0, keys={2: 70.0, 6: 30.0})
    src.amounts["body_blemish"] = Curve(base=35.0)
    src.body_detail, src.body_blemish_size, src.body_scope = 20.0, 75.0, "arm,leg"
    src.amounts["skin"] = Curve(base=44.0)
    assert "bodyskin" in PS.active_groups(src)
    d = PS.make(src, 2, "몸", ["bodyskin"], False)
    PS.save(d)
    got = next(x for x in PS.list_presets() if x["name"] == "몸")
    dst = PartParams()
    dst.amounts["skin"] = Curve(base=12.0)
    dst.body_memo = {"body_skin": Curve(base=99.0).to_dict()}
    dst.body_marks = [dict(id="k", kind="exclude", t=0, pts=[[1.0, 2.0]], r=3.0)]
    PS.apply(dst, got)
    assert dst.amounts["body_skin"].base == 70.0 and not dst.amounts["body_skin"].keys, "a preset is one value: the frame's"
    assert dst.amounts["body_blemish"].base == 35.0
    assert (dst.body_detail, dst.body_blemish_size, dst.body_scope) == (20.0, 75.0, "arm,leg")
    assert dst.amounts["skin"].base == 12.0, "the face's group was not in the preset"
    assert dst.body_memo == {} and len(dst.body_marks) == 1, "the marks are the cut's own, a preset does not touch them"
    # a preset without the group leaves the body's values alone
    PS.apply(dst, PS.make(PartParams(), 0, "얼굴", ["skin"], False))
    assert dst.amounts["body_skin"].base == 70.0 and dst.body_scope == "arm,leg"


def test_copy_between_cuts_takes_the_values_and_leaves_the_marks(tmp_path):
    from conftest import make_video
    from contour.core import copyset as CS
    from contour.core import media as M
    from contour.core.project import Curve, Person, Project
    prj = Project.create(str(tmp_path), "p")
    a = prj.add_cut(M.probe(str(make_video(tmp_path / "a.mp4"))), "a")
    b = prj.add_cut(M.probe(str(make_video(tmp_path / "b.mp4"))), "b")
    for c in (a, b):
        c.persons = [Person(id=1, name="A", color="#ff8800", role="target")]
    pa = a.params_for(1)
    pa.amounts["body_skin"] = Curve(base=60.0, keys={1: 20.0, 4: 80.0})
    pa.amounts["body_blemish"] = Curve(base=45.0)
    pa.set_body_on("body_blemish", False)
    pa.body_detail, pa.body_blemish_size, pa.body_scope = 10.0, 90.0, "torso"
    pa.body_marks = [dict(id="m", kind="remove", t=1, pts=[[5.0, 5.0]], r=4.0)]
    pb = b.params_for(1)
    pb.body_marks = [dict(id="own", kind="exclude", t=0, pts=[[9.0, 9.0]], r=4.0)]
    CS.apply_copy(a, b, {1: 1}, 4, CS.CopyOptions())
    pb = b.params_for(1)
    assert pb.amounts["body_skin"].base == 80.0 and not pb.amounts["body_skin"].keys, "the value at the copied frame"
    assert not pb.body_on("body_blemish") and pb.body_memo["body_blemish"]["base"] == 45.0, "off, with what it comes back to"
    assert (pb.body_detail, pb.body_blemish_size, pb.body_scope) == (10.0, 90.0, "torso")
    assert [m["id"] for m in pb.body_marks] == ["own"], "marks belong to the cut they were drawn on"


# ------------------------------------------------------------------------------------------------- the window
@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("bodyui")
    mp = pytest.MonkeyPatch()
    (d / "home").mkdir()
    mp.setenv("HOME", str(d / "home"))
    mp.setenv("USERPROFILE", str(d / "home"))
    import contour.core.settings as S
    S._settings = None
    try:
        prj, cut = B.make_project(d)
    finally:
        mp.undo()
        S._settings = None
    return prj.path


@pytest.fixture
def win(qapp, monkeypatch, scene, tmp_path):
    from contour.core import frames as FR
    from contour.core.settings import settings
    from contour.ui import main_window as MW
    settings().ask_proxy = False
    settings().proxy_long_side = B.AW
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    B.truth_parser(monkeypatch)
    FR.pool().clear()
    path = tmp_path / "body.dpprj"
    shutil.copytree(scene, path)
    w = MW.MainWindow()
    w.toasts = []
    real = w.toast
    monkeypatch.setattr(w, "toast", lambda text, kind="": (w.toasts.append(text), real(text, kind)))
    monkeypatch.setattr(w, "show_error", lambda title, text: None)
    w.resize(1400, 900)
    w.show()
    w.open_project(str(path), wait=True)
    qapp.processEvents()
    w.select_person(0)
    yield w
    w.proxy_builder.cancel()
    w.thumbcache.stop()
    w.session.dirty = False
    w._discard_on_close = True
    w.close()


def _pp(w):
    return w.session.cut.params_for(0)


def test_panel_switches_strengths_regions_and_undo(win, qapp):
    page = win.inspector.pages["skin"]
    page.refresh()
    tg, row = page.body_toggles["body_skin"], page.body_rows["body_skin"]
    assert not tg.isChecked() and not row.isEnabled() and not page.body_detail.isEnabled()
    assert not page.body_scope_boxes["all"].isEnabled(), "regions wait for an effect to be on"
    tg.setChecked(True)
    assert _pp(win).body_on("body_skin") and _pp(win).amounts["body_skin"].base == 50.0
    assert row.isEnabled() and page.body_detail.isEnabled() and not page.body_size.isEnabled()
    assert page.body_scope_boxes["all"].isChecked() and page.body_scope_boxes["all"].isEnabled()
    win.param_changed("body_skin", 80.0, True)
    win.set_body("detail", 30.0, True)
    assert _pp(win).amounts["body_skin"].base == 80.0 and _pp(win).body_detail == 30.0
    # regions: a region alone, two of them, back to 전신
    page.body_scope_boxes["arm"].setChecked(True)
    assert _pp(win).body_scope == "arm" and not page.body_scope_boxes["all"].isChecked()
    page.body_scope_boxes["leg"].setChecked(True)
    assert _pp(win).body_scope == "arm,leg"
    page.body_scope_boxes["arm"].setChecked(False)
    page.body_scope_boxes["leg"].setChecked(False)
    assert _pp(win).body_scope == "all" and page.body_scope_boxes["all"].isChecked(), "never nothing: 전신 again"
    page.body_scope_boxes["torso"].setChecked(True)
    assert _pp(win).body_scope == "torso"
    # off keeps the strength, on brings it back; the panel shows it meanwhile
    tg.setChecked(False)
    assert not _pp(win).body_on("body_skin") and _pp(win).body_memo["body_skin"]["base"] == 80.0
    assert abs(row.slider.value() - 80.0) < 1e-6 and not row.isEnabled()
    tg.setChecked(True)
    assert _pp(win).amounts["body_skin"].base == 80.0
    # undo and redo, step by step
    win.undo()
    assert not _pp(win).body_on("body_skin")
    win.undo()
    assert _pp(win).body_on("body_skin") and _pp(win).body_scope == "torso"
    win.undo()
    assert _pp(win).body_scope == "all"
    win.redo()
    win.redo()
    win.redo()
    assert _pp(win).amounts["body_skin"].base == 80.0 and _pp(win).body_scope == "torso" and _pp(win).body_detail == 30.0
    page.refresh()
    assert tg.isChecked() and page.body_scope_boxes["torso"].isChecked() and abs(page.body_detail.slider.value() - 30.0) < 1e-6


def test_the_preview_follows_the_switch(win, qapp):
    """The picture the viewer shows changes on the person's skin when 몸 피부 보정 is switched on, and is the
    original's again when it is switched off."""
    win.set_frame(2)
    s = win.session
    base = np.asarray(s.proxy.frame(2)).copy()
    assert s.retouch(2) is None
    win.inspector.pages["skin"].body_toggles["body_skin"].setChecked(True)
    win.param_changed("body_skin", 100.0, True)
    rf = s.retouch(2)
    assert rf is not None and rf.jobs and rf.jobs[0].body
    from contour.retouch import engine as RE
    out = RE.apply_rgb(base.copy(), rf, s.proxy.sx)
    k = B.kind(2, (s.proxy.w, s.proxy.h))
    ch = np.abs(out.astype(np.int16) - base.astype(np.int16)).max(-1)
    assert int((ch[k == B.SKIN_K] > 0).sum()) > 100 and int(ch[k == B.CLOTH].max()) == 0
    win.inspector.pages["skin"].body_toggles["body_skin"].setChecked(False)
    assert s.retouch(2) is None


def _click(v, pos, button=Qt.LeftButton, move_to=None):
    def ev(kind, p, btn, btns):
        return QMouseEvent(kind, QPointF(p), QPointF(p), btn, btns, Qt.NoModifier)
    v.mousePressEvent(ev(QEvent.MouseButtonPress, pos, button, button))
    if move_to is not None:
        for a in np.linspace(0, 1, 6)[1:]:
            q = QPointF(pos.x() + (move_to.x() - pos.x()) * a, pos.y() + (move_to.y() - pos.y()) * a)
            v.mouseMoveEvent(ev(QEvent.MouseMove, q, Qt.NoButton, button))
        pos = move_to
    v.mouseReleaseEvent(ev(QEvent.MouseButtonRelease, pos, button, Qt.NoButton))


def test_marks_are_drawn_in_the_viewer_follow_the_body_and_are_undone(win, qapp):
    from contour.retouch import bodyskin as BS
    page = win.inspector.pages["skin"]
    v = win.viewer
    win.set_frame(1)
    t = int(win.frame)
    assert t == 1
    # the tool waits for 몸 잡티 제거
    assert not page.body_mark_btns["remove"].isEnabled()
    win.set_body_mark_mode("remove")
    assert v.tool != "bodymark" and any("몸 잡티 제거를 먼저" in x for x in win.toasts)
    page.body_toggles["body_blemish"].setChecked(True)
    assert page.body_mark_btns["remove"].isEnabled() and page.body_size.isEnabled()
    page.body_mark_btns["remove"].setChecked(True)
    assert v.tool == "bodymark" and v.mark_kind == "remove" and win.ov["skin"].isChecked()
    win.set_body_mark_radius(5.0)
    # a dot on the mole
    mole = B.local(B.EL, B.WR, B.MOLE, t)
    _click(v, v.src_to_w(float(mole[0]), float(mole[1])))
    marks = _pp(win).body_marks
    assert len(marks) == 1 and marks[0]["kind"] == "remove" and marks[0]["part"] == "forearm_l" and marks[0]["r"] == 5.0
    assert np.hypot(*(np.asarray(marks[0]["pts"][0]) - mole)) < 2.5, "the dot is where it was clicked"
    st = win.session.tracker.store
    for s in (0, 6):
        got = BS.marks_at(st, 0, s, marks)
        assert np.hypot(*(got[0][1][0] - B.local(B.EL, B.WR, B.MOLE, s))) < 2.5, f"frame {s}: the mark left the mole"
    # a stroke that is kept from removal
    page.body_mark_btns["exclude"].setChecked(True)
    assert v.mark_kind == "exclude" and not page.body_mark_btns["remove"].isChecked()
    a, b = B.local(B.EL, B.WR, (0.55, 0.0), t), B.local(B.EL, B.WR, (0.68, 0.0), t)
    _click(v, v.src_to_w(float(a[0]), float(a[1])), move_to=v.src_to_w(float(b[0]), float(b[1])))
    marks = _pp(win).body_marks
    assert len(marks) == 2 and marks[1]["kind"] == "exclude" and len(marks[1]["pts"]) >= 2
    assert "2" in page.body_mark_info.text()
    # undo, redo, right click on a mark, clear
    win.undo()
    assert len(_pp(win).body_marks) == 1
    win.redo()
    assert len(_pp(win).body_marks) == 2
    _click(v, v.src_to_w(float(mole[0]), float(mole[1])), button=Qt.RightButton)
    assert [m["kind"] for m in _pp(win).body_marks] == ["exclude"]
    win.clear_body_marks()
    assert _pp(win).body_marks == [] and not page.body_mark_btns["clear"].isEnabled()
    win.undo()
    assert len(_pp(win).body_marks) == 1
    # a click beside the body is fixed to the picture and said so; Esc and switching the effect off end the tool
    win.toasts.clear()
    _click(v, v.src_to_w(900.0, 690.0))
    assert not _pp(win).body_marks[-1].get("part") and any("같은 자리에 고정" in x for x in win.toasts)
    win.set_tool("edit")
    assert v.tool == "edit" and not page.body_mark_btns["exclude"].isChecked()
    page.body_mark_btns["remove"].setChecked(True)
    assert v.tool == "bodymark"
    page.body_toggles["body_blemish"].setChecked(False)
    assert v.tool == "edit" and not page.body_mark_btns["remove"].isEnabled()


def test_overlay_shows_the_area_and_what_was_found(win, qapp):
    """K: the body's area and its spots (removed, and found but left) are drawn; with nothing on, the overlay is
    what it was."""
    from contour.ui import viewer as V
    v = win.viewer
    page = win.inspector.pages["skin"]
    win.set_frame(2)
    page.body_toggles["body_skin"].setChecked(True)
    page.body_toggles["body_blemish"].setChecked(True)
    win.set_body("blemish_size", B.SPOT_SIZE, True)
    win.ov["skin"].setChecked(True)
    key, t, rgba = v._compute_skin((v._skin_key(), int(win.frame), 0))
    assert rgba is not None and rgba.shape[:2] == (win.session.proxy.h, win.session.proxy.w)
    kinds = {k for k, _ in v._det_kinds[1]}
    assert {"body", "bspot", "bheld"} <= kinds, kinds
    # the area's colour lies on the person's skin and not on the top; the spots that go are marked where they are
    k = B.kind(int(win.frame), (win.session.proxy.w, win.session.proxy.h))
    inside = cv2.erode((k == B.SKIN_K).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert float((rgba[..., 3][inside] > 60).mean()) > 0.95
    assert float((rgba[..., 3][cv2.erode((k == B.CLOTH).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] > 60).mean()) < 0.01
    q = B.local(B.EL, B.WR, B.PIMPLES[0], int(win.frame)) * win.session.proxy.sx
    red = np.asarray(V.DET_COLOURS["bspot"], np.float32)
    near = rgba[int(q[1]) - 2:int(q[1]) + 3, int(q[0]) - 2:int(q[0]) + 3, :3].reshape(-1, 3).astype(np.float32)
    assert float(np.abs(near - red).sum(1).min()) < 40, "the pimple that goes is marked in the spots' colour"
    img = v.grab().toImage()
    assert img.width() > 0
    # with the marks painted while the tool is on
    win.set_body_mark_mode("exclude")
    q = B.local(B.EL, B.WR, B.PIMPLES[0], int(win.frame))
    _click(v, v.src_to_w(float(q[0]), float(q[1])))
    assert len(v._marks_now()) == 1
    v.grab()


def test_several_people_selected_get_the_same_switch_and_regions(win, qapp):
    cut = win.session.cut
    cut.person(1).role = "target"
    win.select_person(0)
    win.sel_pids = {0, 1}
    assert len(win._others()) == 1
    win.set_body_on("body_skin", True)
    win.set_body("scope", "leg", True)
    win.set_body("detail", 15.0, True)
    for pid in (0, 1):
        pp = cut.params_for(pid)
        assert pp.body_on("body_skin") and pp.body_scope == "leg" and pp.body_detail == 15.0
    win.set_body_on("body_skin", False)
    assert not cut.params_for(0).body_on("body_skin") and not cut.params_for(1).body_on("body_skin")


# ------------------------------------------------------------------------------------------------- skin-coloured clothing
def _wait_job(w, qapp, s=30.0):
    import time
    t0 = time.time()
    while w.job is not None and time.time() - t0 < s:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert w.job is None


def test_switching_a_body_effect_on_looks_at_the_cut_for_skin_coloured_clothing(win, qapp, monkeypatch):
    """몸 피부 보정 switched on: the cut is looked at with the garment words in the job panel, and the box says what
    came of it. (The scene's parser and the garment words agree here: nothing to leave out.)"""
    from contour.track import clothing as CL, models as M, skincloth as SC
    from contour.ui.inspector import body_cloth_line
    s = win.session
    seen = []

    class Finder:
        stats = dict(frames=0)

    def one(gf, st, proxy, pres, t, frame, lab, rgb, mw, mh):
        seen.append(int(t))
        k = B.kind(int(t), (mw, mh))
        for p in pres:
            frame[lab == p] = 1
            frame[(lab == p) & np.isin(k, (B.CLOTH, B.STRAP))] = 5
        return frame
    monkeypatch.setattr(CL, "GarmentFinder", Finder)
    monkeypatch.setattr(CL, "_garment_frame", one)
    page = win.inspector.pages["skin"]
    page.refresh()
    assert body_cloth_line(s.tracker, s.cut, False) == ("", False), "nothing is said while no body effect is on"
    # SAM 3.1 is not on this computer: said once, nothing is started, the box says the look is still to come
    page.body_toggles["body_skin"].setChecked(True)
    assert win.job is None and any("SAM 3.1" in t for t in win.toasts)
    page.refresh()
    assert "확인 전" in page.body_cloth_info.text() and page.body_cloth_btn.isEnabled()
    n_told = sum("SAM 3.1" in t for t in win.toasts)
    page.body_toggles["body_blemish"].setChecked(True)
    assert sum("SAM 3.1" in t for t in win.toasts) == n_told, "not said again with every switch"
    # with the models there: the button (and the switch) start the look
    monkeypatch.setattr(SC, "models_missing", lambda: [])
    page.body_cloth_btn.click()
    assert win.job is not None
    _wait_job(win, qapp)
    assert sorted(seen) == [0, B.N - 1] and SC.info(s.tracker)["state"] == "current"
    assert any("피부색 옷 확인 끝" in t and "없습니다" in t for t in win.toasts)
    page.refresh()
    assert "확인됨" in page.body_cloth_info.text() and not page.body_cloth_btn.isEnabled()
    # nothing more to do: switching the other effect off and on starts nothing
    seen.clear()
    page.body_toggles["body_blemish"].setChecked(False)
    page.body_toggles["body_blemish"].setChecked(True)
    assert win.job is None and seen == []
