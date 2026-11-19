"""Instruction 2026-11-19 §7: the 보호 tab, its drawing tools and the '분리 + 배경 복원' mode are gone; one warp with
the people who are not retouched always kept out of it. Older projects (separate mode, drawn lines and shapes,
other protection settings, 보호 people) open, preview and render with the warp, keep their data in the file and are
told once; a 보호 person stays out of the retouch."""
import subprocess
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from conftest import make_video


def _old_project(tmp_path, name="old"):
    from contour.core import media as M
    from contour.core.project import Person, Project, ProtectLine, ProtectShape
    v = tmp_path / f"{name}.mp4"
    make_video(v)
    prj = Project.create(str(tmp_path), name)
    cut = prj.add_cut(M.probe(str(v)), "c1")
    cut.mode = "separate"
    cut.protect.others = False
    cut.protect.reach = 2.2
    cut.protect.lines.append(ProtectLine(id="l1", keys={0: [1.0, 2.0, 30.0, 2.0]}))
    cut.protect.shapes.append(ProtectShape(id="s1", kind="ellipse", keys={0: [20.0, 20.0, 5.0, 5.0]}))
    cut.persons = [Person(id=1, name="A", color="#ff8800", role="target"), Person(id=2, name="B", color="#00aaff", role="protect")]
    prj.save()
    return prj


def test_old_data_is_kept_in_the_file_but_not_applied(tmp_path):
    from contour.core.project import Project
    from contour.warp import field as F
    prj = _old_project(tmp_path)
    again = Project.load(prj.path)
    c = again.cuts[0]
    d = c.to_dict()
    assert d["mode"] == "separate" and len(d["protect"]["lines"]) == 1 and len(d["protect"]["shapes"]) == 1
    assert d["protect"]["others"] is False and d["protect"]["reach"] == 2.2
    assert set(F.legacy_protection(c)) == {"분리 + 배경 복원 모드", "직선·영역 보호", "보호 설정"}
    fi = F.build_input(c, NS(store=None), 0)
    assert fi.protect_others is True and fi.reach == F.WARP_SETTINGS["reach"] and fi.border is True
    assert fi.falloff_pct == F.WARP_SETTINGS["falloff_pct"]
    assert not hasattr(fi, "lines") and not hasattr(fi, "shapes")
    assert [p.role for p in c.persons] == ["target", "protect"]
    assert [p.id for p in c.targets()] == [1]


def test_old_separate_cut_renders_with_the_warp(tmp_path):
    from contour.core import media as M
    from contour.core.project import OutputSettings
    from contour.warp.render import render_cut
    import contour.warp.render as R
    M.configure_ffmpeg()
    prj = _old_project(tmp_path, "rend")
    cut = prj.cuts[0]
    tracker = NS(store=None, matte=NS(w=cut.info.width // 2, h=cut.info.height // 2))
    rep = render_cut(prj, cut, tracker, OutputSettings(format="prores_422_hq", out_dir=str(tmp_path / "out"),
                                                        include_audio=False))
    assert rep.frames == cut.info.frames and not rep.cancelled
    assert not hasattr(rep, "inpaint_frames")
    assert "composite" not in vars(R)
    with pytest.raises(ImportError):
        import contour.warp.separate  # noqa: F401
    with pytest.raises(ImportError):
        import contour.warp.protect  # noqa: F401


@pytest.fixture
def win(qapp, monkeypatch):
    from contour.core.settings import settings
    from contour.ui import main_window as MW
    settings().ask_proxy = False
    settings().legacy_notified = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW.MainWindow()
    w.toasts = []
    real = w.toast
    monkeypatch.setattr(w, "toast", lambda text, kind="": (w.toasts.append(text), real(text, kind)))
    monkeypatch.setattr(w, "show_error", lambda title, text: None)
    yield w
    w.proxy_builder.cancel()
    w.thumbcache.stop()
    w.session.dirty = False
    w._discard_on_close = True
    w.close()


def test_no_protect_tab_tools_or_shortcuts(win):
    from PySide6.QtGui import QShortcut
    assert "protect" not in win.inspector.pages
    assert [k for k, _ in win.inspector.tabs.options] == ["skin", "body", "track", "output"]
    assert set(win.tool_btns) == {"edit"}
    assert "prot" not in win.ov
    keys = {s.key().toString() for s in win.findChildren(QShortcut)}
    assert "L" not in keys and "P" not in keys
    assert not hasattr(win, "set_protect") and not hasattr(win, "set_mode") and not hasattr(win, "protect_item")
    body = win.inspector.pages["body"]
    assert not hasattr(body, "mode") or body.mode is None
    assert not hasattr(win.viewer, "lineAdded") and not hasattr(win.viewer, "show_protect")


def test_old_project_opens_told_once_and_keeps_who_is_retouched(win, qapp, tmp_path):
    prj = _old_project(tmp_path, "tell")
    win.open_project(prj.path, wait=True)
    told = [t for t in win.toasts if "더 이상 적용되지 않고" in t]
    assert len(told) == 1 and "분리 + 배경 복원 모드" in told[0]
    cut = win.session.cut
    assert cut.person(2).role == "protect" and [p.id for p in cut.targets()] == [1]
    # the toggle: 보정 대상 on / off
    win.select_person(2)
    page = win.inspector.pages["skin"]
    page.refresh()
    assert page.role.isChecked() is False
    page.role.setChecked(True)
    assert cut.person(2).role == "target"
    page.role.setChecked(False)
    assert cut.person(2).role == "protect"
    win.session.dirty = False
    win.close_project()
    win.toasts.clear()
    win.open_project(prj.path, wait=True)
    assert not [t for t in win.toasts if "더 이상 적용되지 않고" in t], "told once per project"
    # the preview of the old separate cut is the warp's
    t = 0
    img = win.session.result(t) if hasattr(win.session, "result") else None
    assert img is None or img.shape[:2] == (win.session.proxy.h, win.session.proxy.w)
