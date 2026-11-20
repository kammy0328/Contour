"""Model packages, the operator's side (instruction 2026-11-20 §4-2, §4-3): what is said when a cut was analysed with
package versions that are not on this computer (the needed version and the explicit choice, never another version on
its own), 모델 관리 (what each kind of work needs and has, bringing a package in from a folder or a zip without a
connection, going back to a version), and that editing and opening what is saved go on without the models."""
import time

import pytest

from contour.track import modelpkg as MP
from test_modelpkg import A1, KEYS, RVM, RVM2, make_registry


def _until(qapp, cond, timeout=8.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    return cond()


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setattr(MP, "TRUSTED", dict(KEYS))
    MP._store = None
    yield MP.store()
    MP._store = None


@pytest.fixture
def win(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from contour.ui import main_window as MW
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW.MainWindow()
    w.errors = []
    monkeypatch.setattr(w, "show_error", lambda title, text: w.errors.append((title, text)))
    w.toasts = []
    monkeypatch.setattr(w, "toast", lambda text, kind="": w.toasts.append((kind, text)))
    yield w
    w.proxy_builder.cancel()
    w.thumbcache.stop()
    w.session.dirty = False
    w._discard_on_close = True
    w.close()


def _analysed_project(tmp_path, versions):
    """A project whose one cut says it was analysed with these package versions (no tracks: the notes and the
    choice do not need them)."""
    from conftest import make_video
    from contour.core import media as M
    from contour.core.project import Project
    prj = Project.create(str(tmp_path / "prj"), "p")
    cut = prj.add_cut(M.probe(str(make_video(tmp_path / "a.mp4"))), "c1")
    cut.analyzed = True
    cut.model_versions = dict(versions)
    prj.save()
    return prj, cut


# ------------------------------------------------------------------------------------------------- 모델 관리
def test_the_dialog_says_what_each_kind_of_work_has_and_brings_a_package_in(qapp, tmp_path, trusted):
    from contour.ui.modelsdlg import ModelsDialog
    d = ModelsDialog()
    try:
        t = d.tasks.text()
        assert "분석" in t and "모델 없음" in t and "YOLOX-m" in t and "새 분석" in t, "what is missing and what it stops"
        assert "얼굴·목 영역</b>: 사용 가능" in t, "the face model is part of the app"
        rows = {d.tree.topLevelItem(i).data(0, 0x0100)["key"]: d.tree.topLevelItem(i) for i in range(d.tree.topLevelItemCount())}
        assert rows["matte_rvm"].text(1) == "없음" and d.b_dir.isEnabled() and d.b_zip.isEnabled()
        assert not d.b_use.isEnabled() and not d.b_del.isEnabled() and not d.b_check.isEnabled()
        root = tmp_path / "reg"
        make_registry(root, [RVM, RVM2, A1])
        (root / "pose_x" / "2026.11.1" / "model.onnx").write_bytes(b"?" * 5000)            # changed after signing
        d.bring_in(str(root))
        assert not d.b_dir.isEnabled(), "one thing at a time"
        assert _until(qapp, lambda: d.job is None)
        assert "RVM 인물 매트 2026.12.1" in d.msg.text() and "SHA-256" in d.msg.text() and d.changed
        assert trusted.active("matte_rvm") == "2026.12.1" and trusted.active("pose_x") is None
        rows = {d.tree.topLevelItem(i).data(0, 0x0100)["key"]: d.tree.topLevelItem(i) for i in range(d.tree.topLevelItemCount())}
        assert (rows["matte_rvm"].text(1), rows["matte_rvm"].text(2)) == ("패키지", "2026.12.1")
        # an older version beside it, then back to it and forward again, then the unused one removed
        MP.bring_in(root, keys=["matte_rvm"], versions={"matte_rvm": "2026.11.1"}, activate=False)
        d.refresh()
        d.tree.setCurrentItem(next(d.tree.topLevelItem(i) for i in range(d.tree.topLevelItemCount())
                                   if d.tree.topLevelItem(i).data(0, 0x0100)["key"] == "matte_rvm"))
        assert d.tree.currentItem().text(3) == "2026.11.1" and d.b_use.isEnabled() and d.b_del.isEnabled()
        d.use_version("matte_rvm", "2026.11.1")
        assert trusted.active("matte_rvm") == "2026.11.1" and "분석한 버전" in d.msg.text()
        with MP.pinned({"matte_rvm": "2026.12.1"}):             # work going on that began with it
            d.remove_versions("matte_rvm", ["2026.12.1"])
            assert trusted.versions("matte_rvm") == ["2026.11.1", "2026.12.1"] and "남겼습니다" in d.msg.text()
        d.remove_versions("matte_rvm", ["2026.12.1"])
        assert trusted.versions("matte_rvm") == ["2026.11.1"]
        # the installed files are read again; one that changed takes its version out of use
        d._check()
        assert _until(qapp, lambda: d.job is None) and "목록과 같습니다" in d.msg.text()
        trusted.path("matte_rvm").write_bytes(b"X" * 4000)
        d._check()
        assert _until(qapp, lambda: d.job is None)
        assert "손상된 버전" in d.msg.text() and trusted.active("matte_rvm") is None
    finally:
        d.close()


def test_without_a_publishing_key_nothing_can_be_installed_and_the_dialog_says_so(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QLabel
    from contour.ui.modelsdlg import ModelsDialog
    assert MP.TRUSTED == {}, "no key is trusted until the publisher's own is put in a release"
    d = ModelsDialog()
    try:
        assert not d.b_dir.isEnabled() and not d.b_zip.isEnabled() and not d.b_net.isEnabled()
        assert any("배포 키가 아직 없어" in w.text() for w in d.findChildren(QLabel))
    finally:
        d.close()
    make_registry(tmp_path / "reg", [RVM])
    with pytest.raises(MP.PackageError) as e:
        MP.bring_in(tmp_path / "reg")
    assert e.value.why == "signature"


def test_a_zip_is_brought_in_like_a_folder(qapp, tmp_path, trusted):
    import zipfile
    from contour.ui.modelsdlg import ModelsDialog
    root = tmp_path / "reg"
    make_registry(root, [RVM])
    z = tmp_path / "models.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(f, f.relative_to(root).as_posix())
    d = ModelsDialog()
    try:
        d.bring_in(str(z))
        assert _until(qapp, lambda: d.job is None)
        assert trusted.active("matte_rvm") == "2026.11.1" and "설치" in d.msg.text()
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".modelpkg_")], "nothing left beside the zip"
    finally:
        d.close()


# ------------------------------------------------------------------------------------------------- a cut's versions
def test_a_cut_analysed_with_versions_that_are_not_here_says_what_is_needed(win, qapp, tmp_path, trusted, monkeypatch):
    from contour.ui import inspector as INS
    from contour.ui import modelsdlg as MD
    reg = make_registry(tmp_path / "reg", [RVM, RVM2])
    trusted.install(reg, "matte_rvm", "2026.12.1", trusted=KEYS)
    prj, cut = _analysed_project(tmp_path, {"matte_rvm": "2026.11.1"})
    win.open_project(prj.path, wait=True)
    assert _until(qapp, lambda: win.session.cut is not None and win.session.proxy is not None)
    cut = win.session.cut
    warn = [t for k, t in win.toasts if k == "warn" and "2026.11.1" in t]
    assert len(warn) == 1, win.toasts
    assert "RVM 인물 매트 2026.11.1" in warn[0] and "설치된 버전 2026.12.1" in warn[0]
    assert "다시 분석" in warn[0] and "미리보기·렌더" in warn[0], "the choice, and what goes on working"
    txt, blocked = INS.models_line(cut)
    assert blocked and "RVM 인물 매트 2026.11.1" in txt
    win.inspector.show_tab("track")
    tab = win.inspector.tabs_w["track"] if hasattr(win.inspector, "tabs_w") else None
    # said once for the cut in this state, not on every refresh
    n = len(win.toasts)
    win._after_open()
    assert len([t for k, t in win.toasts[n:] if "2026.11.1" in t]) == 0
    # work that would open the models: nothing is made with the installed version; the choice is asked
    asked = []
    monkeypatch.setattr(MD, "ask_versions", lambda parent, text: asked.append(text) or "")

    def work(progress, cancelled):
        with MP.for_cut(cut):
            raise AssertionError("ran with another version")
    win.run_job("인물 매트", work, lambda res: None)
    assert _until(qapp, lambda: win.job is None)
    assert len(asked) == 1 and "2026.11.1" in asked[0] and not win.errors, "the choice, not an error box"
    # 'models' opens 모델 관리, 'reanalyse' starts the whole analysis; nothing happens on its own
    went = []
    monkeypatch.setattr(win, "open_models", lambda: went.append("models"))
    monkeypatch.setattr(win, "run_analysis", lambda: went.append("reanalyse"))
    for pick in ("models", "reanalyse", ""):
        monkeypatch.setattr(MD, "ask_versions", lambda parent, text, pick=pick: pick)
        win._versions_needed("x")
    assert went == ["models", "reanalyse"]
    # the version installed: the note is gone and the cut's work runs with the version it was analysed with
    trusted.install(reg, "matte_rvm", "2026.11.1", trusted=KEYS, activate=False)
    assert MD.needed_text(cut) == "" and INS.models_line(cut) == ("분석한 모델 패키지 1개 · 버전 2026.11.1", False)
    with MP.for_cut(cut):
        assert MP.version_in_use("matte_rvm") == "2026.11.1" and trusted.active("matte_rvm") == "2026.12.1"


def test_the_matte_refresh_rests_while_the_cuts_versions_are_not_here(win, qapp, tmp_path, trusted, monkeypatch):
    from types import SimpleNamespace as NS
    from contour.track import matte_refresh as MR
    reg = make_registry(tmp_path / "reg", [RVM, RVM2])
    trusted.install(reg, "matte_rvm", "2026.12.1", trusted=KEYS)
    prj, cut = _analysed_project(tmp_path, {"matte_rvm": "2026.11.1"})
    win.open_project(prj.path, wait=True)
    assert _until(qapp, lambda: win.session.cut is not None and win.session.proxy is not None)
    cut = win.session.cut
    ran = []
    monkeypatch.setattr(MR, "compute", lambda *a, **k: ran.append(1))
    win.session.tracker = NS(store=NS(), has_seg=lambda: True, cut=cut)
    cut.status = dict(cut.status or {}, matte_dirty=[[1, 3]])
    win.refresher._go()
    qapp.processEvents()
    assert not ran and win.refresher.running is None, "nothing is made with another version"
    assert win.refresher.pending(cut) == [[1, 3]], "the marks wait for the version"
    assert "2026.11.1" in win.refresher.last_error
    win.session.tracker = None


def test_editing_and_opening_go_on_without_any_model(win, qapp, project, monkeypatch):
    """No model on this computer and no connection: the project opens, cuts switch, values are edited and saved;
    what needs a model says which one."""
    import socket
    import urllib.request
    from contour.core.project import Person, Project
    from contour.track import models as M
    calls = []

    def no_network(*a, **k):
        calls.append(a)
        raise OSError("network is unreachable")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert not M.availability()[1]["ok"]
    win.open_project(project.path, wait=True)
    assert _until(qapp, lambda: win.session.cut is not None and win.session.proxy is not None)
    a, b = win.session.project.cuts[0], win.session.project.cuts[1]
    a.persons = [Person(id=1, name="A", color="#ff8800", role="target")]
    a.params_for(1).amounts["body_skin"].base = 40.0
    win.session.dirty = True
    win.open_cut(b.id)
    assert _until(qapp, lambda: win.session.cut is not None and win.session.cut.id == b.id and win.session.proxy is not None)
    win.open_cut(a.id)
    assert _until(qapp, lambda: win.session.cut is not None and win.session.cut.id == a.id)
    assert win.save()
    assert Project.load(project.path).cuts[0].params_for(1).amounts["body_skin"].base == 40.0
    assert not calls, "nothing asked for a connection"
    with pytest.raises(M.ModelMissing) as e:
        M.ensure_model("det_yolox_m")
    assert "YOLOX-m" in str(e.value) and "모델 관리" in str(e.value)
