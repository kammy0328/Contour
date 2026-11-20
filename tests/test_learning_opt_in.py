"""Learning data and automatic learning only in a project where they were switched on by hand (instruction 2026-11-20
§4-4): a new project is off, a project from before the rule (no 'learn' in its file, or the old default true) is off
and keeps what was collected, switching it off by hand removes what was collected as before, and every place that
collects pictures or starts learning goes through the one rule."""
import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path, name="p"):
    from conftest import make_video
    from contour.core import media as M
    from contour.core.project import Project
    prj = Project.create(str(tmp_path / name), name)
    prj.add_cut(M.probe(str(make_video(tmp_path / f"{name}.mp4"))), "c1")
    prj.save()
    return prj


def _raw(prj) -> dict:
    return json.loads((Path(prj.path) / "project.json").read_text(encoding="utf-8"))


def _write(prj, d: dict) -> None:
    (Path(prj.path) / "project.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def test_a_new_project_is_off_and_the_choice_is_saved(tmp_path):
    from contour.core.project import Project
    from contour.learn import allowed, chosen_off
    prj = _project(tmp_path)
    cut = prj.cuts[0]
    assert not prj.learn and not prj.learn_set and not allowed(prj) and not allowed(prj, cut) and not chosen_off(prj)
    assert _raw(prj)["learn"] is False and _raw(prj)["learn_set"] is False
    prj.set_learn(True)
    prj.save()
    again = Project.load(prj.path)
    assert again.learn and again.learn_set and allowed(again) and allowed(again, again.cuts[0]) and not again.learn_before
    again.cuts[0].learn = False                      # one cut taken out of it
    assert not allowed(again, again.cuts[0]) and chosen_off(again, again.cuts[0]) and allowed(again)
    again.set_learn(False)
    again.save()
    off = Project.load(prj.path)
    assert not off.learn and off.learn_set and chosen_off(off) and not off.learn_before


@pytest.mark.parametrize("stored", ["missing", True, False])
def test_a_project_from_before_the_rule_is_off_whatever_its_file_says(tmp_path, stored):
    """The file of 1.22.2 and earlier always wrote learn: true (the default, never a choice), and older ones have no
    such key: neither starts learning. What the file said is not changed by opening the project."""
    from contour.core.project import Project
    from contour.learn import allowed, chosen_off
    prj = _project(tmp_path)
    d = _raw(prj)
    d.pop("learn_set", None)
    if stored == "missing":
        d.pop("learn", None)
    else:
        d["learn"] = stored
    _write(prj, d)
    before = (Path(prj.path) / "project.json").read_bytes()
    old = Project.load(prj.path)
    assert not old.learn and not old.learn_set and not allowed(old) and not allowed(old, old.cuts[0])
    assert not chosen_off(old), "never chosen: what was collected before stays"
    assert old.learn_before == (stored is not False), "said once on opening when the old file meant 'on'"
    assert (Path(prj.path) / "project.json").read_bytes() == before
    # a file with learn: true and the marker forged false is still off (the marker is the choice)
    d["learn"], d["learn_set"] = True, False
    _write(prj, d)
    assert not allowed(Project.load(prj.path))


def _tracker(prj, cut, n=6):
    """A stand-in with what the collectors look at before they decide (they stop before reading frames)."""
    st = NS(pids=[1], has=lambda pid: False, has_face=True, raw={}, present=np.ones((1, n), bool))
    return NS(store=st, proxy=NS(n=n), project=prj, cut=cut)


def test_nothing_is_collected_from_a_project_that_did_not_choose(tmp_path, monkeypatch):
    from contour.core.project import Project
    from contour.core.settings import settings
    from contour.learn import fusion as FU, identity as ID, refiner as RF
    from contour.track import cloudcollect as CC
    monkeypatch.setenv("DIAP_CLOUD", str(tmp_path / "Cloud"))
    prj = _project(tmp_path)
    d = _raw(prj)
    d.pop("learn_set")
    d["learn"] = True                                    # as 1.22.2 saved it
    _write(prj, d)
    prj = Project.load(prj.path)
    cut = prj.cuts[0]
    tr = _tracker(prj, cut)
    assert settings().collect_learning_data and settings().self_learning, "the app-wide switches alone start nothing"
    # what was collected under the old rule is left where it is
    old_user = RF.samples_dir() / f"{prj.name}__{cut.id}.npz"
    old_auto = RF.auto_shard(prj, cut)
    for f in (old_user, old_auto):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"old")
    assert RF.collect_cut(prj, cut, tr) == 0 and RF.collect_auto(prj, cut, tr) == 0
    assert old_user.read_bytes() == b"old" and old_auto.read_bytes() == b"old"
    assert FU.collect(prj, cut, tr) == 0 and ID.collect(prj, cut) is False
    assert CC.collect(tr) == 0 and not list((tmp_path / "Cloud").rglob("*.jpg")) and not list((tmp_path / "Cloud").rglob("*.png"))
    # switched off by hand: what was collected from it goes, as before
    prj.set_learn(False)
    assert RF.collect_cut(prj, cut, tr) == 0 and RF.collect_auto(prj, cut, tr) == 0
    assert not old_user.exists() and not old_auto.exists()


def test_switched_on_the_collectors_run(tmp_path, monkeypatch):
    """The rule lets them through (what they then do is theirs: here they are only seen to start)."""
    from contour.learn import allowed, refiner as RF
    from contour.track import cloudcollect as CC
    prj = _project(tmp_path)
    prj.set_learn(True)
    cut = prj.cuts[0]
    tr = _tracker(prj, cut)
    seen = []
    monkeypatch.setattr(CC, "cloud_dir", lambda: seen.append("cloud") or None)
    assert allowed(prj, cut) and CC.collect(tr) == 0 and seen == ["cloud"], "past the rule (no folder: nothing written)"
    monkeypatch.setattr(RF, "pseudo_mask", lambda st, cut: (_ for _ in ()).throw(RuntimeError("past the rule")))
    with pytest.raises(RuntimeError, match="past the rule"):
        RF.collect_auto(prj, cut, tr)


def test_every_place_that_learns_goes_through_the_rule():
    """Read from the code: no collector or automatic learning is reached on the project's old 'learn' value alone."""
    learn = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "contour" / "learn").glob("*.py")}
    for name in ("refiner.py", "fusion.py", "identity.py"):
        assert "allowed(project, cut)" in learn[name], name
        assert not re.search(r"project\.learn\b", learn[name]), f"{name} decides on project.learn itself"
    assert "allowed(" in (ROOT / "contour" / "track" / "cloudcollect.py").read_text(encoding="utf-8")
    assert "allowed(project)" in (ROOT / "contour" / "track" / "xlink.py").read_text(encoding="utf-8")
    for rel in ("contour/ui/main_window.py", "contour/batch.py", "contour/ui/pages.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"(prj|project)\.learn and", src), f"{rel} decides on project.learn itself"
    mw = (ROOT / "contour" / "ui" / "main_window.py").read_text(encoding="utf-8")
    # starting the self-training, the status rules and the identity weights: each behind the rule
    for call in ("self.models_page.maybe_self_train", "SC.refit()", "ID.refit()"):
        for m in re.finditer(re.escape(call), mw):
            before = mw[max(0, m.start() - 420):m.start()]
            assert "learn_allowed" in before, f"{call} is reached without the rule"
    assert "allowed(" in (ROOT / "contour" / "ui" / "pages.py").read_text(encoding="utf-8").split("def maybe_self_train")[1][:900]


@pytest.fixture
def win(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from contour.ui import main_window as MW
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW.MainWindow()
    w.toasts = []
    monkeypatch.setattr(w, "toast", lambda text, kind="": w.toasts.append(text))
    monkeypatch.setattr(w, "show_error", lambda title, text: None)
    yield w
    w.proxy_builder.cancel()
    w.thumbcache.stop()
    w.session.dirty = False
    w._discard_on_close = True
    w.close()


def test_an_older_project_is_told_once_and_the_caches_it_needs_are_used(win, qapp, tmp_path, monkeypatch):
    import time
    from contour.core.settings import settings
    from contour.learn import statuscal as SC, identity as ID
    prj = _project(tmp_path)
    d = _raw(prj)
    d.pop("learn_set")
    d["learn"] = True
    _write(prj, d)

    def until(cond, timeout=8.0):
        t0 = time.time()
        while not cond() and time.time() - t0 < timeout:
            qapp.processEvents()
            time.sleep(0.005)
        return cond()
    win.open_project(prj.path, wait=True)
    assert until(lambda: win.session.cut is not None and win.session.proxy is not None), "it opens and shows as before"
    assert win.session.proxy.frame(0) is not None, "the frames the edit needs are read as before"
    told = [t for t in win.toasts if "직접 켠 프로젝트에서만" in t]
    assert len(told) == 1 and "프로젝트 메뉴" in told[0]
    assert not win._learn_allowed()
    # learning that would start from what the person does in this project: not reached
    ran = []
    monkeypatch.setattr(SC, "record", lambda *a, **k: ran.append("status"))
    monkeypatch.setattr(SC, "refit", lambda *a, **k: ran.append("status-refit"))
    monkeypatch.setattr(ID, "refit", lambda *a, **k: ran.append("identity-refit"))
    monkeypatch.setattr(win.models_page, "train", lambda *a, **k: ran.append("train"))
    assert win.models_page.maybe_self_train() is False
    win._relink(quiet=True)
    assert ran == []
    # opened again: not told twice; the file still says what it said
    win.toasts.clear()
    win._tell_learning(win.session.project)
    assert not win.toasts and str(win.session.project.created) in settings().learn_notified
    # switched on by hand: saved as a choice, and learning may start
    win.session.project.set_learn(True)
    assert win._learn_allowed()
    win._relink(quiet=True)
    assert "identity-refit" in ran
