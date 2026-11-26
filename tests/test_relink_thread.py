"""1.27: the people of a timeline are linked on a thread (ui/main_window._relink), not on the UI thread a second after
every analysis (the app's freeze log, 11/26: 7 to 21 s in identity.refit and xlink.link with a timeline of 80 cuts).
The window answers meanwhile; the result is put into the project when it is still the newest one asked for and the
members are as they were read; the people panel on a project never linked links at once."""
import threading
import time

import numpy as np
import pytest


def _project(tmp_path, cuts=3, people=2):
    from conftest import make_video
    from contour.core import media as M
    from contour.core.project import Person, Project
    prj = Project.create(str(tmp_path / "p"), "p")
    info = M.probe(str(make_video(tmp_path / "p.mp4")))
    rng = np.random.default_rng(0)
    proto = rng.normal(size=(people, 48)).astype(np.float32)
    for i in range(cuts):
        c = prj.add_cut(info, f"c{i}")
        c.analyzed = True
        c.tl = {"kind": "clip", "index": i}
        c.persons = [Person(id=j, name=f"인물 {j + 1}", color="#fff", role="target", first=0, last=5,
                            faces=[(proto[j] + 0.3 * rng.normal(size=48)).astype(np.float32).tolist() for _ in range(4)])
                     for j in range(people)]
    prj.save()
    return prj


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


@pytest.fixture
def opened(win, qapp, tmp_path, monkeypatch):
    from contour.learn import identity as ID
    monkeypatch.setattr(ID, "active", lambda kind: None)
    monkeypatch.setattr(ID, "threshold", lambda kind, d: d)
    prj = _project(tmp_path)

    def until(cond, timeout=10.0):
        t0 = time.time()
        while not cond() and time.time() - t0 < timeout:
            qapp.processEvents()
            time.sleep(0.005)
        return cond()
    win.open_project(prj.path, wait=True)
    assert until(lambda: win.session.project is not None)
    win.until = until
    win.idle = lambda: until(lambda: not [t for t in threading.enumerate() if t.name == "diap-relink" and t.is_alive()]) \
        and (qapp.processEvents() or True)
    return win


def _members(win):
    return {q.member for c in win.session.project.cuts for q in c.persons}


def test_the_window_answers_while_the_linking_is_worked_out(opened, monkeypatch):
    from contour.learn import identity as ID
    win = opened
    win.session.project.set_learn(True)
    gate, ran = threading.Event(), []

    def slow_refit():
        ran.append(threading.current_thread().name)
        gate.wait(10.0)
        return {}
    monkeypatch.setattr(ID, "refit", slow_refit)
    t0 = time.perf_counter()
    assert win._relink(quiet=True) is None
    assert time.perf_counter() - t0 < 0.5                       # back at once: the fit and the linking are not waited for
    assert win.until(lambda: bool(ran)) and ran == ["diap-relink"]
    assert _members(win) == {""}                                 # nothing put in yet
    ticks = []
    for _ in range(20):                                          # the UI thread goes on meanwhile
        win.until(lambda: False, timeout=0.01)
        ticks.append(1)
    assert len(ticks) == 20 and _members(win) == {""}
    gate.set()
    win.idle()
    assert win.until(lambda: "" not in _members(win))
    prj = win.session.project
    assert len(prj.members) == 2 and all(m["cuts"] == 3 for m in prj.members.values())
    assert not win.session.dirty                                 # saved with the project, as before


def test_a_newer_linking_takes_the_place_of_one_on_its_way(opened, monkeypatch):
    from contour.track import xlink as XL
    win = opened
    applied, gate = [], threading.Event()
    plan0 = XL.plan

    def plan(snap, keep_user=True):
        if not gate.is_set():
            gate.wait(10.0)                                      # the first one stands here while the second is asked for
        return plan0(snap, keep_user)
    monkeypatch.setattr(XL, "plan", plan)
    apply0 = XL.apply
    monkeypatch.setattr(XL, "apply", lambda prj, pl: applied.append(1) or apply0(prj, pl))
    win._relink(quiet=True)
    win._relink(quiet=True)
    gate.set()
    win.idle()
    assert win.until(lambda: "" not in _members(win))
    win.idle()
    assert applied == [1]                                        # put in once: the older one was dropped


def test_members_changed_by_hand_meanwhile_are_not_written_over(opened, monkeypatch):
    from contour.track import xlink as XL
    win = opened
    prj = win.session.project
    XL.link(prj)
    a = prj.cuts[0].persons[0]
    gate, n = threading.Event(), [0]
    plan0 = XL.plan

    def plan(snap, keep_user=True):
        n[0] += 1
        if n[0] == 1:
            gate.wait(10.0)
        return plan0(snap, keep_user)
    monkeypatch.setattr(XL, "plan", plan)
    win._relink(quiet=True)
    assert win.until(lambda: n[0] == 1)
    mid = XL.assign(prj, prj.cuts[0].id, a.id, None)             # the user, while it is worked out: a member of their own
    gate.set()
    win.idle()
    assert win.until(lambda: n[0] == 2)                          # worked out again from the members as they are now
    win.idle()
    assert a.member == mid and a.member_conf == 1.0 and [prj.cuts[0].id, a.id] in prj.members[mid]["fixed"]


def test_the_people_panel_links_at_once_without_the_learning(opened, monkeypatch):
    from contour.learn import identity as ID
    win = opened
    win.session.project.set_learn(True)
    ran = []
    monkeypatch.setattr(ID, "refit", lambda: ran.append(1) or {})
    r = win._relink(quiet=True, wait=True)
    assert r is not None and r["members"] == 2 and "" not in _members(win)
    assert ran == [] and not [t for t in threading.enumerate() if t.name == "diap-relink" and t.is_alive()]


def test_a_linking_of_a_project_closed_meanwhile_is_dropped(opened, monkeypatch, tmp_path):
    from contour.track import xlink as XL
    win = opened
    prj = win.session.project
    gate = threading.Event()
    plan0 = XL.plan
    monkeypatch.setattr(XL, "plan", lambda snap, keep_user=True: (gate.wait(10.0), plan0(snap, keep_user))[1])
    applied = []
    monkeypatch.setattr(XL, "apply", lambda p, pl: applied.append(1))
    win._relink(quiet=True)
    win.session.project = None                                   # (as closing the project leaves the session)
    gate.set()
    win.idle()
    assert applied == [] and all(q.member == "" for c in prj.cuts for q in c.persons)
    win.session.project = prj


def test_an_error_while_linking_is_said_and_nothing_is_changed(opened, monkeypatch):
    from contour.track import xlink as XL
    win = opened

    def bad(snap, keep_user=True):
        raise ValueError("no")
    monkeypatch.setattr(XL, "plan", bad)
    win._relink()
    win.idle()
    assert win.until(lambda: any("인물 연결 실패" in t for t in win.toasts))
    assert _members(win) == {""}
