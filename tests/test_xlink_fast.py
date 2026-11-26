"""1.27: people linked across the cuts (track/xlink.py) without the cube of the people on the UI thread.

The joining keeps the groups' sums and gives the groups the loop of 1.26.3 gave (kept here to compare with: every
round, every pair's mean from the people's matrix); the linking is worked out from a snapshot of the project (plan: any
thread, the project not touched) and put into it afterwards (apply)."""
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from contour.track import xlink as XL


def _reference(groups, E, Wt, user):
    """The joining of 1.26.3, as it was."""
    groups = [list(g) for g in groups]

    def gscore(a, b):
        sub = E[np.ix_(a, b)]
        w = Wt[np.ix_(a, b)]
        if (sub <= -49).any():
            return -99.0
        if w.sum() <= 0:
            return -99.0
        return float((sub * w).sum() / w.sum())
    alive = [True] * len(groups)
    mid_of_group = set(user)
    while True:
        best = (0.0, None, None)
        for a in range(len(groups)):
            if not alive[a]:
                continue
            for b in range(a + 1, len(groups)):
                if not alive[b]:
                    continue
                if a in mid_of_group and b in mid_of_group:
                    continue
                s = gscore(groups[a], groups[b])
                if s > best[0]:
                    best = (s, a, b)
        if best[1] is None:
            break
        a, b = best[1], best[2]
        if b in mid_of_group and a not in mid_of_group:
            a, b = b, a
        groups[a] += groups[b]
        alive[b] = False
    return [g for k, g in enumerate(groups) if alive[k]]


def _case(rng, n, members, users, never, noweight):
    """People of `members` real members: evidence above 0 within a member, below between (with noise that crosses 0),
    some pairs "never one member", some without any evidence, the first `users` groups the user's own."""
    who = rng.integers(0, members, n)
    E = np.zeros((n, n), np.float32)
    Wt = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            e = (1.2 if who[i] == who[j] else -1.0) + rng.normal(0, 0.9)
            w = float(rng.choice([0.35, 0.625, 0.75, 1.0]))
            r = rng.random()
            if r < never:
                e, w = -50.0, 1.0
            elif r < never + noweight:
                e, w = 0.0, 0.0
            E[i, j] = E[j, i] = e
            Wt[i, j] = Wt[j, i] = w
    order = list(rng.permutation(n))
    groups = []
    for _ in range(users):
        k = int(rng.integers(1, 4))
        g, order = order[:k], order[k:]
        if g:
            groups.append([int(x) for x in g])
    user = set(range(len(groups)))
    groups += [[int(x)] for x in sorted(order)]
    return groups, E, Wt, user


@pytest.mark.parametrize("seed", range(40))
def test_the_joining_gives_the_groups_of_the_loop_before(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 70))
    groups, E, Wt, user = _case(rng, n, int(rng.integers(1, 9)), int(rng.integers(0, 4)), float(rng.choice([0.0, 0.03, 0.15])),
                                float(rng.choice([0.0, 0.05, 0.3])))
    want = _reference(groups, E, Wt, user)
    got = XL._merge([list(g) for g in groups], E, Wt, user)
    assert got == want                                   # the same groups, their people in the same order


def test_equal_pairs_are_joined_in_the_same_order():
    """A shot used twice in the edit: people with the very same features, so pairs with the very same evidence."""
    rng = np.random.default_rng(3)
    for trial in range(12):
        base, E0, W0, _ = _case(rng, 14, 3, 0, 0.05, 0.1)
        idx = np.r_[np.arange(14), rng.integers(0, 14, 10)]          # ten of the people a second time
        E, Wt = E0[np.ix_(idx, idx)].copy(), W0[np.ix_(idx, idx)].copy()
        np.fill_diagonal(E, 0)
        np.fill_diagonal(Wt, 0)
        groups = [[i] for i in range(len(idx))]
        assert XL._merge([list(g) for g in groups], E, Wt, set()) == _reference(groups, E, Wt, set())


def test_joining_300_people_is_no_longer_a_wait():
    rng = np.random.default_rng(1)
    groups, E, Wt, user = _case(rng, 300, 7, 2, 0.02, 0.05)
    t0 = time.perf_counter()
    got = XL._merge(groups, E, Wt, user)
    assert time.perf_counter() - t0 < 5.0                # the loop before: about a minute
    assert sorted(i for g in got for i in g) == list(range(300))


# ------------------------------------------------------------------------------------------------ a project
def _project(tmp_path, rng, cuts=6, members=4, per_cut=3):
    proto_f = rng.normal(size=(members, 64)).astype(np.float32)
    proto_r = rng.normal(size=(members, 32)).astype(np.float32)
    who = {}

    def person(cid, pid, m):
        who[(cid, pid)] = m
        return NS(id=pid, role="target", name=f"인물 {pid + 1}", first=0, last=30, member="", member_conf=0.0,
                  faces=[(proto_f[m] + 0.35 * rng.normal(size=64)).astype(np.float32).tolist() for _ in range(5)],
                  reid=(proto_r[m] + 0.35 * rng.normal(size=32)).astype(np.float32).tolist())
    cs = []
    for i in range(cuts):
        cid = f"c{i:02d}"
        ps = [person(cid, j, int(m)) for j, m in enumerate(rng.choice(members, size=per_cut, replace=False))]
        c = NS(id=cid, analyzed=True, status={}, tl={"kind": "clip", "index": i}, persons=ps)
        c.person = lambda pid, c=c: next((q for q in c.persons if q.id == pid), None)
        cs.append(c)
    prj = NS(members={}, member_cannot=[], cuts=cs, folder=Path(tmp_path))
    prj.timeline_cuts = lambda: sorted([c for c in prj.cuts if c.tl], key=lambda c: c.tl["index"])
    prj.cut = lambda cid: next((c for c in prj.cuts if c.id == cid), None)
    return prj, who


@pytest.fixture
def raw_features(monkeypatch, tmp_path):
    from contour.learn import identity as ID
    monkeypatch.setattr(ID, "active", lambda kind: None)
    monkeypatch.setattr(ID, "threshold", lambda kind, d: d)


def test_people_of_one_member_are_linked_and_people_of_one_cut_never(tmp_path, raw_features):
    prj, who = _project(tmp_path, np.random.default_rng(0))
    r = XL.link(prj)
    assert r["people"] == 18 and r["members"] == 4 and r["multi"] == 4
    by = {}
    for c in prj.cuts:
        assert len({q.member for q in c.persons}) == len(c.persons)      # one cut: all different members
        for q in c.persons:
            by.setdefault(q.member, set()).add(who[(c.id, q.id)])
            assert 0.0 <= q.member_conf <= 1.0 and q.name == prj.members[q.member]["name"]
    assert all(len(v) == 1 for v in by.values())                        # a member is one real person
    # again: the same members keep their ids and names
    names = {m: v["name"] for m, v in prj.members.items()}
    XL.link(prj)
    assert {m: v["name"] for m, v in prj.members.items()} == names


def test_the_plan_does_not_touch_the_project_and_apply_puts_it_in(tmp_path, raw_features):
    prj, who = _project(tmp_path, np.random.default_rng(1))
    snap = XL.snapshot(prj)
    pl = XL.plan(snap)
    assert prj.members == {} and all(q.member == "" for c in prj.cuts for q in c.persons)
    # the snapshot is a copy: what the user does meanwhile is not in it, and is not undone by reading it
    prj.cuts[0].persons[0].role = "ignore"
    assert len(snap["cuts"][0]["people"]) == 3
    r = XL.apply(prj, pl)
    assert r == pl["result"] and prj.members is pl["members"]
    assert all(q.member in prj.members for c in prj.cuts for q in c.persons)
    # a cut that is gone meanwhile, a person that is gone: the rest is put in
    prj2, _ = _project(tmp_path, np.random.default_rng(1))
    pl2 = XL.plan(XL.snapshot(prj2))
    prj2.cuts.pop(2)
    prj2.cuts[0].persons.pop(0)
    XL.apply(prj2, pl2)
    assert all(q.member in prj2.members for c in prj2.cuts for q in c.persons)


def test_the_users_links_and_never_links_are_kept(tmp_path, raw_features):
    prj, who = _project(tmp_path, np.random.default_rng(2))
    XL.link(prj)
    a = prj.cuts[0].persons[0]
    # the user: this person is a member of their own, never with the people it was with
    others = [x for x in prj.members[a.member]["people"] if x != ["c00", a.id]]
    assert others
    mid = XL.assign(prj, "c00", a.id, None)
    XL.link(prj)
    assert a.member == mid and a.member_conf == 1.0
    for cid, pid in others:
        assert prj.cut(cid).person(pid).member != mid
    assert ["c00", a.id] in prj.members[mid]["fixed"]


def test_the_cut_identity_file_is_read_in_one_piece(tmp_path, raw_features, monkeypatch):
    prj, who = _project(tmp_path, np.random.default_rng(4), cuts=3)
    d = tmp_path / "cuts" / "c00"
    d.mkdir(parents=True)
    rng = np.random.default_rng(9)
    q = prj.cuts[0].persons[0]
    F = rng.normal(size=(7, 64)).astype(np.float32)
    np.savez_compressed(d / "identity.npz", face=F, face_pid=np.full(7, q.id, np.int32), reid=np.zeros((0, 0), np.float32),
                        reid_pid=np.zeros(0, np.int32), cooccur=np.zeros((0, 2), np.int32))
    insts = XL._instances(XL.snapshot(prj))
    it = next(i for i in insts if i.cid == "c00" and i.pid == q.id)
    assert it.faces.shape == (7, 64)                                    # the file's samples, not the few kept with the person
    assert not (tmp_path / "cuts" / "c01").exists()                     # reading makes no folder
