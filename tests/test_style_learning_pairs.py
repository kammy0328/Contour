"""Style learning from a hand-retouched timeline (2026-11-18, LNGSHOT Bananas BEFORE/AFTER + XML): every shot of the
timeline is a sample of its own (they replaced each other: keyed by the plate's file), the pair counts once in the
preset, and acne/spot removal is not learned (DIAP's own): the learned preset leaves the 잡티 slider alone and the
comparison leaves the spots out."""
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.fixture
def data(tmp_path, monkeypatch):
    from contour.core import settings as S
    S._settings = S.AppSettings(data_dir=str(tmp_path / "data"))
    yield tmp_path
    S._settings = None


def _smp(off, skin, err=(10.0, 6.0), after="Y:/x/AFTER.mov", blem=None):
    vals = {"skin": skin, "shine": 20}
    if blem is not None:
        vals["blemish"] = blem
    cut = NS(source="Y:/x/BEFORE.mov", name=f"{off:03d}", tl={"kind": "clip"}, info=NS(offset=off))
    from contour.learn import style as SL
    return dict(clip=cut.name, source=cut.source, after=after, key=SL.sample_key(cut), not_learned=["blemish"],
                values=vals, detail=45.0, err_before=err[0], err_after=err[1])


def test_every_shot_of_a_timeline_is_kept(data):
    from contour.learn import style as SL
    for off, sk in ((0, 30), (81, 40), (133, 50)):
        SL.add_sample(_smp(off, sk))
    assert len(SL.load_samples()) == 3
    SL.add_sample(_smp(81, 45))                      # the same shot learned again replaces itself only
    ss = SL.load_samples()
    assert len(ss) == 3 and sorted(s["values"]["skin"] for s in ss) == [30, 45, 50]


def test_a_pair_counts_once_and_blemish_is_not_learned(data):
    from contour.learn import style as SL
    from contour.core import presets as PR
    from contour.core.project import PartParams, Curve
    many = [_smp(o, 60, after="Y:/a/AFTER.mov") for o in range(0, 300, 30)]       # 10 shots of one pair
    one = [dict(_smp(0, 20, after="Y:/b/B_after.mov"), key="b#0", err_before=10.0, err_after=4.0)]
    one2 = [dict(_smp(0, 20, after="Y:/c/C_after.mov"), key="c#0", err_before=10.0, err_after=4.0)]
    agg = SL.aggregate(many + one + one2)
    assert agg["values"]["skin"] == 20, agg            # two pairs at 20 outweigh one pair of ten shots at 60
    assert "blemish" not in agg["values"]
    for s in many + one + one2:
        SL.add_sample(s)
    pre = PR.load(SL.PRESET_NAME) if hasattr(PR, "load") else None
    if pre is None:
        import json
        pre = json.loads(PR._file(SL.PRESET_NAME).read_text(encoding="utf-8"))
    pp = PartParams()
    pp.amounts["blemish"] = Curve(base=70.0)
    PR.apply(pp, pre)
    assert pp.amounts["blemish"].base == 70.0, "the user's own 잡티 stays"
    assert pp.amounts["skin"].base == pre["values"]["skin"]


def test_comparison_leaves_spots_and_outliers_out():
    from contour.learn import style as SL
    rng = np.random.default_rng(0)
    d = rng.normal(0, 0.002, 20000).astype(np.float32)
    base = SL._trimmed_rms(d)
    d2 = d.copy()
    d2[:150] = 0.08                                  # a pimple the hand took out and the trial did not
    assert SL._trimmed_rms(d2) < 1.3 * base
    assert float(np.sqrt((d2 ** 2).mean())) > 3 * base

    # the detector's spots are left out of the face skin compared
    h = w = 240
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    o = np.full((h, w, 3), (0.72, 0.56, 0.48), np.float32)
    o = o + rng.normal(0, 0.006, (h, w, 1)).astype(np.float32)
    spot = np.exp(-((xx - 150) ** 2 + (yy - 120) ** 2) / (2 * 2.6 ** 2))[..., None]
    o = o * (1 - 0.18 * spot)
    m = np.zeros((h, w), bool)
    m[40:200, 40:200] = True
    lm = np.zeros((68, 2), np.float32)
    lm[:, 0] = np.linspace(20, 220, 68)
    lm[:, 1] = 120
    sp = SL._spots(o, lm, m)
    assert sp[120, 150] and sp.mean() < 0.05
