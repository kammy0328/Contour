"""1.26.2 (the user 11/26, LNGSHOT BEEP BEEP: "여드름, 요철, 턱주름, 목주름 제거" saved in the project): 요철 and 턱주름 are sliders of their own,
kept with the project, and they switch the engine's bump and chin stages on for that person (the batch's settings FINAL_BUMP / CHIN_DIMPLE /
FINAL_CHIN did it for every face)."""
import inspect

from contour.core.params import PARAM, RETOUCH_KEYS
from contour.core.project import Curve, PartParams
from contour.retouch import engine as RE


def test_the_sliders_exist_and_are_kept():
    for k, lab in (("bumps", "요철"), ("chin_lines", "턱주름")):
        assert k in PARAM and PARAM[k].label == lab and k in RETOUCH_KEYS
        assert PARAM[k].default == 0                      # off unless set: projects of before look as they did
    p = PartParams()
    p.amounts["bumps"] = Curve(base=100.0)
    p.amounts["chin_lines"] = Curve(base=80.0)
    q = PartParams.from_dict(p.to_dict())
    assert q.amounts["bumps"].base == 100.0 and q.amounts["chin_lines"].base == 80.0


def test_the_engine_reads_them_per_person():
    src = inspect.getsource(RE._final_pass)
    assert 'job.v.get("bumps"' in src and 'job.v.get("chin_lines"' in src
    assert 'v.get("chin_lines"' in inspect.getsource(RE)
