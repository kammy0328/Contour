"""Train -> evaluate -> (maybe) activate. Never switches to a model that scores worse."""
from __future__ import annotations

from typing import Callable, Optional

from . import benchmark as BM
from . import refiner as RF


def train_and_compare(progress: Optional[Callable[[float, str], None]] = None, cancel=None,
                      auto_activate: bool = True) -> dict:
    def sub(lo, hi):
        return (lambda f, m="": progress(lo + (hi - lo) * f, m)) if progress else None

    meta = RF.train_new_version(sub(0.0, 0.75), cancel=cancel)
    ver = meta["version"]
    cur = RF.active_version()
    benches = BM.list_benchmarks()
    decision = ""
    new_score = cur_score = None
    if benches:
        r_new = BM.evaluate(ver, sub(0.75, 0.88))
        r_cur = BM.evaluate(cur, sub(0.88, 1.0))
        new_score, cur_score = BM.summary(r_new), BM.summary(r_cur)
        RF.update_meta(ver, bench=r_new, bench_score=new_score)
        better = new_score is not None and cur_score is not None and new_score < cur_score - 1e-4
    else:
        # better on unseen footage, or as good on clean frames and clearly better on hard ones
        better = meta["improvement"] > 0.03 or (
            meta["improvement"] > -0.005 and (meta.get("improve_degraded") or 0) > 0.05)
    # self-learning safety: the new model must also help on degraded frames (flash, colour light, blur) and
    # must not get worse on anything the user corrected by hand
    if meta.get("improve_degraded") is not None and meta["improve_degraded"] <= 0:
        better = False
    if meta.get("improve_human") is not None and meta["improve_human"] < 0:
        better = False
    if better and meta["improvement"] > -0.005 and auto_activate:
        RF.set_active(ver)
        decision = "activated"
    else:
        decision = "kept"
    RF.update_meta(ver, decision=decision, compared_to=cur, compared_score=cur_score)
    meta.update(decision=decision, bench_score=new_score, compared_score=cur_score, compared_to=cur)
    if progress:
        progress(1.0, "학습 완료")
    return meta
