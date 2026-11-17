"""Batch work on one cut in its own process (python -m contour.batch analyze <project> <cut id> ...).

A timeline brings 50 to 200 cuts; analysing them one after another in the app's own process slowly fills
memory (models, caches, numpy buffers that are never handed back). One process per cut starts clean every
time, and a crash takes one cut down, not the session.

The process never writes project.json (the app owns it). It writes the cut's own files (proxy, tracking
store) and leaves the updated cut description in cuts/<id>/batch_result.json for the app to take over.
Progress goes to stdout as lines: PROGRESS <0..1> <message>.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _say(f: float, msg: str = "") -> None:
    try:
        sys.stdout.write(f"PROGRESS {float(f):.4f} {msg}\n")
        sys.stdout.flush()
    except Exception:
        pass


def result_path(project, cut) -> Path:
    return project.cut_dir(cut) / "batch_result.json"


def prescan_people(cut, n: int = 3) -> bool:
    """A quick look at a few frames: is anyone large enough to retouch?"""
    import numpy as np
    from .core import media as M
    from .track.infer import Detector
    info = cut.info
    det = Detector("det_yolox_m", score_thr=0.45, nms_thr=0.6)
    W = 1280
    H = max(2, int(round(info.height * W / max(1, info.width))) // 2 * 2)
    ts = sorted({int(x) for x in np.linspace(0, max(0, info.frames - 1), n)})
    for t in ts:
        with M.FrameReader(info, t, 1, "rgb24", size=(W, H), rgb_convert=True, prefetch=1) as rd:
            pl = rd.read()
        if pl is None:
            continue
        try:
            boxes, scores = det(np.ascontiguousarray(pl[0]))
        except Exception:
            return True
        for b in boxes:
            x0, y0, x1, y1 = [float(v) for v in b[:4]]
            if (y1 - y0) >= 0.08 * H:
                return True
    return False


def analyze(project_path: str, cut_id: str, accuracy: str = "high", all_targets: bool = True,
            prescan: bool = True) -> int:
    from .core.colour import manager
    from .core.project import Project
    from .core.proxy import Proxy
    from .core.settings import settings
    from .track.analyze import CutTracker
    from .track import status as ST
    from .warp.render import _vs_for
    from .core import media as M

    s = settings()
    M.configure_ffmpeg()
    prj = Project.load(project_path)
    cut = prj.cut(cut_id)
    if cut is None:
        print("ERROR 컷을 찾을 수 없습니다", flush=True)
        return 2
    out = result_path(prj, cut)
    if out.exists():
        out.unlink()
    t0 = time.time()
    from .core.frames import frames_for, proxy_for, proxy_mode
    vs = _vs_for(cut)
    px = frames_for(prj, cut, vs)
    if px is None or not px.is_valid(vs.key()):
        # neither the original nor a proxy made in this colour view (an old proxy in another view is for looking
        # at only): not analysed, and not taken for a cut without people either (before the quick check)
        if px is not None:
            px.close()
        print("ERROR 원본 파일을 찾을 수 없습니다: " + str(cut.source or cut.info.path), flush=True)
        return 3
    if prescan and cut.tl.get("kind", "clip") == "clip" and not cut.analyzed:
        _say(0.01, "인물 확인 중")
        try:
            people = prescan_people(cut)
        except Exception:
            traceback.print_exc()
            people = True
        if not people:
            cut.status = dict(cut.status or {}, auto="none", reason="인물 없음 (빠른 검사)", prescan=True)
            out.write_text(json.dumps(dict(cut=cut.to_dict(), seconds=time.time() - t0, analyzed=False),
                                      ensure_ascii=False), encoding="utf-8")
            _say(1.0, "인물 없음")
            return 0
    if px.kind == "source" and proxy_mode() == "always":
        # 설정 > 프록시 만들기 > 항상: made first, the analysis then reads it. It is optional: when it cannot be made
        # (no room, a write error) the analysis reads the original as it would without the setting
        _say(0.02, "프록시 만드는 중")
        try:
            pf = proxy_for(prj, cut)
            if pf.build(manager(), vs, lambda f, m="": _say(0.02 + 0.18 * f, "프록시 만드는 중"), lambda: False):
                px.close()
                px = frames_for(prj, cut, vs)
        except Exception as e:
            print(f"WARN 프록시를 만들지 못해 원본에서 읽어 분석합니다: {str(e)[:200]}", flush=True)
    tr = CutTracker(prj, cut, px)
    try:
        tr.analyze(lambda f, m="": _say(0.2 + 0.75 * f, m), lambda: False, accuracy=accuracy)
    except RuntimeError as e:
        if "인물을 찾지 못했습니다" not in str(e):
            raise
        cut.status = dict(cut.status or {}, auto="none", reason="인물 없음")
        out.write_text(json.dumps(dict(cut=cut.to_dict(), seconds=time.time() - t0, analyzed=False),
                                  ensure_ascii=False), encoding="utf-8")
        _say(1.0, "인물 없음")
        return 0
    cut.stats.analysis_runs += 1
    cut.stats.auto_ok_rate = tr.auto_ok_rate(cut.tracking.conf_threshold)
    try:
        from .track.models import take_cpu_fallbacks
        fb = take_cpu_fallbacks()
        if fb:
            why = "GPU 메모리가 부족해" if any(w == "GPU 메모리 부족" for _, w in fb) else "GPU에서 열지 못해"
            print(f"WARN {why} 일부 모델을 CPU로 실행했습니다 ({', '.join(sorted({n for n, _ in fb}))[:120]}). "
                  "분석이 많이 느려질 수 있습니다", flush=True)
    except Exception:
        pass
    try:
        from .learn.refiner import collect_auto
        from .learn import fusion as FU
        _say(0.96, "자가학습 데이터 모으는 중")
        collect_auto(prj, cut, tr)
        if FU.collect(prj, cut, tr):
            FU.refit()
    except Exception:
        traceback.print_exc()
    ST.evaluate(cut, tr, make_targets=all_targets)
    tr.save()
    try:
        from .learn import identity as ID
        ID.collect(prj, cut)
    except Exception:
        traceback.print_exc()
    # a hand retouch of this shot (or of the whole timeline) came with it: learn the style right here
    if cut.retouch_ref and Path(cut.retouch_ref).exists() and prj.learn and cut.learn \
            and (cut.status or {}).get("auto") in ("good", "bad"):
        try:
            from .learn import style as SL
            _say(0.97, "수작업 결과와 비교 중")
            smp = SL.fit_pair(cut, tr, cut.retouch_ref, lambda f, m="": _say(0.97 + 0.02 * f, m))
            cut.style_learned = smp
            SL.add_sample(smp)
        except Exception as e:
            print(f"style: {e}", flush=True)
    out.write_text(json.dumps(dict(cut=cut.to_dict(), seconds=time.time() - t0, analyzed=True), ensure_ascii=False),
                   encoding="utf-8")
    _say(1.0, "완료")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="contour.batch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("project")
    a.add_argument("cut")
    a.add_argument("--accuracy", default="high")
    a.add_argument("--all-targets", type=int, default=1)
    a.add_argument("--prescan", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "analyze":
            return analyze(args.project, args.cut, args.accuracy, bool(args.all_targets), bool(args.prescan))
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR {e.__class__.__name__}: {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
