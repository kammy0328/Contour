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


def prescan_people(cut, frames) -> dict:
    """The quick look for people (track/prescan.py): {'verdict': 'present' | 'none' | 'unsure', 'reason': ...}.
    frames: the cut's frames as the analysis sees them (an object with frame(t), in its colour view)."""
    from .track import prescan as PS
    a, b = cut.work_range
    return PS.Prescan().look(lambda t: frames.frame(a + t), max(0, b - a))


def _skip_status(cut, r: dict) -> dict:
    figs = {k: r[k] for k in ("looked", "read", "best_score", "tallest", "seconds") if k in r}
    return dict(cut.status or {}, auto="none", reason=r.get("reason", "인물 없음 (빠른 검사)"), prescan=True,
                prescan_info=figs)


def prescan(project_path: str, cut_ids: list) -> int:
    """One process, one detector, every listed cut (in their order on each source). Prints
    PRESCAN <cut id> <verdict> <json> per cut and writes cuts/<id>/prescan.json; never project.json."""
    from .core.project import Project
    from .core.frames import frames_for
    from .core import media as M
    from .track import prescan as PS
    from .warp.render import _vs_for
    M.configure_ffmpeg()
    prj = Project.load(project_path)
    cuts = [c for c in (prj.cut(i) for i in cut_ids) if c is not None]
    t0 = time.time()
    res = PS.scan(prj, cuts, progress=_say, frames_for=lambda p, c: frames_for(p, c, _vs_for(c)))
    for c in cuts:
        r = res.get(c.id) or dict(verdict=PS.UNSURE, reason="확인하지 못함")
        try:
            (prj.cut_dir(c) / "prescan.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        print(f"PRESCAN {c.id} {r['verdict']} {json.dumps(r, ensure_ascii=False)}", flush=True)
    print(f"INFO 인물 빠른 확인 {len(cuts)}컷 {time.time() - t0:.1f}초", flush=True)
    return 0


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
        # before the large models are opened: only a clear 'nobody' skips the cut (not sure: analysed)
        _say(0.01, "인물 확인 중")
        try:
            r = prescan_people(cut, px)
        except Exception:
            traceback.print_exc()
            r = dict(verdict="unsure")
        if r.get("verdict") == "none":
            cut.status = _skip_status(cut, r)
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
    cut.stats.analysis_seconds = round(time.time() - t0, 1)
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
    from .learn import allowed as _learn_allowed
    if cut.retouch_ref and Path(cut.retouch_ref).exists() and _learn_allowed(prj, cut) \
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
    q = sub.add_parser("prescan")
    q.add_argument("project")
    q.add_argument("cuts", nargs="+")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "analyze":
            return analyze(args.project, args.cut, args.accuracy, bool(args.all_targets), bool(args.prescan))
        if args.cmd == "prescan":
            return prescan(args.project, list(args.cuts))
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR {e.__class__.__name__}: {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
