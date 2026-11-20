"""The same short run on every system (instruction 2026-11-20 §5 실행 검증): a project opened, its original read, a
proxy made, the models that come with the app run once (and held against the CPU when an accelerator ran them),
one cut exported, and the child job started the way the app starts it.

    python tools/ci_smoke.py [--out DIR] [--json FILE] [--keep]

No model is downloaded and nothing is sent: the project is the frozen one of the tests (saved by DIAP 1.22.2 on
Windows, so its original is named by a Windows path and is connected again here, as on any other computer) or,
where the tests are not installed, one made on the spot. Prints what each step did and took, and one JSON line;
the exit code is 0 only when every step passed. Run by the CI of each system and by hand on a new computer."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CONTOUR_NO_UPDATE", "1")

FIXTURE = ROOT / "tests" / "data" / "baseline_1222"
TOLERANCE = 2            # of 1023: what a render on another system may differ from the frozen one (the tests' own)
RES: dict = dict(steps=[], ok=True)


def step(name):
    def deco(fn):
        t0 = time.time()
        d = dict(name=name, ok=False)
        try:
            d.update(fn() or {})
            d["ok"] = True
        except Exception as e:
            d["error"] = f"{type(e).__name__}: {e}"
            d["trace"] = traceback.format_exc()[-1500:]
            RES["ok"] = False
        d["seconds"] = round(time.time() - t0, 2)
        RES["steps"].append(d)
        print(("ok  " if d["ok"] else "FAIL") + f" {name} ({d['seconds']} s) "
              + json.dumps({k: v for k, v in d.items() if k not in ("name", "ok", "seconds", "trace")}, ensure_ascii=False)[:600],
              flush=True)
        return fn
    return deco


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep = "--keep" in argv
    out_dir = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    tmp = Path(tempfile.mkdtemp(prefix="diap_smoke_"))
    work = tmp / "작업 폴더 (smoke)"                    # spaces and Korean in every path the run makes
    work.mkdir()
    # the app's own data folder stays untouched: settings, cache and logs of this run are the run's
    import contour.core.settings as S
    S._settings = S.AppSettings(data_dir=str(tmp / "data"), cache_dir=str(tmp / "cache"), proxy_mode="off")
    S.AppSettings.save = lambda self: None
    state: dict = {}

    @step("runtime")
    def _():
        from contour.track import backend as BK
        d = BK.diagnostics()
        return dict(system=d["system"], python=d["python"], onnxruntime=d["onnxruntime"], available=d["available"],
                    offered=d["offered"], packages=d["packages"], gpu=d["gpu"], warnings=d["warnings"],
                    app=S.APP_VERSION, frozen=bool(getattr(sys, "frozen", False)))

    @step("ffmpeg")
    def _():
        from contour.core import media as MD
        f, p = MD.configure_ffmpeg()
        assert f and p, "ffmpeg / ffprobe not found"
        v = subprocess.run([f, "-hide_banner", "-version"], capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
        enc = subprocess.run([f, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
        assert " prores_ks " in enc, "this ffmpeg has no ProRes encoder"
        pv = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
        return dict(ffmpeg=v[:80], ffprobe=pv[:60], x265=" libx265 " in enc, nvenc=" hevc_nvenc " in enc,
                    videotoolbox=" hevc_videotoolbox " in enc, own=Path(f).parent != Path(shutil.which("ffmpeg") or "x").parent)

    @step("project")
    def _():
        from contour.core import media as MD, relink as RL
        from contour.core.project import Project
        if FIXTURE.is_dir():
            # the frozen project names its original by the Windows path it was saved with: connected again here
            dst = work / "baseline.dpprj"
            shutil.copytree(FIXTURE / "project", dst)
            media = work / "원본 미디어"
            media.mkdir()
            shutil.copyfile(FIXTURE / "src.mkv", media / "src.mkv")
            prj = Project.load(str(dst))
            miss = RL.missing(prj)
            assert miss, "the frozen project's Windows path was found on this computer?"
            res = RL.relink(prj, media)
            assert res["linked"] and not res["different"] and not res["not_found"], res
            prj.save()
            kind = "frozen 1.22.2 project (Windows path connected again)"
        else:
            src = work / "원본 src.mov"
            subprocess.run([MD.ffmpeg_bin(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=192x144:r=24:d=0.25",
                            "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le", str(src)], check=True)
            prj = Project.create(str(work), "smoke")
            prj.add_cut(MD.probe(str(src)), "c1")
            prj.save()
            kind = "made here (the tests are not installed)"
        prj = Project.load(prj.path)
        cut = prj.cuts[0]
        from contour.core.frames import frames_for
        fr = frames_for(prj, cut)
        try:
            f0 = fr.frame(0)
            last = fr.frame(cut.info.frames - 1)
        finally:
            fr.close()
        assert f0 is not None and f0.ndim == 3 and last is not None
        state.update(prj=prj, cut=cut)
        return dict(kind=kind, cut=cut.name, frames=cut.info.frames, size=[cut.info.width, cut.info.height],
                    pix_fmt=cut.info.pix_fmt, bit_depth=cut.info.bit_depth, analysed=bool(cut.analyzed),
                    persons=len(cut.persons), first_frame=list(f0.shape), from_="original")

    @step("proxy")
    def _():
        from contour.core.colour import BUILTIN_STUDIO, manager
        from contour.core.frames import proxy_for
        from contour.ui.session import EditSession
        prj, cut = state["prj"], state["cut"]
        s = EditSession()
        s.project = prj
        vs = s.view_settings(cut)
        px = proxy_for(prj, cut)
        ok = px.build(manager(BUILTIN_STUDIO), vs)
        assert ok and px.is_valid(vs.key()), "the proxy was not made whole"
        px.open()
        try:
            f = px.frame(0)
        finally:
            px.close()
        assert f is not None and f.shape[2] == 3
        return dict(size=[px.w, px.h], frames=px.n, valid=True)

    @step("inference")
    def _():
        import numpy as np
        import onnxruntime as ort
        from contour.track import backend as BK, models as M
        out = {}
        rng = np.random.default_rng(7)
        f32 = lambda *shape: rng.random(shape).astype(np.float32)
        data, onnx = ROOT / "contour" / "track" / "data", ROOT / "assets" / "onnx"
        # the models that are part of the app, each with an input of the kind it gets in use
        files = {"spotnet_v1.onnx": (data / "spotnet_v1.onnx", lambda: {"x": f32(1, 4, 128, 128)}),
                 "mp_selfie_multiclass_256.onnx": (data / "mp_selfie_multiclass_256.onnx",
                                                   lambda: {"input_29": f32(1, 256, 256, 3)}),
                 "warp_plane.onnx": (onnx / "warp_plane.onnx",
                                     lambda: {"D": (f32(1, 2, 18, 24) - 0.5) * 4, "P": f32(1, 1, 144, 192) * 1023,
                                              "size": np.array([1, 2, 144, 192], np.int64),
                                              "steps": np.array([1, 1], np.int64)}),
                 "reid_osnet_ibn_x1_0.onnx": (onnx / "reid_osnet_ibn_x1_0.onnx", lambda: {"input": f32(2, 3, 256, 128)})}
        for name, (f, make) in files.items():
            if not f.is_file():
                out[name] = "not in this install"
                continue
            s = M.session(f)
            feed = make()
            t0 = time.perf_counter()
            got = s.run(None, feed)
            ms = (time.perf_counter() - t0) * 1000
            rec = next((r for r in BK.records() if r["file"] == f.name), {})
            d = dict(backend=rec.get("backend"), asked=rec.get("asked"), first_ms=round(ms, 1),
                     reasons=rec.get("reasons") or [])
            if rec.get("backend") not in (None, "", "cpu"):
                # the same input on the CPU: what the accelerator's answer differs by
                ref = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"]).run(None, feed)
                d["max_diff_vs_cpu"] = float(max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()
                                                 for a, b in zip(got, ref)))
            out[name] = d
        assert any(isinstance(v, dict) for v in out.values()), "no model that comes with the app was found"
        return dict(models=out, device=M.device_label())

    @step("export")
    def _():
        import numpy as np
        from contour.core import media as MD
        from contour.core.frames import frames_for
        from contour.core.project import OutputSettings
        from contour.track.analyze import CutTracker
        from contour.warp.render import render_cut
        prj, cut = state["prj"], state["cut"]
        dst = out_dir or (work / "내보내기 out")
        o = OutputSettings(format="prores_4444", out_dir=str(dst), include_audio=False)
        gold = {}
        golden = FIXTURE / "golden.npz"
        if golden.is_file() and FIXTURE.is_dir():
            # first, with the project as it was saved (the export below may add what an older project lacks): the
            # planes this system hands to the encoder, against the ones DIAP 1.22.2 made (frozen with the tests)
            sys.path.insert(0, str(ROOT / "tests"))
            import baseline_util as BU
            g = np.load(golden)
            with BU.NoModels():
                pl = BU.render_planes(prj, cut)
            diff = np.abs(pl.astype(int) - g["planes"].astype(int))
            gold = dict(golden_env=str(g["env"]), env=BU.env_key(), planes_max_diff=int(diff.max()),
                        planes_differ=int((diff > 0).sum()), planes_n=int(diff.size), tolerance=TOLERANCE)
        fr = frames_for(prj, cut)
        try:
            tr = CutTracker(prj, cut, fr)
            rep = render_cut(prj, cut, tr, o)
        finally:
            fr.close()
        files = sorted(Path(dst).glob("*.mov"))
        assert files, "no file was written"
        info = MD.probe(str(files[0]))
        a, b = cut.work_range
        assert info.frames == b - a, f"{info.frames} frames written, {b - a} asked"
        assert (info.width, info.height) == (cut.info.width, cut.info.height) and info.bit_depth >= 10
        d = dict(file=files[0].name, frames=info.frames, size=[info.width, info.height], pix_fmt=info.pix_fmt,
                 bit_depth=info.bit_depth, fps=info.fps_fraction, timecode=info.timecode,
                 seconds_render=round(float(getattr(rep, "seconds", 0.0) or 0.0), 2))
        d.update(gold)
        if gold:
            assert gold["planes_max_diff"] <= TOLERANCE, \
                f"the render differs from the frozen one by {gold['planes_max_diff']} of 1023"
        return d

    @step("worker")
    def _():
        from contour.ui import batchrun as BR
        exe, args = BR.worker_command(["--help"])
        r = subprocess.run([exe] + args, capture_output=True, text=True, timeout=180, cwd=str(ROOT),
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"), encoding="utf-8",
                           errors="replace")
        assert r.returncode == 0 and "analyze" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-400:]
        return dict(program=Path(exe).name, arguments=args[:2])

    RES["platform"] = f"{platform.system()} {platform.release()} {platform.machine()}"
    line = json.dumps(RES, ensure_ascii=False, default=str)
    print("SMOKE " + line, flush=True)
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(line, encoding="utf-8")
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if RES["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
