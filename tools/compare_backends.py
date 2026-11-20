"""Every model on every backend this computer offers, held against the CPU with the same input (instruction
2026-11-20 §5): what the hardware table of a release is filled from, and how a model gets onto Core ML's list.

    python tools/compare_backends.py [--image FILE] [--models K,K] [--skip-cpu-slow] [--runs N] [--json FILE]

For each model that is on this computer (the ones that are part of the app, and the downloaded or packaged ones;
nothing is downloaded here): opened through the app's own model session on the CPU and on each accelerator of this
system, run with one realistic input (the picture given, else a made-up one), and reported:
- which backend the run was really on (operators and time per provider from the run's own record) and the reasons
  the session gives when it ended on the CPU,
- the largest and the mean difference of its outputs from the CPU's,
- milliseconds a run (the median of N after a first one), and for the first run,
- for an NVIDIA GPU the memory the process held after the runs.
Prints a table and one JSON line (BACKENDS ...). --skip-cpu-slow leaves the CPU run of the SAM encoders out (minutes
a picture): they are then listed without a difference. Nothing is sent anywhere; the picture is only read."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CONTOUR_NO_UPDATE", "1")

import numpy as np  # noqa: E402

SLOW_ON_CPU = ("seg_sam31", "seg_sam31_find", "seg_sam2_s", "seg_sam2_b")


def picture(path: str = "") -> np.ndarray:
    """RGB uint8, 1080p: the file's first frame / the image, else shapes and texture a model has something to do with."""
    import cv2
    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            cap = cv2.VideoCapture(path)
            ok, img = cap.read()
            cap.release()
            if not ok:
                raise SystemExit(f"읽지 못함: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        s = 1920 / max(img.shape[:2])
        return cv2.resize(img, (int(img.shape[1] * s) // 2 * 2, int(img.shape[0] * s) // 2 * 2), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(11)
    H, W = 1080, 1920
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([70 + 60 * xx / W, 80 + 50 * yy / H, 100 + 30 * np.sin(xx / 40)], -1)
    for cx, cy, r in ((700, 380, 150), (1250, 420, 120)):                 # two 'heads' on 'bodies'
        cv2.ellipse(img, (cx, cy), (r, int(r * 1.25)), 0, 0, 360, (205, 160, 135), -1)
        cv2.rectangle(img, (cx - 2 * r, cy + int(1.2 * r)), (cx + 2 * r, H), (60, 90, 150), -1)
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def feeds(rgb: np.ndarray) -> dict:
    """model key -> (path of the graph, function giving {input name: array}); only models on this computer."""
    import cv2
    from contour.track import models as M
    data, onnx = ROOT / "contour" / "track" / "data", ROOT / "assets" / "onnx"
    f32 = np.float32

    def chw(size_wh, mean=0.0, std=1.0, n=1, bgr=False, scale=1.0):
        im = cv2.resize(rgb, size_wh, interpolation=cv2.INTER_AREA).astype(f32)
        if bgr:
            im = im[..., ::-1]
        x = ((im * scale - mean) / std).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(np.repeat(x, n, 0).astype(f32))
    MEAN, STD = np.array([123.675, 116.28, 103.53], f32), np.array([58.395, 57.12, 57.375], f32)
    out = {}

    def add(key, path, make):
        if Path(path).is_file():
            out[key] = (Path(path), make)
    # part of the app
    add("spotnet", data / "spotnet_v1.onnx",
        lambda: {"x": np.concatenate([chw((256, 256), scale=1 / 255.0), np.ones((1, 1, 256, 256), f32)], 1)})
    add("mp_selfie_multiclass", data / "mp_selfie_multiclass_256.onnx",
        lambda: {"input_29": np.ascontiguousarray(cv2.resize(rgb, (256, 256)).astype(f32)[None] / 255.0)})
    add("warp_plane", onnx / "warp_plane.onnx",
        lambda: {"D": (np.random.default_rng(3).random((1, 2, 34, 60)).astype(f32) - 0.5) * 6,
                 "P": np.ascontiguousarray(cv2.resize(rgb, (1920, 1080))[..., 1].astype(f32)[None, None] * 4.0),
                 "size": np.array([1, 2, 1080, 1920], np.int64), "steps": np.array([1, 1], np.int64)})
    add("reid_osnet", onnx / "reid_osnet_ibn_x1_0.onnx", lambda: {"input": chw((128, 256), MEAN, STD, n=4)})
    # downloaded by the app or installed as packages
    for key in ("det_yolox_m", "det_yolox_x"):
        if M.is_downloaded(key):
            add(key, M.model_path(key), lambda: {None: chw((640, 640), bgr=True)})
    for key in ("pose_rtmpose_x", "pose_rtmw_x", "pose_rtmpose_m", "pose_dwpose_l", "face_rtm106", "hand_rtm21"):
        if M.is_downloaded(key):
            w, h = M.MODELS[key]["input"]
            add(key, M.model_path(key), lambda w=w, h=h: {None: chw((w, h), MEAN, STD, n=4)})
    if M.is_downloaded("matte_rvm"):
        z = np.zeros((1, 1, 1, 1), f32)
        add("matte_rvm", M.model_path("matte_rvm"),
            lambda: {"src": chw((960, 540), scale=1 / 255.0), "r1i": z, "r2i": z, "r3i": z, "r4i": z,
                     "downsample_ratio": np.array([0.5], f32)})
    for key in ("parse_clothes", "face_parse"):
        if M.is_downloaded(key):
            add(key, M.model_path(key), lambda: {None: chw((512, 512), MEAN, STD)})
    if M.is_downloaded("seg_sam31"):
        d = M.model_path("seg_sam31")
        enc = d / "image-encoder-gpu.onnx"
        add("seg_sam31", enc if enc.is_file() else d / "image-encoder.onnx",
            lambda: {None: np.ascontiguousarray((cv2.resize(rgb, (1008, 1008), interpolation=cv2.INTER_AREA).astype(f32)
                                                 / 255.0 * 2.0 - 1.0).transpose(2, 0, 1)[None])})
    for key in ("seg_sam2_s", "seg_sam2_b"):
        if M.is_downloaded(key):
            add(key, M.model_path(key) / "vision_encoder.onnx",
                lambda: {None: chw((1024, 1024), np.array([0.485, 0.456, 0.406], f32) * 255,
                                   np.array([0.229, 0.224, 0.225], f32) * 255)})
    return out


def vram_mb() -> int:
    try:
        kw = dict(capture_output=True, text=True, timeout=8)
        if os.name == "nt":
            kw["creationflags"] = 0x08000000
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], **kw)
        for ln in (r.stdout or "").splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) >= 2 and p[0] == str(os.getpid()) and p[1].isdigit():
                return int(p[1])
    except Exception:
        pass
    return 0


def run_one(path: Path, make, backend: str, runs: int) -> dict:
    """The model opened by the app's own session with 추론 장치 = backend, run 1 + runs times."""
    from contour.core.settings import settings
    from contour.track import backend as BK, models as M
    settings().gpu = backend
    M.reset_providers()
    M._verified.clear()
    BK.forget()
    M.take_cpu_fallbacks()
    t0 = time.perf_counter()
    s = M.session(path)
    t_open = time.perf_counter() - t0
    feed = make()
    if None in feed:                                   # one input: by the graph's own name for it
        v = feed.pop(None)
        i0 = s.get_inputs()[0]
        feed[i0.name] = v.astype(np.float16) if "float16" in i0.type else v
    names = {i.name for i in s.get_inputs()}
    feed = {k: v for k, v in feed.items() if k in names}
    t0 = time.perf_counter()
    out = s.run(None, feed)
    first = time.perf_counter() - t0
    ts = []
    for _ in range(max(0, runs)):
        t0 = time.perf_counter()
        out = s.run(None, feed)
        ts.append(time.perf_counter() - t0)
    rec = next((r for r in BK.records() if r["file"] == Path(path).name), {})
    d = dict(asked=rec.get("asked"), session=rec.get("session"), backend=rec.get("backend"),
             placement=rec.get("placement") or {}, cpu_share=rec.get("cpu_share"), reasons=rec.get("reasons") or [],
             open_s=round(t_open, 2), first_ms=round(first * 1000, 1),
             ms=round(float(np.median(ts)) * 1000, 2) if ts else None, vram_mb=vram_mb())
    del s
    return d, [np.asarray(o) for o in out]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    arg = lambda n, d="": argv[argv.index(n) + 1] if n in argv else d
    runs = int(arg("--runs", "5"))
    only = [k for k in arg("--models").split(",") if k]
    skip_slow = "--skip-cpu-slow" in argv
    import contour.core.settings as S
    S.AppSettings.save = lambda self: None
    from contour.track import backend as BK, models as M
    rgb = picture(arg("--image"))
    todo = feeds(rgb)
    if only:
        todo = {k: v for k, v in todo.items() if k in only}
    S.settings().gpu = "auto"
    M.reset_providers()
    acc = [b for b in BK.offered() if b != "cpu"]
    res = dict(system=BK.diagnostics(), accelerators=acc, picture=list(rgb.shape), runs=runs, models={})
    print(BK.text(res["system"]).split("\n\n")[0], flush=True)
    print(f"\n{'모델':<24}{'장치':<9}{'실제':<8}{'연산(가속/CPU)':<16}{'CPU시간':>7}{'첫 실행':>10}{'실행':>10}{'CPU 대비 최대차':>16}{'평균차':>12}  사유",
          flush=True)
    for key, (path, make) in todo.items():
        row = {}
        ref = None
        if not (skip_slow and key in SLOW_ON_CPU):
            try:
                row["cpu"], ref = run_one(path, make, "cpu", runs if key not in SLOW_ON_CPU else 0)
            except Exception as e:
                row["cpu"] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
        for b in acc:
            try:
                d, out = run_one(path, make, b, runs)
                if ref is not None:
                    diffs = [np.abs(a.astype(np.float64) - c.astype(np.float64)) for a, c in zip(out, ref)
                             if a.shape == c.shape and a.size]
                    if diffs:
                        d["max_diff"] = float(max(x.max() for x in diffs))
                        d["mean_diff"] = float(np.mean([x.mean() for x in diffs]))
                        d["out_range"] = float(max(np.abs(c).max() for c in ref if c.size))
                row[b] = d
            except Exception as e:
                row[b] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
        res["models"][key] = dict(file=path.name, **row)
        for b, d in row.items():
            if "error" in d:
                print(f"{key:<24}{BK.LABEL.get(b, b):<9}오류: {d['error']}", flush=True)
                continue
            pl = d.get("placement") or {}
            a_n = sum(v["nodes"] for p, v in pl.items() if BK.BACKEND.get(p, "cpu") != "cpu")
            c_n = sum(v["nodes"] for p, v in pl.items() if BK.BACKEND.get(p, "cpu") == "cpu")
            print(f"{key:<24}{BK.LABEL.get(b, b):<9}{BK.LABEL.get(d.get('backend') or '', d.get('backend') or '-'):<8}"
                  f"{(str(a_n) + '/' + str(c_n)) if pl else '-':<16}"
                  f"{(format(d['cpu_share'] * 100, '.0f') + '%') if pl and d.get('cpu_share') is not None else '-':>7}"
                  f"{d['first_ms']:>8.0f}ms{(format(d['ms'], '.1f') + 'ms') if d.get('ms') is not None else '-':>10}"
                  f"{format(d['max_diff'], '.2e') if 'max_diff' in d else '-':>16}"
                  f"{format(d['mean_diff'], '.2e') if 'mean_diff' in d else '-':>12}  {'; '.join(d.get('reasons') or [])}",
                  flush=True)
    line = json.dumps(res, ensure_ascii=False, default=str)
    print("BACKENDS " + line, flush=True)
    if "--json" in argv:
        Path(arg("--json")).write_text(line, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
