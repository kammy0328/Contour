"""Run by the installer after it copied the files: packages, GPU runtime, ffmpeg and models.

Every step is skipped when it is already done, so an update whose packages did not change finishes in
seconds. Output also goes to {app}\\setup_log.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

APP = Path(__file__).resolve().parents[1]          # {app}\app
ROOT = APP.parent                                  # {app}
PY = Path(sys.executable)
STAMP = PY.parent / ".contour_packages"
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
LOG = ROOT / "setup_log.txt"


def say(*a):
    t = " ".join(str(x) for x in a)
    print(t, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(t + "\n")
    except OSError:
        pass


def run(args) -> int:
    say(">", " ".join(str(a) for a in args))
    p = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding="utf-8", errors="replace")
    out = p.stdout or ""
    print(out, flush=True)
    tail = "\n".join(out.strip().splitlines()[-15:])
    if tail:
        say(tail)
    return p.returncode


def pip_ok() -> bool:
    return subprocess.run([str(PY), "-m", "pip", "--version"], capture_output=True).returncode == 0


def has_torch() -> bool:
    return subprocess.run([str(PY), "-c", "import torch"], capture_output=True).returncode == 0


def req_key(learn: bool) -> str:
    h = hashlib.sha256((APP / "requirements.txt").read_bytes())
    if learn:
        h.update((APP / "requirements-learn.txt").read_bytes())
    return h.hexdigest()


ORT_VER = "1.24.4"                    # the runtime requirements.txt pins (onnxruntime-gpu[cuda,cudnn]==1.24.4)


def gpu(path=None) -> str:
    """What a model really ran on here (tools/check_gpu.py: CUDA / DML / CPU / NONE). path: a folder with another
    runtime to try instead of the installed one."""
    args = [str(PY), str(APP / "tools" / "check_gpu.py")] + (["--path", str(path)] if path else [])
    p = subprocess.run(args, capture_output=True, text=True)
    lines = [l for l in (p.stdout or "").splitlines() if l.strip()]
    return lines[-1].split()[0] if lines else "NONE"


def runtimes() -> list:
    """The onnxruntime packages installed in this Python (one is right; two break each other)."""
    code = ("from importlib import metadata as m; print(' '.join(sorted({(d.metadata.get('Name') or '').lower().replace('_', '-') "
            "for d in m.distributions() if (d.metadata.get('Name') or '').lower().replace('_', '-') in "
            "('onnxruntime', 'onnxruntime-gpu', 'onnxruntime-directml')})))")
    p = subprocess.run([str(PY), "-c", code], capture_output=True, text=True)
    return (p.stdout or "").split()


def gpu_step(run=None, gpu=None, runtimes=None, say=None, alt_dir=None) -> str:
    """CUDA when it works. When it does not: DirectML is tried in a folder of its own first, and only when a model
    really ran on it there is the installed runtime changed for it (from the files already downloaded for the
    trial). Anything else (no connection, DirectML not working here, the change failing) leaves or puts back the
    runtime that works, which then runs on the CPU. Never two runtimes at once. Returns what models run on."""
    import shutil as _sh
    run = run or globals()["run"]
    gpu = gpu or globals()["gpu"]
    runtimes = runtimes or globals()["runtimes"]
    say = say or globals()["say"]
    alt = Path(alt_dir) if alt_dir else ROOT / "ort_alt"
    have = runtimes()
    if len(have) > 1:
        # left by an interrupted change: both removed, the pinned one installed again
        say(f"[3/5] ONNX Runtime 패키지가 겹쳐 설치되어 있어 정리합니다: {', '.join(have)}")
        run([PY, "-m", "pip", "uninstall", "-y"] + have)
        run([PY, "-m", "pip", "install", "--no-warn-script-location", f"onnxruntime-gpu[cuda,cudnn]=={ORT_VER}"])
    g = gpu()
    if g in ("CUDA", "DML"):
        return g
    if g == "NONE":
        say("[3/5] ONNX Runtime을 불러오지 못했습니다. 설치 파일을 다시 실행하세요")
        return g
    if "onnxruntime-directml" in runtimes():
        say("[3/5] DirectML 런타임이 설치되어 있지만 GPU를 쓰지 못했습니다. CPU로 실행됩니다")
        return g
    say("[3/5] CUDA를 쓸 수 없습니다. DirectML이 이 PC에서 되는지 먼저 확인합니다 (지금 런타임은 그대로 둡니다)")
    try:
        _sh.rmtree(alt, ignore_errors=True)
        wheels, probe = alt / "wheels", alt / "probe"
        ok = run([PY, "-m", "pip", "download", "--no-deps", "-d", wheels, f"onnxruntime-directml=={ORT_VER}"]) == 0 \
            and run([PY, "-m", "pip", "install", "--no-deps", "--no-index", "--find-links", wheels, "--target", probe,
                     f"onnxruntime-directml=={ORT_VER}"]) == 0 \
            and gpu(probe) == "DML"
        if not ok:
            say("[3/5] DirectML을 확인하지 못해 런타임을 바꾸지 않았습니다. 지금은 CPU로 실행됩니다 "
                "(NVIDIA 드라이버를 업데이트하고 설치 파일을 다시 실행하면 CUDA로 돌아갑니다)")
            return g
        say("[3/5] DirectML이 확인되어 런타임을 바꿉니다 (NVIDIA 드라이버를 업데이트하면 다시 설치할 때 CUDA로 돌아갑니다)")
        run([PY, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu"])
        rc = run([PY, "-m", "pip", "install", "--no-deps", "--no-index", "--find-links", wheels,
                  f"onnxruntime-directml=={ORT_VER}"])
        g2 = gpu()
        if rc != 0 or g2 == "NONE":
            say("[3/5] 바꾸지 못해 이전 런타임으로 되돌립니다")
            run([PY, "-m", "pip", "uninstall", "-y", "onnxruntime-directml"])
            run([PY, "-m", "pip", "install", "--no-warn-script-location", f"onnxruntime-gpu[cuda,cudnn]=={ORT_VER}"])
            g2 = gpu()
        return g2
    finally:
        _sh.rmtree(alt, ignore_errors=True)


RTX_EP = "onnxruntime-ep-nv-tensorrt-rtx-cu12==0.5.0"      # the version checked (contour/track/backend.py RTX_VERSION)


def gpu_name() -> str:
    """The first NVIDIA GPU's name ('' when nvidia-smi does not answer)."""
    try:
        kw = dict(capture_output=True, text=True, timeout=10)
        if os.name == "nt":
            kw["creationflags"] = 0x08000000
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], **kw)
        return (r.stdout or "").strip().splitlines()[0].strip() if r.returncode == 0 and (r.stdout or "").strip() else ""
    except Exception:
        return ""


def rtx_devices() -> int:
    """How many GPUs of this computer NVIDIA's TensorRT for RTX plugin takes (asked in a child process; -1: not installed or
    it could not be asked)."""
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from contour.track import backend as BK\n"
            "r = BK.rtx()\n"
            "print('RTX', len(r['devices']) if r.get('state') in ('ok', 'no_device') else -1, r.get('state'), r['why'])") % str(APP)
    p = subprocess.run([str(PY), "-c", code], capture_output=True, text=True)
    for ln in (p.stdout or "").splitlines():
        if ln.startswith("RTX "):
            try:
                return int(ln.split()[1])
            except (ValueError, IndexError):
                return -1
    return -1


def rtx_step(g: str, run=None, say=None, name=None, devices=None) -> str:
    """(1.27) NVIDIA's TensorRT for RTX provider beside the CUDA runtime, on a computer whose GPU it runs on (RTX 30 series and
    later): SAM 3.1's encoder, 42 % of an analysis, runs six times faster on it. A plugin of the runtime, about 115 MB to download;
    nothing depends on it: no GPU for it, no connection, an install that fails all leave the app on CUDA as before. Returns
    'on' | 'off' | 'none' (not for this computer)."""
    run = run or globals()["run"]
    say = say or globals()["say"]
    name = gpu_name() if name is None else name
    devices = devices or rtx_devices
    if g != "CUDA" or "RTX" not in name.upper():
        return "none"
    n = devices()
    if n > 0:
        return "on"
    if n == 0:
        # installed before, and this GPU is not one it takes: its 280 MB are of no use here
        run([PY, "-m", "pip", "uninstall", "-y", RTX_EP.split("==")[0]])
        return "none"
    say(f"[3/5] {name}: TensorRT RTX 가속 구성요소 설치 (약 115MB)")
    if run([PY, "-m", "pip", "install", "--no-warn-script-location", "--no-deps", RTX_EP]) != 0:
        say("[3/5] TensorRT RTX 구성요소를 받지 못했습니다. CUDA로 실행합니다 (설치 파일을 다시 실행하면 다시 시도합니다)")
        return "off"
    n = devices()
    if n > 0:
        return "on"
    say("[3/5] 이 GPU는 TensorRT RTX 대상이 아니어서 구성요소를 다시 지웁니다 (CUDA로 실행)")
    run([PY, "-m", "pip", "uninstall", "-y", RTX_EP.split("==")[0]])
    return "none"


def ffmpeg_found() -> bool:
    # in a child process: the packages were installed after this one started (its import caches miss them)
    code = ("import sys; sys.path.insert(0, %r); from contour.core import media as M; "
            "print('FOUND' if M.configure_ffmpeg()[0] else 'NONE')") % str(APP)
    p = subprocess.run([str(PY), "-c", code], capture_output=True, text=True)
    if "FOUND" in (p.stdout or ""):
        return True
    if "NONE" not in (p.stdout or ""):
        raise RuntimeError((p.stderr or "").strip().splitlines()[-1] if p.stderr else "check failed")
    return False


def get_ffmpeg():
    dst = ROOT / "ffmpeg" / "bin"
    say("[ffmpeg] 받는 중 (약 150MB)")
    req = urllib.request.Request(FFMPEG_URL, headers={"User-Agent": "Contour-setup"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in z.namelist():
            if "/bin/" in n and n.lower().endswith(".exe"):
                with z.open(n) as s, open(dst / Path(n).name, "wb") as d:
                    shutil.copyfileobj(s, d)
    say("[ffmpeg] 설치:", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", default="auto")       # 1 | 0 | auto (keep what is installed)
    ap.add_argument("--no-models", action="store_true")
    a = ap.parse_args()
    ci = bool(os.environ.get("CONTOUR_CI"))
    say(f"=== Contour 설치 마무리 · Python {sys.version.split()[0]} · {PY}")

    # 1. pip
    if not pip_ok():
        say("[1/5] pip 준비")
        if run([PY, PY.parent / "get-pip.py", "--no-warn-script-location"]) != 0:
            say("pip을 설치하지 못했습니다. 인터넷 연결을 확인하고 설치 파일을 다시 실행하세요.")
            return 1
    learn = has_torch() if a.learn == "auto" else a.learn == "1"

    # 2. packages (only when the requirements changed since the last install)
    key = req_key(learn)
    if not STAMP.exists() or STAMP.read_text().strip() != key:
        say("[2/5] 필수 패키지 설치 (처음엔 5~10분)")
        if run([PY, "-m", "pip", "install", "--no-warn-script-location", "-r", APP / "requirements.txt"]) != 0:
            say("패키지 설치에 실패했습니다. 인터넷 연결을 확인하고 설치 파일을 다시 실행하세요.")
            return 1
        if learn:
            say("      누적 학습 (PyTorch 약 2.5GB)")
            if run([PY, "-m", "pip", "install", "--no-warn-script-location", "-r",
                    APP / "requirements-learn.txt"]) != 0:
                say("      PyTorch 설치 실패 (분석, 보정, 렌더는 그대로 쓸 수 있습니다)")
                key = req_key(False)
        STAMP.write_text(key)
    else:
        say("[2/5] 패키지 변경 없음")

    # 3. GPU runtime: CUDA, else DirectML when it is seen to work here (the runtime that works is never taken
    #    out for one that was not tried)
    if not ci:
        g = gpu_step()
        say(f"[3/5] 추론 장치: {g}")
        try:
            r = rtx_step(g)
            if r != "none":
                say("[3/5] TensorRT RTX 가속: " + ("사용" if r == "on" else "쓰지 않음"))
        except Exception as e:
            say(f"[3/5] TensorRT RTX 확인 실패 ({e}): CUDA로 실행합니다")

    # 4. ffmpeg: packed into the installer ({app}\ffmpeg\bin); only checked here
    try:
        say("[4/5] ffmpeg 확인됨" if ffmpeg_found() else "[4/5] ffmpeg가 없습니다 (설치 파일 문제: 다시 설치하세요)")
    except Exception as e:
        say(f"[4/5] ffmpeg 확인 실패 ({e})")

    # 5. models (once; they live in the data folder and survive updates)
    if not (a.no_models or ci):
        say("[5/5] 모델 확인 (처음 한 번 약 1.1GB)")
        if not models_step(run):
            say("=== 일부 모델을 받지 못했습니다. 인터넷 연결을 확인하고 설치 파일을 다시 실행하세요 "
                "(처음 분석할 때도 다시 받으려고 시도합니다)")
            return 3
    say("=== 완료")
    return 0


def models_step(runner) -> bool:
    """Downloads the models; False when a required one could not be had (download_models exits non-zero)."""
    return runner([PY, APP / "tools" / "download_models.py", "--max"]) == 0


if __name__ == "__main__":
    sys.exit(main())
