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


def gpu() -> str:
    p = subprocess.run([str(PY), str(APP / "tools" / "check_gpu.py")], capture_output=True, text=True)
    lines = [l for l in (p.stdout or "").splitlines() if l.strip()]
    return lines[-1].split()[0] if lines else "NONE"


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

    # 3. GPU runtime: CUDA, else DirectML
    if not ci:
        g = gpu()
        if g not in ("CUDA", "DML"):
            say("[3/5] CUDA를 쓸 수 없어 DirectML로 바꿉니다 (NVIDIA 드라이버를 업데이트하면 CUDA로 돌아갑니다)")
            run([PY, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu"])
            run([PY, "-m", "pip", "install", "onnxruntime-directml==1.24.4"])
            g = gpu()
        say(f"[3/5] 추론 장치: {g}")

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
