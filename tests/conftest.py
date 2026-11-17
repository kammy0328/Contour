"""Shared fixtures: an isolated home (settings, data folder), a tiny project made from generated frames, and an
offscreen Qt application. No model is needed."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["CONTOUR_NO_UPDATE"] = "1"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    import contour.core.settings as S
    S._settings = None
    yield h
    S._settings = None


def make_video(path: Path, n: int = 6, w: int = 64, h: int = 48, fps: int = 24, fmt: str = "mp4") -> Path:
    """n frames of a moving gradient, written with ffmpeg (rawvideo in)."""
    from contour.core import media as MD
    MD.configure_ffmpeg()
    frames = []
    for i in range(n):
        x = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
        f = np.clip(x + i * 10, 0, 255).repeat(h, 0).repeat(3, 2).astype(np.uint8)
        frames.append(f)
    cmd = [MD.ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-"]
    if fmt == "mov":
        cmd += ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv444p", "-qp", "0"]
    cmd += [str(path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return path


@pytest.fixture
def project(tmp_path):
    """A project with two plain cuts (no analysis)."""
    from contour.core import media as MD
    from contour.core.project import Project
    v1 = make_video(tmp_path / "a.mp4")
    v2 = make_video(tmp_path / "b.mp4")
    prj = Project.create(str(tmp_path), "T")
    prj.add_cut(MD.probe(str(v1)), "a")
    prj.add_cut(MD.probe(str(v2)), "b")
    prj.save()
    return prj


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from contour.ui import theme
    theme.apply(app)
    return app
