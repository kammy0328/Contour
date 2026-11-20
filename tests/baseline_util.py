"""What the 1.22.2 baseline fixture is made with and compared by (tools/make_baseline_fixture.py, run with the
1.22.2 code, and tests/test_baseline_1222.py, run with the code of now): the same synthetic footage and tracks, the
same model-free settings, the same way of rendering. Nothing here knows which version it runs on."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

W, H, N = 192, 144, 6
MW, MH = W // 2, H // 2
PLACEHOLDER = "C:\\DIAP_Baseline\\src.mkv"        # the path the frozen project.json holds (a Windows project)
ALLOWED_NEW = ("matte_prompts.npz",)               # files a later version may add to an old project when it is opened


def env_key() -> str:
    """What the numbers of a render can depend on besides the code: where the same key is found again, the result
    must be the same bit for bit; elsewhere within a rounding step."""
    import platform
    import cv2
    return f"{platform.system()}-{platform.machine()}|numpy {np.__version__}|opencv {cv2.__version__}"


def frames_rgb() -> np.ndarray:
    """N frames (H, W, 3) uint8: a person in a medium close-up at the left (skin with fine texture and a few
    spots, a striped top), a smaller one at the right, on a textured background; everyone drifts a little."""
    rng = np.random.default_rng(1222)
    bg = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg[..., 0] = 60 + 40 * xx / W
    bg[..., 1] = 70 + 30 * yy / H
    bg[..., 2] = 90 + 20 * np.sin(xx / 9.0)
    bg += rng.normal(0, 3.0, (H, W, 1)).astype(np.float32)
    skin_tex = rng.normal(0, 6.0, (H, W)).astype(np.float32)
    out = []
    for t in range(N):
        f = bg.copy()
        for (cx, top, hw, body_w, col) in ((62 + t, 18, 30, 54, (40, 60, 150)), (150 - t, 40, 12, 22, (150, 60, 40))):
            # body: a striped top from the shoulders down
            x0, x1 = int(cx - body_w / 2), int(cx + body_w / 2)
            y0 = int(top + hw * 2.3)
            stripes = 25 * ((xx[y0:H, max(0, x0):min(W, x1)] // 4) % 2)
            f[y0:H, max(0, x0):min(W, x1)] = np.array(col, np.float32) + stripes[..., None]
            # head: skin with texture
            m = ((xx - cx) / hw) ** 2 + ((yy - (top + hw * 1.15)) / (hw * 1.25)) ** 2 <= 1.0
            f[m] = np.array((205, 160, 135), np.float32) + skin_tex[m][:, None]
            for dx, dy in ((-10, 8), (9, 14), (-3, 30), (12, 34)):        # spots
                sx, sy = int(cx + dx * hw / 30), int(top + hw * 0.6 + dy * hw / 30)
                if 1 <= sx < W - 1 and 1 <= sy < H - 1:
                    f[sy - 1:sy + 2, sx - 1:sx + 2] -= 45
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(out)


def write_source(path: Path) -> Path:
    """The frames as a lossless file (decoded bit for bit on every platform)."""
    from contour.core import media as MD
    MD.configure_ffmpeg()
    fr = frames_rgb()
    cmd = [MD.ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv444p", "-qp", "0", "-color_range", "tv", "-colorspace",
           "bt709", str(path)]
    subprocess.run(cmd, input=fr.tobytes(), check=True)
    return path


def face68(cx: float, top: float, w: float) -> np.ndarray:
    """68 landmarks of a frontal face whose jaw is w wide, the brows at top."""
    k = w / 230.0
    lm = np.zeros((68, 3), np.float64)
    a = np.linspace(np.pi, 0, 17)
    lm[:17, 0] = cx - np.cos(a) * w / 2
    lm[:17, 1] = top + 60 * k + np.sin(a) * w * 0.55
    lm[17:22, :2] = np.c_[np.linspace(cx - 90 * k, cx - 30 * k, 5), np.full(5, top)]
    lm[22:27, :2] = np.c_[np.linspace(cx + 30 * k, cx + 90 * k, 5), np.full(5, top)]
    lm[27:31, :2] = np.c_[np.full(4, cx), np.linspace(top + 20 * k, top + 70 * k, 4)]
    lm[31:36, :2] = np.c_[np.linspace(cx - 20 * k, cx + 20 * k, 5), np.full(5, top + 80 * k)]
    for a0, ex in ((36, cx - 60 * k), (42, cx + 60 * k)):
        ang = np.linspace(0, 2 * np.pi, 7)[:-1]
        lm[a0:a0 + 6, :2] = np.c_[ex + 18 * k * np.cos(ang), top + 25 * k + 7 * k * np.sin(ang)]
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    lm[48:60, :2] = np.c_[cx + 35 * k * np.cos(ang), top + 120 * k + 12 * k * np.sin(ang)]
    lm[60:68, :2] = np.c_[cx + 25 * k * np.cos(ang[::3].repeat(2)[:8]), top + 120 * k + 6 * k * np.sin(ang[::3].repeat(2)[:8])]
    lm[:, 2] = 1
    return lm


def make_project(folder: Path, src: Path):
    """The analysed project, written by whatever version of the code is imported (the fixture: 1.22.2)."""
    from contour.core import media as M
    from contour.core.params import skin_basic_values
    from contour.core.project import Curve, Person, Project
    from contour.track.store import TrackStore
    prj = Project.create(str(folder), "baseline")
    cut = prj.add_cut(M.probe(str(src)), "c1")
    cut.analyzed = True
    cut.analysis_version = 2
    cut.cs_analysed = ""
    cut.persons = [Person(id=0, name="A", color="#ff8800", role="target"),
                   Person(id=1, name="B", color="#00aaff", role="protect")]
    pp = cut.params_for(0)
    pp.amounts["waist"] = Curve(base=25.0)
    for k, v in skin_basic_values("std", ["acne"]).items():
        pp.amounts[k] = Curve(base=float(v))
    pp.skin_basic = {"level": "std", "remove": ["acne"]}
    st = TrackStore(N, [0, 1])
    st.has_vis = st.has_face = True
    for p, (cx0, top, hw, body_w, sign) in enumerate(((62, 18, 30, 54, 1), (150, 40, 12, 22, -1))):
        for t in range(N):
            cx = cx0 + sign * t
            sh_y = top + hw * 2.4
            kp = np.array([[cx, top + hw * 1.3], [cx + hw * 0.35, top + hw], [cx - hw * 0.35, top + hw],
                           [cx + hw * 0.8, top + hw * 1.2], [cx - hw * 0.8, top + hw * 1.2],
                           [cx + body_w * 0.5, sh_y], [cx - body_w * 0.5, sh_y],
                           [cx + body_w * 0.6, sh_y + hw * 1.3], [cx - body_w * 0.6, sh_y + hw * 1.3],
                           [cx + body_w * 0.55, sh_y + hw * 2.4], [cx - body_w * 0.55, sh_y + hw * 2.4],
                           [cx + body_w * 0.3, sh_y + hw * 2.6], [cx - body_w * 0.3, sh_y + hw * 2.6],
                           [cx + body_w * 0.3, sh_y + hw * 4.2], [cx - body_w * 0.3, sh_y + hw * 4.2],
                           [cx + body_w * 0.3, sh_y + hw * 5.8], [cx - body_w * 0.3, sh_y + hw * 5.8]], np.float32)
            st.final[p, t, :, :2] = kp
            st.final[p, t, :, 2] = 0.9
            st.boxes[p, t] = [cx - body_w * 0.7, top - 4, cx + body_w * 0.7, H + 40]
            st.present[p, t] = True
            lm = face68(cx, top + hw * 0.55, hw * 1.7)
            st.face[p, t] = lm.astype(np.float32)
            st.face_raw[p, t] = lm.astype(np.float32)
            st.face_vis[p, t] = 1.0
        st.scale[p] = hw * 6.0
        st.conf[p] = 0.9
        st.widths[p] = body_w * 0.5
    st.save(prj.cut_dir(cut) / "tracks.npz")
    cache = prj.cache_dir(cut)
    mm = np.memmap(cache / f"matte_{MW}x{MH}.u8", np.uint8, "w+", shape=(N, MH, MW))
    lm_ = np.memmap(cache / f"seglab_{MW}x{MH}.u8", np.uint8, "w+", shape=(N, MH, MW))
    yy, xx = np.mgrid[0:MH, 0:MW].astype(np.float32)
    for t in range(N):
        for p, (cx0, top, hw, body_w, sign) in enumerate(((62, 18, 30, 54, 1), (150, 40, 12, 22, -1))):
            cx = (cx0 + sign * t) / 2
            head = ((xx - cx) / (hw / 2)) ** 2 + ((yy - (top + hw * 1.15) / 2) / (hw * 1.25 / 2)) ** 2 <= 1.0
            body = (np.abs(xx - cx) <= body_w / 4) & (yy >= (top + hw * 2.3) / 2)
            m = head | body
            mm[t][m] = 255
            lm_[t][m] = p + 1
    mm.flush()
    lm_.flush()
    del mm, lm_
    (cache / "seg_meta.json").write_text(json.dumps(dict(model="SAM 3.1", made=1795000000.0, refreshed=[],
                                                         range=[0, N])), encoding="utf-8")
    # widths and outline points measured on the matte, as the end of an analysis leaves them
    from contour.core.frames import frames_for
    from contour.track.analyze import CutTracker
    fr = frames_for(prj, cut)
    tr = CutTracker(prj, cut, fr)
    tr.update_widths()
    tr.save()
    fr.close()
    prj.save()
    return prj, cut


class NoModels:
    """No model file is opened (the baseline must not depend on what is installed): the face parser counts as
    absent, every session raises. Used as a context manager by the generator and by the test."""

    def __enter__(self):
        import contour.track.faceparse as FP
        import contour.track.models as MOD
        self._fp, self._mod = FP, MOD
        self._old = (FP.available, getattr(MOD, "session", None))
        FP.available = lambda: False

        def none(*a, **k):
            raise RuntimeError("no model in the baseline")
        MOD.session = none
        return self

    def __exit__(self, *exc):
        self._fp.available = self._old[0]
        if self._old[1] is not None:
            self._mod.session = self._old[1]
        return False


class Recorder:
    """Stands in for the output file of a render: keeps the planes handed to the encoder."""
    path = ""

    def __init__(self):
        self.frames = []

    def write(self, planes):
        self.frames.append([np.array(p) for p in planes])


def steady_rng() -> None:
    """The mask steadying fits its motion with OpenCV's LMEDS, which draws from OpenCV's own random generator: what
    it gives depends on how much was drawn before in the process (a code value here and there). The same seed
    before the fixture is made and before it is compared."""
    import cv2
    cv2.setRNGSeed(1222)


def render_planes(prj, cut) -> np.ndarray:
    """(frames, 3, H, W) uint16: the planes a ProRes 4444 render of the cut hands to the encoder."""
    steady_rng()
    from contour.core.frames import frames_for
    from contour.core.project import OutputSettings
    from contour.track.analyze import CutTracker
    from contour.warp.render import render_cut
    fr = frames_for(prj, cut)
    tr = CutTracker(prj, cut, fr)
    rec = Recorder()
    try:
        render_cut(prj, cut, tr, OutputSettings(format="prores_4444", out_dir=str(Path(prj.path) / "out"),
                                                include_audio=False), ext_writer=rec)
    finally:
        fr.close()
    return np.stack([np.stack([np.asarray(p, np.uint16) for p in f[:3]]) for f in rec.frames])


def preview_frames(prj, cut, frames=(0, 3, 5)) -> np.ndarray:
    """(len(frames), H', W', 3) uint8: what the viewer shows as the result of those frames."""
    from contour.ui.session import EditSession
    steady_rng()
    s = EditSession()
    s.project = prj
    s.open_cut(cut)
    try:
        return np.stack([np.asarray(s.result(int(t), final=True)) for t in frames])
    finally:
        s.open_cut(None)


def relink(project_dir: Path, src: Path) -> None:
    """The frozen project points at its original where it was made (a Windows path): pointed at src."""
    f = Path(project_dir) / "project.json"
    text = f.read_text(encoding="utf-8")
    a, b = json.dumps(PLACEHOLDER)[1:-1], json.dumps(str(src))[1:-1]
    assert a in text, "the fixture's project.json no longer holds the placeholder path"
    f.write_text(text.replace(a, b), encoding="utf-8")


def digest(folder: Path) -> dict:
    import hashlib
    out = {}
    for f in sorted(Path(folder).rglob("*")):
        if f.is_file():
            out[f.relative_to(folder).as_posix()] = hashlib.sha1(f.read_bytes()).hexdigest()
    return out


def copy_fixture(fixture: Path, tmp: Path):
    """The fixture copied to tmp and pointed at its source there: (project dir, source path)."""
    dst = Path(tmp) / "baseline.dpprj"
    shutil.copytree(Path(fixture) / "project", dst)
    src = Path(tmp) / "src.mkv"
    shutil.copyfile(Path(fixture) / "src.mkv", src)
    relink(dst, src)
    return dst, src
