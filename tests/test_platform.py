"""The same project and edit values on Windows and on a Mac (instruction 2026-11-20 §5 경로·패키징): where the data
and the cache are on each system (an existing Windows install keeps its own), ffmpeg and ffprobe found in each
kind of install, the child analysis started without opening a second window in a packed app, pause and cancel
reaching the child's own children, media connected again when its path is another one (another system, an external
volume, names with spaces and Korean), and the update file of each system."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------- data and cache
def test_where_the_data_and_the_cache_are_on_each_system(tmp_path, monkeypatch):
    import contour.core.settings as S
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    # Windows: where it always was
    monkeypatch.setattr(sys, "platform", "win32")
    assert S.default_data_dir() == home / "Documents" / "DIAP_Data"
    assert S.default_cache_dir(S.default_data_dir()) == home / "Documents" / "DIAP_Data" / "cache"
    (home / "Documents" / "Contour_Data").mkdir()
    assert S.default_data_dir() == home / "Documents" / "Contour_Data", "an install from before the rename keeps its folder"
    # a Mac: the system's places for an app's data and for what can be made again
    monkeypatch.setattr(sys, "platform", "darwin")
    data = home / "Library" / "Application Support" / "DIAP"
    assert S.default_data_dir() == data and S.default_cache_dir(data) == home / "Library" / "Caches" / "DIAP"
    (home / "DIAP_Data").mkdir()
    assert S.default_data_dir() == home / "DIAP_Data", "a source run's folder from before 1.24 goes on being used"
    assert S.default_cache_dir(home / "DIAP_Data") == home / "DIAP_Data" / "cache"
    monkeypatch.setattr(sys, "platform", "linux")
    assert S.default_data_dir() == home / "DIAP_Data"


def test_an_existing_windows_install_finds_its_settings_models_and_cache(tmp_path, monkeypatch):
    """Nothing is moved on Windows: the settings file, the models and the proxies of an install made by 1.22.2 are
    where the new version looks."""
    import contour.core.settings as S
    home = tmp_path / "home"
    data = home / "Documents" / "DIAP_Data"
    (data / "models" / "onnx").mkdir(parents=True)
    (data / "cache" / "proxies").mkdir(parents=True)
    (data / "models" / "onnx" / "matte_rvm.onnx").write_bytes(b"\x08" + b"m" * 3000)
    (data / "cache" / "proxies" / "p.mp4").write_bytes(b"p")
    # as 1.22.2 wrote it: no keys of later versions, one this version no longer has
    (data / "settings.json").write_text(json.dumps(dict(gpu="cuda", proxy_long_side=2560, last_project="D:\\job\\a.dpprj",
                                                        collect_learning_data=True, self_learning=True, ask_proxy=True,
                                                        gone_setting=1)), encoding="utf-8")
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(sys, "platform", "win32")
    S._settings = None
    s = S.settings()
    assert (s.gpu, s.proxy_long_side, s.last_project) == ("cuda", 2560, "D:\\job\\a.dpprj")
    assert s.data_path == data and s.cache_path == data / "cache" and (s.cache_path / "proxies" / "p.mp4").exists()
    assert s.model_registry == "" and s.learn_notified == [], "new keys take their defaults"
    from contour.track import models as M
    assert M.legacy_path("matte_rvm") == data / "models" / "onnx" / "matte_rvm.onnx" and M.is_downloaded("matte_rvm")
    S._settings = None


# ------------------------------------------------------------------------------------------------- ffmpeg
def _tool(d: Path, name: str, executable: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(0o755 if executable else 0o644)
    return p


@pytest.mark.skipif(sys.platform == "win32", reason="the layout of a packed Mac app")
def test_ffmpeg_and_ffprobe_are_found_inside_a_packed_app(tmp_path, monkeypatch):
    from contour.core import media as MD
    app = tmp_path / "DIAP 앱.app" / "Contents"
    (app / "MacOS").mkdir(parents=True)
    exe = app / "MacOS" / "DIAP"
    exe.write_text("x")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(app / "Frameworks"), raising=False)
    monkeypatch.setattr(MD.shutil, "which", lambda tool: "/usr/bin/" + tool)        # one elsewhere on the computer
    assert MD.configure_ffmpeg() == (None, None), "a packed app uses its own copy, never one found elsewhere"
    with pytest.raises(MD.MediaError):
        MD._tool_paths.update(ffmpeg=None, ffprobe=None)
        MD.ffmpeg_bin()
    f1 = _tool(app / "Resources" / "ffmpeg" / "bin", "ffmpeg")
    p1 = _tool(app / "Resources" / "ffmpeg" / "bin", "ffprobe")
    assert MD.configure_ffmpeg() == (str(f1), str(p1))
    f2 = _tool(app / "Frameworks" / "ffmpeg" / "bin", "ffmpeg")                    # where the packer's own folder is
    _tool(app / "Frameworks" / "ffmpeg" / "bin", "ffprobe", executable=False)       # not runnable: not taken
    assert MD.configure_ffmpeg() == (str(f2), str(p1))


def test_a_source_checkout_uses_the_one_on_the_path(monkeypatch):
    from contour.core import media as MD
    monkeypatch.delattr(sys, "frozen", raising=False)
    f, p = MD.configure_ffmpeg()
    assert f and p and Path(f).name.startswith("ffmpeg") and Path(p).name.startswith("ffprobe")


# ------------------------------------------------------------------------------------------------- the child job
def test_the_child_analysis_is_started_without_a_window(monkeypatch, tmp_path):
    from contour.ui import batchrun as BR
    exe, args = BR.worker_command(["analyze", "프로젝트 폴더/p.dpprj", "cut1", "--prescan", "0"])
    assert exe == sys.executable or Path(exe).name.lower() == "python.exe"
    assert args == ["-m", "contour.batch", "analyze", "프로젝트 폴더/p.dpprj", "cut1", "--prescan", "0"]
    # a packed app has no python to call: its own program with --worker, which never loads the window
    app = tmp_path / "DIAP.app" / "Contents" / "MacOS" / "DIAP"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app))
    assert BR.worker_command(["prescan", "p", "c1", "c2"]) == (str(app), ["--worker", "batch", "prescan", "p", "c1", "c2"])


def test_the_worker_entry_runs_the_job_and_loads_nothing_of_the_window():
    # (what the child found is written out at once: on Windows a process that loaded the GPU runtime can end with
    # an error code after its work is done, and what was still waiting to be written would be lost with it)
    code = ("import sys, runpy\n"
            "sys.argv = ['app.py', '--worker', 'batch', '--help']\n"
            "try:\n"
            "    runpy.run_path(%r, run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    sys.stdout.flush()\n"
            "    print('RC', e.code, 'QT', sorted(m for m in sys.modules if m.startswith('PySide6'))[:3], flush=True)\n"
            ) % str(ROOT / "app.py")
    kw = dict(capture_output=True, text=True, timeout=180, cwd=str(ROOT), encoding="utf-8", errors="replace",
              env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
    r = subprocess.run([sys.executable, "-c", code], **kw)
    out = (r.stdout or "") + (r.stderr or "")
    assert "RC 0 QT []" in out, f"exit {r.returncode}: {out[-1500:]}"
    assert "analyze" in out and "prescan" in out, "the batch tool's own help"
    code2 = code.replace("'batch', '--help'", "'nothing'")
    r = subprocess.run([sys.executable, "-c", code2], **kw)
    out = (r.stdout or "") + (r.stderr or "")
    assert "RC 2 QT []" in out, f"exit {r.returncode}: {out[-1500:]}"


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)             # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                               # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                                                       # a zombie is not alive
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1].split()[0] != "Z"
    except OSError:
        return True


def test_pause_and_cancel_reach_the_childs_own_children(tmp_path):
    """The analysis runs ffmpeg under it. Pause stops the whole tree where it is, resume lets it go on, and a
    cancel leaves none of it behind (1.22.2 ended the child alone)."""
    from contour.ui import batchrun as BR
    beat, pidf = tmp_path / "beat.txt", tmp_path / "pid.txt"
    grand = ("import time, sys\n"
             "n = 0\n"
             "while True:\n"
             "    n += 1\n"
             "    open(sys.argv[1], 'w').write(str(n))\n"
             "    time.sleep(0.02)\n")
    child = ("import subprocess, sys, time\n"
             "p = subprocess.Popen([sys.executable, '-c', %r, sys.argv[1]])\n"
             "open(sys.argv[2], 'w').write(str(p.pid))\n"
             "time.sleep(600)\n") % grand
    p = subprocess.Popen([sys.executable, "-c", child, str(beat), str(pidf)])
    try:
        t0 = time.time()
        while not (pidf.exists() and pidf.read_text().strip() and beat.exists() and beat.read_text().strip()) \
                and time.time() - t0 < 30:
            time.sleep(0.02)
        gpid = int(pidf.read_text())
        assert set(BR._tree(p.pid)) >= {p.pid, gpid}

        def count():
            for _ in range(20):
                try:
                    return int(beat.read_text() or 0)
                except (ValueError, OSError):
                    time.sleep(0.005)
            return -1
        assert BR._suspend(p.pid, True)
        time.sleep(0.15)
        a = count()
        time.sleep(0.4)
        assert count() == a, "paused: the child's child stopped too"
        assert BR._suspend(p.pid, False)
        time.sleep(0.4)
        assert count() > a, "resumed: it goes on"
        kids = [q for q in BR._tree(p.pid) if q != p.pid]
        p.kill()
        p.wait(10)
        BR._kill_tree(kids)
        t0 = time.time()
        while _alive(gpid) and time.time() - t0 < 5:
            time.sleep(0.05)
        assert not _alive(gpid), "cancelled: nothing of it is left running"
    finally:
        try:
            p.kill()
        except Exception:
            pass
        try:
            BR._kill_tree([int(pidf.read_text())])
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------- media somewhere else
def _two_cut_project(tmp_path, where: Path):
    from conftest import make_video
    from contour.core import media as M
    from contour.core.project import Person, Project
    where.mkdir(parents=True, exist_ok=True)
    a = make_video(where / "A001 클로즈업 (1).mp4", n=6)
    b = make_video(where / "B 002.mp4", n=8)
    prj = Project.create(str(tmp_path / "프로젝트 폴더"), "한글 프로젝트")
    c1 = prj.add_cut(M.probe(str(a)), "c1")
    c2 = prj.add_cut(M.probe(str(b)), "c2")
    c1.persons = [Person(id=1, name="A", color="#ff8800", role="target")]
    c1.params_for(1).amounts["waist"].base = 12.0
    c1.analyzed = True
    c1.joint_keys = {1: {5: {2: [10.0, 11.0]}}}
    prj.save()
    return prj, a, b


def test_media_on_another_path_is_connected_again_and_the_work_stays(tmp_path):
    """Spaces and Korean in the names, a path written by the other system, an 'external volume' with another name."""
    from contour.core import relink as RL
    from contour.core.project import Project
    prj, a, b = _two_cut_project(tmp_path, tmp_path / "촬영 원본" / "day 1")
    # as if saved on a Windows computer: its own drive and backslashes
    d = json.loads((Path(prj.path) / "project.json").read_text(encoding="utf-8"))
    for c, f in zip(d["cuts"], (a, b)):
        win = "Y:\\2611 뮤직비디오\\촬영 원본\\day 1\\" + f.name
        c["source"] = win
        c["info"]["path"] = win
    d["cuts"][0]["lut_path"] = "Y:\\2611 뮤직비디오\\LUT\\쇼 룩 v2.cube"
    (Path(prj.path) / "project.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    vol = tmp_path / "Volumes" / "외장 SSD 2" / "뮤직비디오 (복사본)"
    (vol / "footage" / "deep").mkdir(parents=True)
    os.rename(a, vol / "footage" / "deep" / a.name.upper().replace(".MP4", ".mp4"))     # the name's case changed too
    os.rename(b, vol / "footage" / b.name)
    (vol / "lut").mkdir()
    (vol / "lut" / "쇼 룩 v2.cube").write_text("LUT_3D_SIZE 2\n")
    p = Project.load(prj.path)                                                         # opens on this system as it is
    assert RL.base_name(p.cuts[0].source) == a.name and RL.base_name("/Volumes/x/y z.mov") == "y z.mov"
    miss = RL.missing(p)
    assert sorted((m["kind"], m["name"]) for m in miss) == [("lut", "쇼 룩 v2.cube"), ("source", a.name), ("source", b.name)]
    res = RL.relink(p, vol)
    assert sorted((x["kind"], x["name"]) for x in res["linked"]) == sorted((m["kind"], m["name"]) for m in miss)
    assert not res["different"] and not res["not_found"] and not RL.missing(p)
    c1 = p.cuts[0]
    assert Path(c1.source).exists() and c1.source == c1.info.path and Path(c1.lut_path).exists()
    assert c1.analyzed and c1.params_for(1).amounts["waist"].base == 12.0 and c1.joint_keys, "the work is the cut's"
    assert (c1.info.frames, c1.info.width, c1.info.height) == (6, 64, 48)
    p.save()
    again = Project.load(prj.path)
    assert not RL.missing(again) and "연결할 것이 없습니다" in RL.summary(RL.relink(again, vol))
    from contour.core import media as M
    assert M.probe(again.cuts[1].source).frames == 8, "and it reads from there"


def test_a_file_of_the_same_name_that_is_other_footage_is_not_taken(tmp_path):
    from conftest import make_video
    from contour.core import relink as RL
    from contour.core.project import Project
    prj, a, b = _two_cut_project(tmp_path, tmp_path / "src")
    other = tmp_path / "elsewhere"
    other.mkdir()
    make_video(other / a.name, n=9)                       # same name, another length
    make_video(other / b.name, n=8, w=96, h=48)           # same name, another size
    a.unlink()
    b.unlink()
    p = Project.load(prj.path)
    res = RL.relink(p, other)
    assert not res["linked"] and len(res["different"]) == 2
    why = {d["name"]: d["why"] for d in res["different"]}
    assert "프레임 수가 다름" in why[a.name] and "크기가 다름" in why[b.name]
    assert p.cuts[0].source == str(a), "left as it was"
    assert "다른 영상 2개" in RL.summary(res)
    assert RL.relink(p, tmp_path / "프로젝트 폴더")["not_found"], "and not found where it is not"


def test_the_menu_connects_and_opens_the_cut_from_the_new_place(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from contour.ui import main_window as MW
    prj, a, b = _two_cut_project(tmp_path, tmp_path / "src")
    moved = tmp_path / "새 위치"
    moved.mkdir()
    os.rename(a, moved / a.name)
    os.rename(b, moved / b.name)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *x, **k: QMessageBox.No))
    w = MW.MainWindow()
    toasts = []
    monkeypatch.setattr(w, "toast", lambda text, kind="": toasts.append(text))
    monkeypatch.setattr(w, "show_error", lambda title, text: toasts.append("ERROR " + text))
    try:
        w.open_project(prj.path, wait=True)
        assert any("원본을 찾을 수 없는 컷 2개" in t and "원본 다시 연결" in t for t in toasts)
        assert w.session.proxy is None and "원본 다시 연결" in w.viewer.message
        w.relink_media(str(moved))
        t0 = time.time()
        while (w.job is not None or w.session.proxy is None) and time.time() - t0 < 30:
            qapp.processEvents()
            time.sleep(0.01)
        assert w.session.proxy is not None and w.session.proxy.frame(0) is not None, "the open cut shows again"
        assert any("원본 2컷 연결" in t for t in toasts) and not any(t.startswith("ERROR") for t in toasts)
        from contour.core.project import Project
        assert all(Path(c.source).parent == moved for c in Project.load(prj.path).cuts), "saved"
    finally:
        w.proxy_builder.cancel()
        w.thumbcache.stop()
        w.session.dirty = False
        w._discard_on_close = True
        w.close()


# ------------------------------------------------------------------------------------------------- update and export
def test_each_system_is_offered_its_own_update_file():
    from contour.core import updater as U
    assets = {n: dict(name=n) for n in ("DIAP_Setup_1.23.0.exe", "DIAP_Setup_1.23.0.exe.sha256",
                                        "DIAP_1.23.0_macos_arm64.dmg", "DIAP_1.23.0_macos_arm64.dmg.sha256")}
    assert U.pick_asset(assets, "win32")["name"] == "DIAP_Setup_1.23.0.exe"
    assert U.pick_asset(assets, "darwin", "arm64")["name"] == "DIAP_1.23.0_macos_arm64.dmg"
    assert U.pick_asset(assets, "darwin", "x86_64") is None, "an Intel Mac is not handed the Apple Silicon build"
    only_win = {n: a for n, a in assets.items() if ".exe" in n}
    assert U.pick_asset(only_win, "darwin", "arm64") is None, "a Mac is never handed the Windows installer"
    assert U.pick_asset(assets, "linux") is None


def test_the_macs_hardware_encoder_is_used_only_when_asked_for_and_working(monkeypatch):
    from contour.core import media as MD
    monkeypatch.setattr(MD, "_NVENC", {"ok": False})
    monkeypatch.setattr(MD, "_VTB", {"ok": True})
    assert MD.hevc_encoder("auto") == "libx265", "auto: the encoder both systems share"
    assert MD.hevc_encoder("videotoolbox") == "hevc_videotoolbox"
    args = MD.hevc_args("hevc_videotoolbox", "high")
    assert "main10" in args and "p010le" in args and "hvc1" in args, "10 bit, as the other encoders"
    monkeypatch.setattr(MD, "_VTB", {"ok": False})
    with pytest.raises(MD.MediaError):
        MD.hevc_encoder("videotoolbox")
    monkeypatch.setattr(MD, "_VTB", {})
    if sys.platform != "darwin":
        assert MD.videotoolbox_ok() is False
    monkeypatch.setattr(MD, "_NVENC", {"ok": True})
    assert MD.hevc_encoder("auto") == "hevc_nvenc" and MD.hevc_encoder("x265") == "libx265"
