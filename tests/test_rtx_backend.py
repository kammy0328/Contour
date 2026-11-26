"""1.27: NVIDIA's TensorRT for RTX provider beside CUDA (track/backend.py rtx / rtx_for, models._rtx_session): only the model
files checked on it, only where the plugin has a device and the settings allow; a session that does not open or a run that
fails goes on with CUDA and says so; the sessions SAM 3.1's users share (models.session_kept)."""
import time
from pathlib import Path

import pytest

from contour.core.settings import settings
from contour.track import backend as BK
from contour.track import models as M


class SO:
    devs = None

    def add_provider_for_devices(self, devs, opts):
        self.devs, self.opts = list(devs), dict(opts)

    def add_session_config_entry(self, k, v):
        self.__dict__.setdefault("cfg", {})[k] = v


class Sess:
    fail_open = False
    fail_run = None             # an exception a TensorRT RTX run raises
    writes = False              # the provider writes a compiled copy when asked (ep.context_*)
    bad_copy = False            # a compiled copy does not open
    made = []

    def __init__(self, path, sess_options=None, providers=None):
        self.rtx = bool(getattr(sess_options, "devs", None))
        if self.rtx and Sess.fail_open:
            raise RuntimeError("engine build failed")
        cfg = getattr(sess_options, "cfg", {})
        if self.rtx and Path(path).name.endswith("_ctx.onnx") and Sess.bad_copy:
            raise RuntimeError("engine file is not for this build")
        if self.rtx and Sess.writes and cfg.get("ep.context_enable") == "1":
            out = Path(cfg["ep.context_file_path"])
            out.write_bytes(b"ctx")
            (out.parent / "engine.bin").write_bytes(b"e" * 64)
        Sess.made.append(("rtx" if self.rtx else "cuda", Path(path).name))

    def get_providers(self):
        return [BK.RTX_EP, "CPUExecutionProvider"] if self.rtx else ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, *a, **k):
        if self.rtx and Sess.fail_run is not None:
            raise Sess.fail_run
        return ["rtx" if self.rtx else "cuda"]

    def end_profiling(self):
        return ""


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import onnxruntime as ort
    Sess.fail_open, Sess.fail_run, Sess.made, Sess.writes, Sess.bad_copy = False, None, [], False, False
    monkeypatch.setattr(M, "RTX_COMPILED_MIN_FREE", 0)
    monkeypatch.setattr(ort, "InferenceSession", Sess)
    monkeypatch.setattr(M, "_options", lambda dml: SO())
    monkeypatch.setattr(M, "providers", lambda: [("CUDAExecutionProvider", {}), "CPUExecutionProvider"])
    monkeypatch.setattr(BK, "rtx", lambda refresh=False: dict(devices=["gpu0"], why="", version="0.5.0", name=BK.RTX_EP))
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    monkeypatch.setenv("DIAP_VERIFY_BACKEND", "0")
    BK.forget()
    assert settings().rtx == "off"                      # the default: for the user to switch on (설정 > TensorRT RTX 가속)
    settings().rtx = "auto"
    enc = tmp_path / "image-encoder-gpu.onnx"
    other = tmp_path / "finder-gpu.onnx"
    for f in (enc, other):
        f.write_bytes(b"x" * 2048)
    return enc, other


def _rec(name):
    return next(r for r in BK.records() if r["file"] == name)


def test_only_the_checked_files_open_on_tensorrt_rtx(rig):
    enc, other = rig
    s = M._session_raw(enc)
    assert s._rtx and s.run(None, {}) == ["rtx"]
    assert s._s.rtx
    r = _rec(enc.name)
    assert r["asked"][0] == "rtx" and BK.effective(r) == "rtx" and BK.LABEL[BK.effective(r)] == "TensorRT RTX"
    o = M._session_raw(other)
    assert not o._rtx and o.run(None, {}) == ["cuda"]
    assert Sess.made == [("rtx", enc.name), ("rtx", enc.name), ("cuda", other.name)]     # (no compiled copy written by this stand-in: opened as it is)


def test_the_compiled_copy_is_written_once_whole_and_opened_from_then_on(rig):
    enc, _ = rig
    Sess.writes = True
    s = M._session_raw(enc)
    assert s._rtx and s.run(None, {}) == ["rtx"]
    d = M._rtx_compiled_dir(enc)
    assert (d / "image-encoder-gpu_ctx.onnx").is_file() and (d / "engine.bin").is_file()
    assert not list(M.rtx_cache_dir().glob("*.tmp*"))
    assert Sess.made == [("rtx", enc.name), ("rtx", "image-encoder-gpu_ctx.onnx")]       # compiled by a session of its own, then the copy opened
    assert _rec(enc.name)["opened"] == "compiled copy"
    Sess.made.clear()
    assert M._session_raw(enc)._rtx and Sess.made == [("rtx", "image-encoder-gpu_ctx.onnx")]   # the next time: the copy alone
    # the model file replaced: its old copy goes, a new one is made
    enc.write_bytes(b"y" * 4096)
    Sess.made.clear()
    assert M._session_raw(enc)._rtx and M._rtx_compiled_dir(enc) != d and not d.exists() and M._rtx_compiled_dir(enc).is_dir()


def test_a_copy_that_does_not_open_is_removed_and_the_model_opened_as_it_is(rig):
    enc, _ = rig
    Sess.writes = True
    M._session_raw(enc)
    d = M._rtx_compiled_dir(enc)
    Sess.bad_copy = True
    Sess.made.clear()
    s = M._session_raw(enc)
    assert s._rtx and s.run(None, {}) == ["rtx"] and not d.exists()
    assert Sess.made == [("rtx", enc.name)] and _rec(enc.name)["opened"] == "compiled now"


def test_a_provider_that_writes_no_copy_or_a_full_drive_is_still_tensorrt_rtx(rig, monkeypatch):
    enc, _ = rig
    s = M._session_raw(enc)                                   # Sess.writes False: nothing written
    assert s._rtx and not M._rtx_compiled_dir(enc).exists() and not list(M.rtx_cache_dir().glob("*.tmp*"))
    Sess.made.clear()
    assert M._session_raw(enc)._rtx and Sess.made == [("rtx", enc.name)]      # and not asked for again: opened as it is, once
    enc.write_bytes(b"z" * 3000)                              # (another file: its own try)
    Sess.writes = True
    monkeypatch.setattr(M, "RTX_COMPILED_MIN_FREE", 1 << 60)
    Sess.made.clear()
    assert M._session_raw(enc)._rtx and Sess.made == [("rtx", enc.name)] and not M._rtx_compiled_dir(enc).exists()


def test_settings_switch_it_off_and_other_devices_do_not_ask(rig, monkeypatch):
    enc, _ = rig
    settings().rtx = "off"
    s = M._session_raw(enc)
    assert not s._rtx and any("TensorRT RTX" in x for x in _rec(enc.name).get("reasons", []))
    settings().rtx = "auto"
    assert BK.rtx_for(enc, "dml", "auto") == ([], "") and BK.rtx_for(enc, "cpu", "auto") == ([], "")
    assert BK.rtx_for(enc, "cuda", "auto")[0] == ["gpu0"] and BK.rtx_for(enc, "auto", "auto")[0] == ["gpu0"]
    monkeypatch.setenv("DIAP_RTX_FILES", "*")
    assert BK.rtx_for("anything.onnx", "auto", "auto")[0] == ["gpu0"]


def test_no_plugin_or_no_device_is_cuda_with_the_reason(rig, monkeypatch):
    enc, _ = rig
    monkeypatch.setattr(BK, "rtx", lambda refresh=False: dict(devices=[], why="이 GPU는 TensorRT RTX 대상이 아님", version="", name=BK.RTX_EP))
    s = M._session_raw(enc)
    assert not s._rtx and s.run(None, {}) == ["cuda"]
    assert any("대상이 아님" in x for x in _rec(enc.name)["reasons"])


def test_a_session_that_does_not_open_goes_on_with_cuda_and_is_not_tried_again(rig):
    enc, _ = rig
    Sess.fail_open = True
    s = M._session_raw(enc)
    assert not s._rtx and s.run(None, {}) == ["cuda"]
    assert any("열지 못해 CUDA" in x for x in _rec(enc.name)["reasons"])
    Sess.fail_open = False
    Sess.made.clear()
    assert not M._session_raw(enc)._rtx and Sess.made == [("cuda", enc.name)]


def test_a_run_that_fails_moves_the_model_to_cuda_for_good(rig):
    enc, _ = rig
    s = M._session_raw(enc)
    Sess.fail_run = RuntimeError("tensor shape mismatch in engine")
    assert s.run(None, {}) == ["cuda"]                       # the same call answered, by CUDA
    Sess.fail_run = None
    assert s.run(None, {}) == ["cuda"] and not s._rtx
    assert any("실행 오류로 CUDA로 전환" in x for x in _rec(enc.name)["reasons"])


def test_out_of_memory_is_not_a_reason_to_move(rig):
    enc, _ = rig
    s = M._session_raw(enc)
    Sess.fail_run = RuntimeError("CUDA out of memory")
    with pytest.raises(M.GpuMemoryError):                    # the encoder is never run on the CPU: the message says what to close
        s.run(None, {})
    assert s._rtx


def test_runs_take_turns(rig):
    enc, other = rig
    assert M._session_raw(enc)._rtx_lock is not None
    o = M._session_raw(other)
    assert o._rtx_lock is None and o.serialize()._rtx_lock is not None


def test_rtx_itself_says_why_it_is_not_there(monkeypatch):
    BK._rtx = None
    r = BK.rtx(refresh=True)                                 # this computer: no plugin (and no NVIDIA GPU)
    assert r["devices"] == [] and r["why"]
    monkeypatch.setenv("DIAP_RTX", "0")
    assert "DIAP_RTX=0" in BK.rtx(refresh=True)["why"]
    assert "rtx" in BK.diagnostics()


# ------------------------------------------------------------------------------------------------ sessions kept
class Kept:
    n = 0

    def __init__(self):
        Kept.n += 1
        self.id = Kept.n
        self.last_used = time.monotonic()


def test_one_session_for_everybody_until_it_idles_or_memory_runs_out(tmp_path, monkeypatch):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"a" * 2048)
    monkeypatch.setattr(M, "session", lambda p: Kept())
    a, b = M.session_kept(f), M.session_kept(f)
    assert a is b
    f.write_bytes(b"b" * 4096)                               # the model file replaced: a session of its own
    c = M.session_kept(f)
    assert c is not a
    assert M._sweep_once(time.monotonic() + M.KEEP_S / 2) == 2
    c.last_used = time.monotonic() + M.KEEP_S                # c was run meanwhile
    assert M._sweep_once(time.monotonic() + M.KEEP_S + 1) == 1 and M.session_kept(f) is c
    M.free_gpu_memory()                                      # the GPU out of memory: all let go
    assert M.session_kept(f) is not c
    monkeypatch.setenv("DIAP_KEEP_SESSIONS", "0")
    assert M.session_kept(f) is not M.session_kept(f)


# ------------------------------------------------------------------------------------------------ the installer's step
def _setup():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import setup_env as SE
    return SE


def test_installer_puts_the_plugin_in_only_where_it_is_of_use():
    SE = _setup()
    calls, said = [], []
    run = lambda args: (calls.append([str(a) for a in args]) or 0)
    # no CUDA, or a GPU that is no RTX: nothing asked, nothing downloaded
    assert SE.rtx_step("DML", run, said.append, name="NVIDIA GeForce RTX 4070 Ti", devices=lambda: -1) == "none"
    assert SE.rtx_step("CUDA", run, said.append, name="NVIDIA GeForce GTX 1660", devices=lambda: -1) == "none"
    assert SE.rtx_step("CUDA", run, said.append, name="", devices=lambda: -1) == "none" and calls == []
    # an RTX GPU without the plugin: installed (no dependencies touched), and it takes the GPU
    state = iter([-1, 1])
    assert SE.rtx_step("CUDA", run, said.append, name="NVIDIA GeForce RTX 4070 Ti", devices=lambda: next(state)) == "on"
    assert calls == [[str(SE.PY), "-m", "pip", "install", "--no-warn-script-location", "--no-deps", SE.RTX_EP]]
    # already there: nothing done
    calls.clear()
    assert SE.rtx_step("CUDA", run, said.append, name="NVIDIA RTX A4000", devices=lambda: 2) == "on" and calls == []


def test_installer_takes_the_plugin_out_where_the_gpu_is_not_for_it_and_survives_a_failed_download():
    SE = _setup()
    calls, said = [], []
    run = lambda args: (calls.append([str(a) for a in args]) or 0)
    state = iter([-1, 0])                                     # installed, and the plugin has no device here (an RTX 20)
    assert SE.rtx_step("CUDA", run, said.append, name="NVIDIA GeForce RTX 2080", devices=lambda: next(state)) == "none"
    assert calls[-1][:5] == [str(SE.PY), "-m", "pip", "uninstall", "-y"] and calls[-1][-1] == "onnxruntime-ep-nv-tensorrt-rtx-cu12"
    assert SE.rtx_step("CUDA", lambda args: 1, said.append, name="NVIDIA GeForce RTX 4090", devices=lambda: -1) == "off"
    assert any("CUDA로 실행" in x for x in said)
    assert SE.RTX_EP.endswith("==" + BK.RTX_VERSION) and SE.RTX_EP.startswith(BK.RTX_PACKAGE)
