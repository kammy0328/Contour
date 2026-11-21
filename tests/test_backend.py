"""Where a model runs (instruction 2026-11-20 §5): decided in one place per system and per model, told by what a run
really did and not by the name of a provider in the session's list, with the reason when a model that asked for an
accelerator ends on the CPU; with no GPU memory left a batch is halved and a small model goes on on the CPU, a model
that would take minutes there says so instead."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from contour.track import backend as BK
from contour.track import models as M

ROOT = Path(__file__).resolve().parents[1]
SPOT = ROOT / "contour" / "track" / "data" / "spotnet_v1.onnx"
CUDA, DML, COREML, CPU = (BK.PROVIDER[k] for k in ("cuda", "dml", "coreml", "cpu"))


@pytest.fixture(autouse=True)
def clean():
    BK.forget()
    M.take_cpu_fallbacks()
    M._verified.clear()
    M.reset_providers()
    yield
    BK.forget()
    M.take_cpu_fallbacks()
    M._verified.clear()
    M.reset_providers()


def _profile(path: Path, nodes) -> str:
    """A run's record as onnxruntime writes it: [(node name, provider, microseconds)]."""
    ev = [dict(cat="Session", name="model_loading_uri", dur=10, args={})]
    for name, prov, us in nodes:
        ev.append(dict(cat="Node", name=name + "_fence_before", dur=0, args=dict(provider=prov)))
        ev.append(dict(cat="Node", name=name + "_kernel_time", dur=us, args=dict(provider=prov, op_name="Conv")))
    path.write_text(json.dumps(ev), encoding="utf-8")
    return str(path)


class FakeOrt:
    """Stands in for onnxruntime.InferenceSession: says it has the providers it is told to say, runs, and writes
    the record of its run with the providers that 'really' ran the operators."""

    def __init__(self, tmp, session_providers, ran, oom_over=None, outputs=None):
        self.tmp, self.session_providers, self.ran, self.oom_over = tmp, session_providers, ran, oom_over
        self.made = []                     # (providers asked, profiling on?)
        self.runs = []                     # batch sizes that ran

    def __call__(self, path, sess_options=None, providers=None):
        outer = self
        prov = [p if isinstance(p, str) else p[0] for p in (providers or [])]
        self.made.append((prov, bool(getattr(sess_options, "enable_profiling", False))))
        cpu_only = prov == [CPU]

        class S:
            def get_providers(self_):
                return [CPU] if cpu_only else list(outer.session_providers)

            def get_inputs(self_):
                return [NS(name="x", shape=["n", 3, 4, 4], type="tensor(float)")]

            def run(self_, names, feed, run_options=None):
                x = feed["x"]
                if not cpu_only and outer.oom_over is not None and len(x) > outer.oom_over:
                    raise RuntimeError("CUDA failure 2: out of memory ; GPU=0")
                outer.runs.append(("cpu" if cpu_only else "gpu", len(x)))
                return [x.reshape(len(x), -1).sum(1, keepdims=True), np.float32(7.0)]

            def end_profiling(self_):
                return _profile(outer.tmp / f"prof{len(outer.made)}.json", outer.ran)
        return S()


def _open(monkeypatch, tmp_path, fake, base=(CUDA, CPU), name="pose_x.onnx"):
    import onnxruntime as ort
    f = tmp_path / name
    f.write_bytes(b"\x08" + b"m" * 3000)
    monkeypatch.setattr(M, "providers", lambda: list(base))
    monkeypatch.setattr(ort, "InferenceSession", fake)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    return M._session_raw(f), f


# ------------------------------------------------------------------------------------------------- what ran
def test_the_record_of_a_real_run_tells_which_provider_ran_the_operators(tmp_path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "p")
    s = ort.InferenceSession(str(SPOT), sess_options=so, providers=[CPU])
    i = s.get_inputs()[0]
    s.run(None, {i.name: np.zeros([1, 4, 64, 64], np.float32)})
    place = BK.placement(s.end_profiling())
    assert list(place) == [CPU] and place[CPU]["nodes"] > 20 and place[CPU]["ms"] > 0
    assert BK.ran_on(place) == ("cpu", 1.0)
    assert BK.placement(tmp_path / "nothing.json") == {} and BK.ran_on({}) == ("", 0.0)
    # an accelerator beside a few CPU operators is the accelerator; the CPU's share of the time is told
    mixed = BK.placement(_profile(tmp_path / "m.json", [("a", CUDA, 9000), ("b", CUDA, 500), ("shape", CPU, 500)]))
    assert mixed == {CUDA: {"nodes": 2, "ms": 9.5}, CPU: {"nodes": 1, "ms": 0.5}}
    assert BK.ran_on(mixed) == ("cuda", 0.05)
    # Core ML takes a part of the graph as one operator: the time says what it weighs
    assert BK.ran_on(BK.placement(_profile(tmp_path / "c.json", [("CoreML_0", COREML, 8000), ("x", CPU, 100), ("y", CPU, 100)])))[0] == "coreml"


def test_a_provider_in_the_sessions_list_is_not_taken_for_acceleration(monkeypatch, tmp_path):
    """The session says CUDA; every operator ran on the CPU. 1.22.2 looked at the list and showed CUDA."""
    fake = FakeOrt(tmp_path, [CUDA, CPU], ran=[("a", CPU, 900), ("b", CPU, 100)])
    sess, f = _open(monkeypatch, tmp_path, fake)
    assert fake.made == [([CUDA, CPU], True)], "the first run of a model on an accelerator is recorded"
    r = BK.records()[0]
    assert r["asked"] == ["cuda", "cpu"] and r["session"] == [CUDA, CPU] and not r["ran"]
    assert M.device_label() == "CUDA", "before any run: what the session offers"
    sess.run(None, {"x": np.ones((2, 3, 4, 4), np.float32)})
    r = BK.records()[0]
    assert r["ran"] and r["backend"] == "cpu" and r["placement"] == {CPU: {"nodes": 2, "ms": 1.0}}
    assert any("모두 CPU" in w for w in r["reasons"]) and r["first_ms"] >= 0
    assert M.device_label() == "CPU" and M.take_cpu_fallbacks() == [("pose_x.onnx", "GPU 실행 장치를 쓰지 못함")]
    assert not list(tmp_path.glob("prof*.json")), "the record of the run is read and removed"
    # the same model opened again in this process is not recorded a second time
    M._session_raw(f)
    assert fake.made[-1] == ([CUDA, CPU], False)
    # and one that really ran on the accelerator, beside it: the line says both
    fake2 = FakeOrt(tmp_path, [CUDA, CPU], ran=[("a", CUDA, 900), ("s", CPU, 50)])
    s2, _ = _open(monkeypatch, tmp_path, fake2, name="det_y.onnx")
    s2.run(None, {"x": np.ones((1, 3, 4, 4), np.float32)})
    r2 = [r for r in BK.records() if r["file"] == "det_y.onnx"][0]
    assert r2["backend"] == "cuda" and not r2["reasons"] and r2["cpu_share"] < 0.1
    assert M.device_label() == "CUDA (일부 CPU)"
    t = BK.text()
    assert "det_y.onnx" in t and "실제 CUDA" in t and "실제 CPU" in t and "모두 CPU" in t and "연산 배치" in t


def test_heavy_operators_left_on_the_cpu_are_said(monkeypatch, tmp_path):
    # as the SAM 3.1 encoder's Attention was before its graph was rewritten: on the GPU by name, on the CPU by time
    fake = FakeOrt(tmp_path, [CUDA, CPU], ran=[("conv", CUDA, 400), ("attention", CPU, 9600)])
    sess, _ = _open(monkeypatch, tmp_path, fake)
    sess.run(None, {"x": np.ones((1, 3, 4, 4), np.float32)})
    r = BK.records()[0]
    assert r["backend"] == "cuda" and r["cpu_share"] == 0.96 and any("96%" in w and "CPU" in w for w in r["reasons"])


def test_a_session_that_could_only_be_opened_on_the_cpu_says_why(monkeypatch, tmp_path):
    import onnxruntime as ort

    def no_cuda(path, sess_options=None, providers=None):
        names = [p if isinstance(p, str) else p[0] for p in providers]
        if CUDA in names:
            raise RuntimeError("LoadLibrary failed with error 126 when trying to load onnxruntime_providers_cuda.dll "
                               "(cudnn64_9.dll missing)")
        return NS(get_providers=lambda: [CPU], run=lambda names, feed, run_options=None: [feed["x"]])
    f = tmp_path / "pose_x.onnx"
    f.write_bytes(b"\x08" + b"m" * 3000)
    monkeypatch.setattr(M, "providers", lambda: [CUDA, CPU])
    monkeypatch.setattr(ort, "InferenceSession", no_cuda)
    monkeypatch.setattr(BK, "gpu", lambda refresh=False: dict(vendor="nvidia", name="RTX 3060", memory_mb=12288, driver="640.10"))
    s = M._session_raw(f)
    r = BK.records()[0]
    assert r["session"] == [CPU] and any("CUDA 라이브러리" in w and "cudnn64_9.dll" in w for w in r["reasons"])
    s.run(None, {"x": np.ones((1, 3, 4, 4), np.float32)})
    assert BK.records()[0]["backend"] == "cpu" and M.device_label() == "CPU"
    assert ("pose_x.onnx", "GPU에서 열지 못함") in M.take_cpu_fallbacks()
    # with a driver older than CUDA 12 runs with, that is the reason told
    BK.forget()
    monkeypatch.setattr(BK, "system", lambda: "win32")
    monkeypatch.setattr(BK, "gpu", lambda refresh=False: dict(vendor="nvidia", name="GTX 1660", memory_mb=6144, driver="512.15"))
    M._session_raw(f)
    assert any("512.15" in w and "528.33" in w and "업데이트" in w for w in BK.records()[0]["reasons"])


# ------------------------------------------------------------------------------------------------- per system and model
def test_each_system_has_its_order_and_each_model_its_own_list(monkeypatch, tmp_path):
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA, DML, COREML, CPU])
    for plat, want in (("win32", ["cuda", "dml", "cpu"]), ("darwin", ["coreml", "cpu"]), ("linux", ["cuda", "cpu"])):
        monkeypatch.setattr(sys, "platform", plat)
        assert BK.offered() == want
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CPU])
    assert BK.offered() == ["cpu"]
    # Core ML: with 자동 only a model checked on a Mac; asked for by hand, every model it can take; never SAM
    base = [(COREML, dict(BK.COREML_OPTIONS)), CPU]
    pose = tmp_path / "onnx" / "pose_rtmw_x.onnx"
    prov, why = BK.for_model(pose, base, "auto")
    assert prov == [CPU] and "확인 전" in why[0]
    prov, why = BK.for_model(pose, base, "coreml")
    assert prov[0][0] == COREML and prov[0][1]["ModelFormat"] == "MLProgram" and prov[-1] == CPU and not why
    monkeypatch.setitem(BK.COREML_OK, "pose_rtmw_x", {"MLComputeUnits": "CPUAndGPU"})
    prov, why = BK.for_model(pose, base, "auto")
    assert prov[0][1]["MLComputeUnits"] == "CPUAndGPU" and not why
    prov, why = BK.for_model(tmp_path / "onnx" / "seg_sam31" / "image-encoder-gpu.onnx", base, "coreml")
    assert prov == [CPU] and "지원하지 않는" in why[0]
    # CUDA as before for every model, the CPU always last
    assert BK.for_model(pose, [(CUDA, {"a": 1}), CPU], "auto") == ([(CUDA, {"a": 1}), CPU], [])
    assert BK.for_model(pose, [CPU], "cpu") == ([CPU], [])


def test_a_package_says_which_backends_its_model_is_for(monkeypatch, tmp_path):
    from test_modelpkg import KEYS, make_registry
    from contour.track import modelpkg as MP
    MP._store = None
    reg = make_registry(tmp_path / "reg", [("matte_rvm", "2026.11.1", {"rvm.onnx": b"R" * 4000},
                                           dict(main="rvm.onnx", backends=["cuda", "cpu"]))])
    MP.store().install(reg, "matte_rvm", trusted=KEYS, app_version="1.23.0", ort_version="1.24.4")
    p = M.model_path("matte_rvm")
    assert BK.package_backends(p) == ("cuda", "cpu") and BK.model_name(p) == "matte_rvm"
    prov, why = BK.for_model(p, [DML, CPU], "auto")
    assert prov == [CPU] and "DirectML" in why[0] and "대상이 아님" in why[0]
    assert BK.for_model(p, [(CUDA, {}), CPU], "auto")[0][0][0] == CUDA
    assert BK.package_backends(tmp_path / "elsewhere.onnx") == ()
    MP._store = None


def test_the_driver_is_held_against_the_fixed_combination(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    g = lambda d: dict(vendor="nvidia", name="x", driver=d)
    assert BK.driver_state(g("621.14")) == ("ok", "")
    st, txt = BK.driver_state(g("527.56"))
    assert st == "old" and "528.33" in txt and "527.56" in txt
    st, txt = BK.driver_state(g("551.86"))
    assert st == "unchecked" and "572.30" in txt
    assert BK.driver_state({}) == ("unknown", "") and BK.driver_state(dict(vendor="apple", driver="15.1")) == ("unknown", "")
    assert (BK.ORT_PINNED, BK.CUDA_RUNTIME, BK.CUDNN_RUNTIME) == ("1.24.4", "12.8", "9")
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert f"onnxruntime-gpu[cuda,cudnn]=={BK.ORT_PINNED}" in req, "the fixed combination is the one that is installed"


def test_two_runtime_packages_in_one_environment_are_told(monkeypatch):
    from importlib import metadata
    mk = lambda n, v: NS(metadata={"Name": n}, version=v)
    monkeypatch.setattr(metadata, "distributions", lambda: [mk("onnxruntime-gpu", "1.24.4"), mk("onnxruntime", "1.24.4"),
                                                            mk("onnxruntime-extensions", "0.1"), mk("numpy", "2.5.3")])
    assert BK.runtime_packages() == [("onnxruntime", "1.24.4"), ("onnxruntime-gpu", "1.24.4")]
    d = BK.diagnostics()
    assert any("여러 개" in w and "onnxruntime-gpu" in w for w in d["warnings"])
    assert "주의: ONNX Runtime 패키지가 여러 개" in BK.text(d)
    monkeypatch.setattr(metadata, "distributions", lambda: [mk("onnxruntime-gpu", "1.24.4")])
    assert not [w for w in BK.diagnostics()["warnings"] if "여러 개" in w]


# ------------------------------------------------------------------------------------------------- no GPU memory
def test_a_batch_is_halved_when_the_gpu_has_no_memory_for_it(monkeypatch, tmp_path):
    fake = FakeOrt(tmp_path, [CUDA, CPU], ran=[("a", CUDA, 100)], oom_over=2)
    sess, _ = _open(monkeypatch, tmp_path, fake)
    sess.batch_input = "x"
    x = np.arange(7 * 3 * 4 * 4, dtype=np.float32).reshape(7, 3, 4, 4)
    out = sess.run(None, {"x": x})
    assert np.array_equal(out[0], x.reshape(7, -1).sum(1, keepdims=True)), "every item, in its place"
    assert out[1] == 7.0, "what is not per item is passed as it is"
    assert [n for d, n in fake.runs] == [1, 2, 2, 2] and all(d == "gpu" for d, n in fake.runs)
    assert any("배치 7개를 나눠" in w for w in BK.records()[0]["reasons"]) and not M.take_cpu_fallbacks()
    # 1.22.2 gave up here: GpuMemoryError for the whole analysis


def test_a_small_model_goes_on_on_the_cpu_and_a_slow_one_says_what_to_do(monkeypatch, tmp_path):
    fake = FakeOrt(tmp_path, [CUDA, CPU], ran=[("a", CUDA, 100)], oom_over=0)
    sess, _ = _open(monkeypatch, tmp_path, fake)
    out = sess.run(None, {"x": np.ones((1, 3, 4, 4), np.float32)})
    assert out[0].shape == (1, 1) and fake.runs == [("cpu", 1)]
    r = BK.records()[0]
    assert r["backend"] == "cpu" and any("GPU 메모리 부족" in w and "CPU로 전환" in w for w in r["reasons"])
    assert M.take_cpu_fallbacks() == [("pose_x.onnx", "GPU 메모리 부족")]
    sess.run(None, {"x": np.ones((1, 3, 4, 4), np.float32)})
    assert fake.runs == [("cpu", 1), ("cpu", 1)], "it stays there for the rest of the work"
    # the SAM encoder would take minutes a picture on the CPU: the message that says what to close, as before
    fake = FakeOrt(tmp_path, [CUDA, CPU], ran=[("a", CUDA, 100)], oom_over=0)
    enc, _ = _open(monkeypatch, tmp_path, fake, name="image-encoder-gpu.onnx")
    with pytest.raises(M.GpuMemoryError) as e:
        enc.run(None, {"x": np.ones((1, 3, 4, 4), np.float32)})
    assert "GPU 메모리가 부족합니다" in str(e.value) and not fake.runs
    # switched off by hand (DIAP_OOM_CPU=0): the error for every model, as 1.22.2
    monkeypatch.setenv("DIAP_OOM_CPU", "0")
    fake = FakeOrt(tmp_path, [CUDA, CPU], ran=[("a", CUDA, 100)], oom_over=0)
    s2, _ = _open(monkeypatch, tmp_path, fake, name="det_y.onnx")
    with pytest.raises(M.GpuMemoryError):
        s2.run(None, {"x": np.ones((1, 3, 4, 4), np.float32)})


# ------------------------------------------------------------------------------------------------- the real thing
def test_a_real_model_on_this_computer_is_recorded_by_what_it_ran_on():
    s = M.session(SPOT)
    i = s.get_inputs()[0]
    s.run(None, {i.name: np.zeros([1, 4, 64, 64], np.float32)})
    r = [r for r in BK.records() if r["file"] == SPOT.name][0]
    assert r["ran"] and r["backend"] in ("cpu", "cuda", "dml", "coreml") and r["first_ms"] > 0
    assert r["asked"][-1] == "cpu" and BK.effective(r) == r["backend"]
    if r["asked"] == ["cpu"]:
        assert r["backend"] == "cpu" and not r["reasons"], "the CPU when the CPU was asked for is no fallback"
    d = BK.diagnostics()
    assert d["onnxruntime"] and d["offered"][-1] == "cpu" and d["models"] and d["pinned"]["cuda"] == "12.8"
    t = BK.text(d)
    assert "ONNX Runtime" in t and SPOT.name in t and "모델별 실제 실행" in t
    assert str(ROOT) not in t and str(Path.home()) not in t, "no path of this computer in the diagnostics"


def test_a_model_the_runtime_opened_on_the_cpu_by_itself_runs(monkeypatch):
    """j2805 (the 4070 Ti with CUDA_VISIBLE_DEVICES=-1: CUDA does not start): onnxruntime opened every model on the
    CPU by itself, and every run of the small models ended with 'Did not find an arena based allocator registered
    for device-id combination in the memory arena shrink list: gpu:0'. The run's option that gives the CUDA pool's
    spare memory back went by what was asked for, not by what the session has (so in 1.22 too; a DirectML session
    has no such pool either). Here: the runtime of this computer, asked for CUDA."""
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("this computer has CUDA: the session is opened on it")
    monkeypatch.setattr(M, "providers", lambda: [(CUDA, {}), CPU])        # as a PC with the GPU runtime asks
    s = M._session_raw(SPOT)
    assert list(s.get_providers()) == [CPU], "opened on the CPU alone"
    i = s.get_inputs()[0]
    out = s.run(None, {i.name: np.zeros([1, 4, 64, 64], np.float32)})
    assert out and np.isfinite(out[0]).all()
    assert s._ro is None, "no CUDA pool to give memory back to"
    r = [r for r in BK.records() if r["file"] == SPOT.name][0]
    assert r["ran"] and r["backend"] == "cpu" and BK.effective(r) == "cpu" and r["asked"][0] == "cuda"
    assert any("CPU" in w for w in r["reasons"]), r["reasons"]


def test_a_run_whose_memory_option_the_session_has_no_pool_for_goes_on_without_it(monkeypatch):
    """The same message from a session that has the accelerator in its list (a pool of another kind): the option is
    dropped and the run made again, once; any other error is the caller's as before."""
    calls = []

    class S:
        def run(self, names, feed, run_options=None):
            calls.append(run_options is not None)
            if run_options is not None:
                raise RuntimeError("[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Did not find an arena based allocator "
                                   "registered for device-id  combination in the memory arena shrink list: gpu:0")
            return [feed["x"]]

        def get_providers(self):
            return [CUDA, CPU]
    s = M.Session(S(), "small.onnx", shrink=True)
    assert s._ro is not None
    assert s.run(None, {"x": np.ones(2)})[0].sum() == 2 and calls == [True, False]
    assert s._ro is None and s.run(None, {"x": np.ones(2)})[0].sum() == 2 and calls == [True, False, False]

    class Bad(S):
        def run(self, names, feed, run_options=None):
            raise RuntimeError("INVALID_ARGUMENT : Got invalid dimensions for input")
    with pytest.raises(RuntimeError, match="invalid dimensions"):
        M.Session(Bad(), "small.onnx", shrink=True).run(None, {"x": np.ones(2)})


def test_the_pose_crops_of_a_frame_are_a_batch_that_may_be_halved():
    src = (ROOT / "contour" / "track" / "infer.py").read_text(encoding="utf-8")
    assert "self.sess.batch_input = self.inp" in src
