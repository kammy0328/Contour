"""SAM speed and the computer stuttering during analysis (2026-11-18, j2730): the GPU memory pools grow by what is
asked and runs give spare memory back. Measured on the PC: 1249 s -> 58 s for 12 frames of cut 024, GPU memory median
11.9 -> 2.0 GB, the matte identical."""
import os


def test_pools_grow_by_request_and_runs_give_memory_back(monkeypatch):
    from contour.track import models as M
    assert M.ARENA == "same" and M.RUN_SHRINK == "1"
    monkeypatch.delenv("DIAP_ARENA", raising=False)
    monkeypatch.delenv("DIAP_RUN_SHRINK", raising=False)

    class FakeSess:
        def __init__(self):
            self.kw = None

        def run(self, *a, **k):
            self.kw = k
            return []
    fs = FakeSess()
    s = M.Session(fs, "pose_rtmw_x.onnx", shrink=True)
    s.run(None, {"x": 0})
    ro = fs.kw.get("run_options")
    assert ro is not None and ro.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:0"
    fs2 = FakeSess()
    M.Session(fs2, "image-encoder-gpu.onnx", shrink=False).run(None, {"x": 0})
    assert "run_options" not in (fs2.kw or {})


def test_a_model_the_runtime_opened_on_the_cpu_by_itself_runs(monkeypatch):
    """j2805 (the 4070 Ti with CUDA_VISIBLE_DEVICES=-1: CUDA does not start): onnxruntime opened every model on the
    CPU by itself, and every run of the small models ended with 'Did not find an arena based allocator registered
    for device-id combination in the memory arena shrink list: gpu:0'. The run's option that gives the CUDA pool's
    spare memory back went by what was asked for, not by what the session has (a DirectML session has no such pool
    either). Here: the runtime of this computer, asked for CUDA."""
    from pathlib import Path
    import numpy as np
    import pytest
    import onnxruntime as ort
    from contour.track import models as M
    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("this computer has CUDA: the session is opened on it")
    spot = Path(M.__file__).resolve().parent / "data" / "spotnet_v1.onnx"
    monkeypatch.setattr(M, "providers", lambda: [("CUDAExecutionProvider", {}), "CPUExecutionProvider"])
    label = M._device_label
    try:
        s = M._session_raw(spot)
        assert list(s.get_providers()) == ["CPUExecutionProvider"], "opened on the CPU alone"
        i = s.get_inputs()[0]
        out = s.run(None, {i.name: np.zeros([1, 4, 64, 64], np.float32)})
        assert out and np.isfinite(out[0]).all()
        assert s._ro is None, "no CUDA pool to give memory back to"
    finally:
        M._device_label = label
        M.take_cpu_fallbacks()


def test_a_run_whose_memory_option_the_session_has_no_pool_for_goes_on_without_it():
    """The same message from a session that has the accelerator in its list (a pool of another kind): the option is
    dropped and the run made again, once; any other error is the caller's as before."""
    import numpy as np
    import pytest
    from contour.track import models as M
    calls = []

    class S:
        def run(self, names, feed, run_options=None):
            calls.append(run_options is not None)
            if run_options is not None:
                raise RuntimeError("[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Did not find an arena based allocator "
                                   "registered for device-id  combination in the memory arena shrink list: gpu:0")
            return [feed["x"]]
    s = M.Session(S(), "small.onnx", shrink=True)
    assert s._ro is not None
    assert s.run(None, {"x": np.ones(2)})[0].sum() == 2 and calls == [True, False]
    assert s._ro is None and s.run(None, {"x": np.ones(2)})[0].sum() == 2 and calls == [True, False, False]

    class Bad(S):
        def run(self, names, feed, run_options=None):
            raise RuntimeError("INVALID_ARGUMENT : Got invalid dimensions for input")
    with pytest.raises(RuntimeError, match="invalid dimensions"):
        M.Session(Bad(), "small.onnx", shrink=True).run(None, {"x": np.ones(2)})
