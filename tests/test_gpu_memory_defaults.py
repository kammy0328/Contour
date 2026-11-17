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
