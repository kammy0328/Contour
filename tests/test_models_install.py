"""5. The SAM 3.1 person finder is rewritten on the user's PC with the onnx package: a plain install must have it.
6. An empty, cut off or broken model file must not count as downloaded, a cut off download must not be kept, and a
failed model download must not end the setup as a success."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _reqs(name):
    out = {}
    for ln in (ROOT / name).read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if ln and not ln.startswith("-"):
            out[ln.split("==")[0].split("[")[0].split("<")[0].strip().lower()] = ln
    return out


def test_onnx_in_base_requirements():
    import contour.track.onnx_patch as OP           # the finder's rewrite needs it
    assert "onnx" in OP.__dict__
    assert "onnx" in _reqs("requirements.txt"), "onnx must come with every install, not only with the learning extras"


def _tiny_onnx(path: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    m.doc_string = "x" * 2000          # real models are megabytes; this one only needs to pass the size floor
    onnx.save(m, str(path))
    return path


def test_empty_or_garbage_model_is_not_usable(tmp_path):
    from contour.track import models as M
    empty = tmp_path / "a.onnx"
    empty.write_bytes(b"")
    junk = tmp_path / "b.onnx"
    junk.write_bytes(b"<html>404 Not Found</html>")
    good = _tiny_onnx(tmp_path / "c.onnx")
    assert not M._usable(empty)
    assert not M._usable(junk)
    assert M._usable(good)


def test_is_downloaded_false_for_empty_file(monkeypatch, tmp_path):
    from contour.track import models as M
    p = M.model_path("face_yunet")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    assert not M.is_downloaded("face_yunet")


class _Resp(io.BytesIO):
    def __init__(self, data, total):
        super().__init__(data)
        self.headers = {"Content-Length": str(total)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_cut_off_download_is_not_kept(monkeypatch, tmp_path):
    from contour.track import models as M
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(b"x" * 1000, 5000))
    dst = tmp_path / "m.onnx"
    with pytest.raises(Exception):
        M._download("https://example.invalid/m.onnx", dst, None, "m")
    assert not dst.exists()


def test_broken_model_is_set_aside_on_load(tmp_path):
    from contour.track import models as M
    p = tmp_path / "broken.onnx"
    p.write_bytes(b"\x08\x09" + b"\xff" * 4000)       # starts like a model, is not one
    with pytest.raises(Exception) as ei:
        M.session(p)
    assert not p.exists(), "a model file that cannot be read is moved away so it is downloaded again"
    assert "손상" in str(ei.value)


def test_download_models_exit_code(monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import download_models as DM
    from contour.track import models as M
    monkeypatch.setattr(M, "is_downloaded", lambda k: False)

    def fail(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(M, "ensure_model", fail)
    monkeypatch.setattr(sys, "argv", ["download_models.py"])
    rc = DM.main()
    assert rc not in (0, None), "a failed required model must not end as success"


def test_setup_reports_model_failure(monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import setup_env as SE
    assert SE.models_step(lambda args: 1) is False
    assert SE.models_step(lambda args: 0) is True


# ------------------------------------------------------------------ the GPU runtime at install (instruction 2026-11-20 §5)
class _Pip:
    """Stands in for the installer's pip and its GPU check: what is installed, what the checks answer."""

    def __init__(self, check, alt_check="DML", download_ok=True, install_ok=True, installed=("onnxruntime-gpu",)):
        self.installed = set(installed)
        self.check, self.alt_check, self.download_ok, self.install_ok = check, alt_check, download_ok, install_ok
        self.calls = []
        self.said = []

    def run(self, args):
        a = [str(x) for x in args]
        self.calls.append(a)
        if "download" in a:
            return 0 if self.download_ok else 1
        if "uninstall" in a:
            for n in list(self.installed):
                if n in a:
                    self.installed.discard(n)
            return 0
        if "install" in a:
            if "--target" in a:
                return 0
            what = "onnxruntime-directml" if any("directml" in x for x in a) else "onnxruntime-gpu"
            if what == "onnxruntime-directml" and not self.install_ok:
                return 1
            self.installed.add(what)
            return 0
        return 0

    def gpu(self, path=None):
        if path is not None:
            return self.alt_check
        if not self.installed:
            return "NONE"
        if "onnxruntime-directml" in self.installed and "onnxruntime-gpu" not in self.installed:
            return "DML"
        return self.check

    def step(self, tmp_path):
        import setup_env as SE
        return SE.gpu_step(run=self.run, gpu=self.gpu, runtimes=lambda: sorted(self.installed), say=self.said.append,
                           alt_dir=tmp_path / "alt")


def test_the_working_runtime_is_not_removed_for_one_that_was_not_tried(tmp_path):
    """1.22.2 uninstalled onnxruntime-gpu as soon as the GPU check did not say CUDA (also when the check itself
    failed, e.g. without a connection: it downloaded its test model) and then tried to download DirectML: with no
    connection the app was left without any runtime."""
    sys.path.insert(0, str(ROOT / "tools"))
    # CUDA works: nothing is touched
    p = _Pip("CUDA")
    assert p.step(tmp_path) == "CUDA" and p.calls == [] and p.installed == {"onnxruntime-gpu"}
    # no CUDA and no connection: the runtime stays, the CPU runs it
    p = _Pip("CPU", download_ok=False)
    assert p.step(tmp_path) == "CPU" and p.installed == {"onnxruntime-gpu"}
    assert not any("uninstall" in c for c in p.calls) and any("바꾸지 않았습니다" in s for s in p.said)
    # DirectML downloaded but not working on this PC: stays
    p = _Pip("CPU", alt_check="CPU")
    assert p.step(tmp_path) == "CPU" and p.installed == {"onnxruntime-gpu"} and not any("uninstall" in c for c in p.calls)
    # DirectML seen to run a model in its own folder: changed, from the files already here, and only one runtime
    p = _Pip("CPU")
    assert p.step(tmp_path) == "DML" and p.installed == {"onnxruntime-directml"}
    order = [next(w for w in ("download", "uninstall", "install") if w in c) for c in p.calls]
    assert order == ["download", "install", "uninstall", "install"], "tried before the old one is removed"
    assert "--target" in p.calls[1] and "--no-index" in p.calls[3], "the change itself needs no connection"
    # the change fails after the trial worked: the runtime that worked is put back
    p = _Pip("CPU", install_ok=False)
    assert p.step(tmp_path) == "CPU" and p.installed == {"onnxruntime-gpu"} and any("되돌립니다" in s for s in p.said)
    # two runtimes left by an interrupted change: cleaned up to the pinned one
    p = _Pip("CUDA", installed=("onnxruntime-gpu", "onnxruntime-directml"))
    assert p.step(tmp_path) == "CUDA" and p.installed == {"onnxruntime-gpu"}
    assert not (tmp_path / "alt").exists()


def test_the_gpu_check_runs_a_model_that_comes_with_the_app_and_needs_no_connection(monkeypatch, capsys):
    import importlib
    import urllib.request
    sys.path.insert(0, str(ROOT / "tools"))
    import check_gpu as CG
    importlib.reload(CG)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a download")))
    assert CG.MODEL.is_file(), "the test model is part of the app"
    assert CG.main(["--json"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] in ("CUDA", "DML", "COREML", "CPU")
    info = json.loads(out[-2])
    assert info["answer"] == out[-1] and info["version"] and "CPUExecutionProvider" in info["available"]
    # the answer is an accelerator only when operators ran on it: a provider that is listed and ran nothing is not
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(CG, "run_on", lambda ort_, prov: dict(listed=True, nodes=0, cpu=111))
    assert CG.main([]) == 0 and capsys.readouterr().out.strip().splitlines()[-1] == "CPU"
    monkeypatch.setattr(CG, "run_on", lambda ort_, prov: dict(listed=True, nodes=98, cpu=13))
    assert CG.main([]) == 0 and capsys.readouterr().out.strip().splitlines()[-1] == "CUDA"
