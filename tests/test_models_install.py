"""5. The SAM 3.1 person finder is rewritten on the user's PC with the onnx package: a plain install must have it.
6. An empty, cut off or broken model file must not count as downloaded, a cut off download must not be kept, and a
failed model download must not end the setup as a success."""
import io
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
