"""Face parser step training holds out whole cuts (2026-11-18): neighbouring frames in training and check made the
check too kind."""
import importlib.util
import json
import os


def test_held_out_faces_are_whole_cuts(tmp_path):
    spec = importlib.util.spec_from_file_location("step_train", os.path.join(os.path.dirname(__file__), "..", "cloud", "step_train.py"))
    st = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(st)
    except ImportError as e:                       # torch / transformers missing here: test the split on its own
        import pytest
        pytest.skip(str(e))
    for c in range(12):
        d = tmp_path / f"cut{c:02d}"
        d.mkdir()
        json.dump({"items": [{"f": f"{i}.jpg", "lum": 0.5, "vis": 1.0, "wf": 120} for i in range(10)]},
                  open(d / "meta.json", "w"))
    items = st.load_items(str(tmp_path))
    val, tr = st.split_by_cut(items)
    cv = {m["_cut"] for _, _, m in val}
    ct = {m["_cut"] for _, _, m in tr}
    assert cv and ct and not (cv & ct)
    assert len(val) >= 10 and len(tr) + len(val) == 120


def test_exports_do_not_need_onnxscript():
    """j2745: the teacher's export failed after an hour of tuning (torch 2.14 defaults to the exporter that needs
    onnxscript, which the app does not ship)."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "cloud", "step_train.py"), encoding="utf-8").read()
    assert 'kw["dynamo"] = False' in src and "--export-teacher" in src
    ref = open(os.path.join(os.path.dirname(__file__), "..", "contour", "learn", "refiner.py"), encoding="utf-8").read()
    assert "dynamo=False" in ref


def test_install_onto_itself_is_a_no_op(tmp_path):
    src = open(os.path.join(os.path.dirname(__file__), "..", "cloud", "step_train.py"), encoding="utf-8").read()
    ns = {}
    exec(src[src.index("def _install("):src.index("def models_dir(")], {"os": os, "shutil": __import__("shutil")}, ns)
    f = tmp_path / "m.onnx"
    f.write_bytes(b"x")
    ns["_install"](str(f), str(f))                   # was shutil.SameFileError
    ns["_install"](str(f), str(tmp_path / "n.onnx"))
    assert (tmp_path / "n.onnx").read_bytes() == b"x"


def test_models_dir_is_the_apps(tmp_path, monkeypatch):
    """data_path is a property: calling it raised, and every step-trained model stayed in Cloud\\model where the app
    never reads it (j2748)."""
    from contour.core import settings as S
    S._settings = S.AppSettings(data_dir=str(tmp_path / "data"))
    try:
        src = open(os.path.join(os.path.dirname(__file__), "..", "cloud", "step_train.py"), encoding="utf-8").read()
        ns = {"os": os, "MODEL": "CLOUD_MODEL"}
        exec(src[src.index("def models_dir("):src.index("def main(")], ns)
        assert ns["models_dir"]() == os.path.join(str(S.settings().data_path), "models", "onnx")
        from contour.track import cloudcollect as CC
        monkeypatch.delenv("DIAP_CLOUD", raising=False)
        assert CC.cloud_dir() is not None
    finally:
        S._settings = None
