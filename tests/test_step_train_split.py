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
