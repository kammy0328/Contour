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
    from contour.learn.safe_install import install
    f = tmp_path / "m.onnx"
    f.write_bytes(b"x")
    assert install(str(f), str(f))["ok"]             # was shutil.SameFileError
    assert install(str(f), str(tmp_path / "n.onnx"))["ok"]
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


def _st():
    spec = importlib.util.spec_from_file_location("step_train", os.path.join(os.path.dirname(__file__), "..", "cloud", "step_train.py"))
    st = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(st)
    except ImportError as e:
        import pytest
        pytest.skip(str(e))
    return st


def _project(root, name, cuts, per=10):
    for c in range(cuts):
        d = root / name / f"cut{c:02d}"
        d.mkdir(parents=True)
        json.dump({"items": [{"f": f"{i}.jpg", "lum": 0.5, "vis": 1.0, "wf": 120} for i in range(per)]},
                  open(d / "meta.json", "w"))


def test_earlier_projects_check_cuts_are_never_trained_on(tmp_path):
    """Review of 1.21.0: val_old was a random sample of the very crops the replay trained on."""
    st = _st()
    done = tmp_path / "_trained"
    _project(done, "A", 8)
    _project(done, "B", 6)
    fixed = {}
    val_old, old = st.old_split(str(done), fixed)
    assert val_old and old
    key = lambda it: (it[2]["_prj"], it[2]["_cut"])
    assert not ({key(i) for i in val_old} & {key(i) for i in old})
    assert {i[2]["_prj"] for i in val_old} == {"A", "B"}
    # the same next time (and with a fresh choice it would also be the same: chosen from the name)
    v2, o2 = st.old_split(str(done), json.loads(json.dumps(fixed)))
    assert [i[0] for i in v2] == [i[0] for i in val_old]
    v3, _ = st.old_split(str(done), {})
    assert [i[0] for i in v3] == [i[0] for i in val_old]


def test_this_projects_check_cuts_stay_fixed(tmp_path):
    st = _st()
    _project(tmp_path / "data", "P", 12)
    items = st.load_items(str(tmp_path / "data" / "P"))
    fixed = {}
    val, tr = st.fixed_split("P", items, fixed)
    held = {m["_cut"] for _, _, m in val}
    assert len(held) >= st.MIN_VAL_CUTS and not (held & {m["_cut"] for _, _, m in tr})
    # more cuts analysed later: the held ones stay held, the new ones train
    _project(tmp_path / "data2", "P", 16)
    items2 = st.load_items(str(tmp_path / "data2" / "P"))
    val2, tr2 = st.fixed_split("P", items2, fixed)
    assert {m["_cut"] for _, _, m in val2} == held


def test_too_few_check_cuts_hold_the_install(tmp_path):
    st = _st()
    _project(tmp_path, "one", 1, per=40)
    items = st.load_items(str(tmp_path / "one"))
    val, tr = st.fixed_split("one", items, {})
    assert val == [] and st.val_problem(val, [], False)
    _project(tmp_path, "many", 12)
    items = st.load_items(str(tmp_path / "many"))
    val, tr = st.fixed_split("many", items, {})
    assert st.val_problem(val, [], False) == ""
    assert st.val_problem(val, val[:3], True)          # earlier projects present but their check too small


def _tiny_onnx(path, nan=False):
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, None)
    nodes = [helper.make_node("Relu", ["pixel_values"], ["r"])]
    if nan:
        nodes += [helper.make_node("Constant", [], ["z"], value=helper.make_tensor("z", TensorProto.FLOAT, [], [0.0])),
                  helper.make_node("Div", ["r", "z"], ["d"]), helper.make_node("Sub", ["d", "d"], ["logits"])]
    else:
        nodes += [helper.make_node("Identity", ["r"], ["logits"])]
    m = helper.make_model(helper.make_graph(nodes, "g", [x], [y]), opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_onnx_checks(tmp_path):
    import numpy as np
    from contour.learn.safe_install import check_onnx
    p = tmp_path / "ok.onnx"
    _tiny_onnx(p)
    x = np.random.default_rng(0).standard_normal((1, 3, 8, 8)).astype(np.float32)
    ref = np.maximum(x, 0)
    assert check_onnx(str(p), {"pixel_values": x}, ref, (1, 3, 8, 8))["ok"]
    assert not check_onnx(str(p), {"pixel_values": x}, ref, (1, 19, 2, 2))["ok"]          # shape
    assert not check_onnx(str(p), {"pixel_values": x}, ref + 1.0, (1, 3, 8, 8))["ok"]     # disagrees
    q = tmp_path / "nan.onnx"
    _tiny_onnx(q, nan=True)
    assert not check_onnx(str(q), {"pixel_values": x}, None, (1, 3, 8, 8))["ok"]         # NaN
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    assert "로드" in check_onnx(str(bad), {"pixel_values": x})["reason"]


def test_failed_install_keeps_the_model_in_use(tmp_path):
    from contour.learn.safe_install import install
    dst = tmp_path / "models" / "face_parse_own.onnx"
    dst.parent.mkdir()
    dst.write_bytes(b"in use")
    src = tmp_path / "new.onnx"
    src.write_bytes(b"broken new model")
    r = install(str(src), str(dst), check=lambda p: dict(ok=False, reason="출력에 NaN/Inf"))
    assert not r["ok"] and "NaN" in r["reason"]
    assert dst.read_bytes() == b"in use"
    assert sorted(x.name for x in dst.parent.iterdir()) == ["face_parse_own.onnx"]     # no temporary left
    seen = {}
    r = install(str(src), str(dst), check=lambda p: seen.setdefault("p", p) and dict(ok=True))
    assert r["ok"] and dst.read_bytes() == b"broken new model"
    assert os.path.dirname(seen["p"]) == str(dst.parent)                                # checked next to it


def test_history_is_kept_on_a_failed_write(tmp_path, monkeypatch):
    from contour.learn import safe_install as SI
    h = tmp_path / "history.json"
    SI.atomic_json(str(h), [{"a": 1}])

    class Boom:
        pass
    import pytest
    with pytest.raises(TypeError):
        SI.atomic_json(str(h), [{"a": 1}, {"b": Boom()}])
    assert json.load(open(h)) == [{"a": 1}]
    assert not [x.name for x in tmp_path.iterdir() if x.name.endswith(".tmp")]


def test_project_without_check_cuts_is_not_trained(tmp_path, monkeypatch):
    st = _st()
    try:
        import transformers  # noqa: F401  (main imports it first)
    except ImportError as e:
        import pytest
        pytest.skip(str(e))
    _project(tmp_path / "data", "single", 1, per=40)
    for k, v in dict(DATA=tmp_path / "data", DONE=tmp_path / "_trained", MODEL=tmp_path / "model", HERE=tmp_path).items():
        monkeypatch.setattr(st, k, str(v))
    monkeypatch.setattr(st.sys, "argv", ["step_train.py", "single", "10", "--keep"])
    assert st.main() == 3
    h = json.load(open(tmp_path / "history.json", encoding="utf-8"))
    assert h[-1]["installed"] is False and "검증" in h[-1]["why"]
    assert (tmp_path / "data" / "single").is_dir()              # the data stays for when more cuts are in
