"""DIAP's combined model (face classes, clothing classes, person alpha in one ONNX file): off by default; when on,
face_parse, parse_clothes and matte_rvm are neither downloaded nor loaded; a file that does not load, or a run that
fails, sends that feature back to its own model; cuts whose masks were made by another model are found."""
import json
import os

import numpy as np
import pytest

from contour.track import models as M
from contour.track import unified as U


def _tiny_unified(path, face_classes=19):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "h", "w"])
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in ("face", "cloth", "alpha")]
    wf = numpy_helper.from_array(rng.standard_normal((face_classes, 3, 1, 1)).astype(np.float32), "wf")
    wc = numpy_helper.from_array(rng.standard_normal((18, 3, 1, 1)).astype(np.float32), "wc")
    nodes = [helper.make_node("AveragePool", ["image"], ["p4"], kernel_shape=[4, 4], strides=[4, 4]),
             helper.make_node("Conv", ["p4", "wf"], ["face"]),
             helper.make_node("Conv", ["p4", "wc"], ["cloth"]),
             helper.make_node("ReduceMean", ["image"], ["m"], axes=[1], keepdims=1),
             helper.make_node("Sigmoid", ["m"], ["alpha"])]
    g = helper.make_graph(nodes, "u", [x], outs, initializer=[wf, wc])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


@pytest.fixture
def on(monkeypatch):
    from contour.core.settings import settings
    U.reset_session()
    monkeypatch.setenv("DIAP_UNIFIED", "1")
    p = U.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    _tiny_unified(p)
    called = []

    def no_download(key, *a, **k):
        called.append(key)
        raise AssertionError(f"{key} would be downloaded")
    monkeypatch.setattr(M, "ensure_model", no_download)
    yield called
    U.reset_session()


def _lm():
    import cv2
    lm = np.zeros((68, 3), np.float64)
    xs = np.linspace(100, 220, 17)
    lm[:17, 0], lm[:17, 1] = xs, 200 + 40 * np.sin(np.linspace(0, np.pi, 17))
    lm[36:42, :2] = [130, 150]
    lm[42:48, :2] = [190, 150]
    lm[:, 2] = 1
    return lm


def test_off_by_default(monkeypatch):
    U.reset_session()
    monkeypatch.delenv("DIAP_UNIFIED", raising=False)
    assert not U.switched_on() and not U.active("face")
    called = []
    monkeypatch.setattr(M, "ensure_model", lambda key, *a, **k: (called.append(key), (_ for _ in ()).throw(RuntimeError("stop")))[1])
    from contour.track import faceparse as FP
    with pytest.raises(RuntimeError):
        FP.FaceParser()
    assert called == ["face_parse"]


def test_on_replaces_all_three_without_downloading(on):
    from contour.track import faceparse as FP
    from contour.track.clothing import Parser
    from contour.track.infer import Matter
    fp = FP.FaceParser()
    rgb = np.random.default_rng(1).integers(0, 255, (360, 320, 3)).astype(np.uint8)
    r = fp.parse(rgb, _lm())
    assert r is not None and r["skin"].shape == (FP.KEEP, FP.KEEP)
    cm = Parser()(rgb[40:300, 60:260])
    assert cm.shape == (260, 200) and cm.max() < 18
    mt = Matter()
    mt.reset()
    a = mt(rgb)
    assert a.shape == (360, 320) and 0 <= a.min() and a.max() <= 1
    assert on == []                                     # nothing downloaded or loaded in their place
    assert FP.ready()


def test_broken_file_goes_back_to_own_models(on, monkeypatch):
    U.path().write_bytes(b"x" * 4096)
    from contour.track import faceparse as FP
    with pytest.raises(AssertionError):
        FP.FaceParser()                                 # back to face_parse (here: its download is refused)
    assert on == ["face_parse"] and "load" in U._failed


def test_failed_run_affects_that_feature_only(on, monkeypatch):
    from contour.track.clothing import Parser
    from contour.track.infer import Matter
    rt = U.runtime()
    monkeypatch.setattr(rt, "alpha", lambda rgb: (_ for _ in ()).throw(RuntimeError("cuda error")))
    mt = Matter()
    with pytest.raises(AssertionError):
        mt(np.zeros((64, 64, 3), np.uint8))             # RVM again (its download refused here)
    assert on == ["matte_rvm"] and not U.active("matte") and U.active("cloth")
    assert Parser().unified is not None


def test_masks_made_by_another_model_are_found(on, monkeypatch):
    fp = {U.KEY: "u1", "face_parse": "f1", "parse_clothes": "c1", "matte_rvm": "m1"}
    monkeypatch.setattr(M, "fingerprint", lambda k: fp.get(k, ""))
    assert sorted(U.stale({"face_parse": "f1", "matte_rvm": "m1"})) == ["cloth", "face", "matte"]
    assert U.stale({U.KEY: "u1", "face_parse": "f1"}) == []
    assert sorted(U.stale({U.KEY: "u0"})) == ["cloth", "face", "matte"]       # another version of it
    monkeypatch.setenv("DIAP_UNIFIED", "0")
    assert sorted(U.stale({U.KEY: "u1"})) == ["cloth", "face", "matte"]       # made by it, now off
    assert U.stale({"face_parse": "f1", "matte_rvm": "m1"}) == []
    assert U.stale({"face_parse": "f0"}) == ["face"]


def test_analysis_records_the_combined_model(on, monkeypatch):
    monkeypatch.setattr(M, "fingerprint", lambda k: "x" if k == U.KEY else "")
    assert M.fingerprints() == {U.KEY: "x"}


def _fake_teach(root, project, cut, n=2):
    import cv2
    from contour.learn.unified import teach as T
    rng = np.random.default_rng(len(project))
    d = os.path.join(root, project, cut)
    os.makedirs(d, exist_ok=True)
    for t in range(n):
        img = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
        base = dict(project=project, cut=cut, cut_id=cut, t=t, pair_of=t - 1 if t else None)
        face = rng.random((6, 16, 16))
        np.savez_compressed(os.path.join(d, f"face_{t:05d}_p1.npz"), img=T._jpg(img), face=T._q(face / face.sum(0)),
                            meta=np.array(json.dumps(dict(base, kind="face", pid=1))))
        cl = rng.random((18, 16, 16))
        np.savez_compressed(os.path.join(d, f"person_{t:05d}_p1.npz"), img=T._jpg(img), cloth=T._q(cl / cl.sum(0)),
                            alpha=(rng.random((64, 64)) * 255).astype(np.uint8),
                            meta=np.array(json.dumps(dict(base, kind="person", pid=1))))
        np.savez_compressed(os.path.join(d, f"frame_{t:05d}.npz"), img=T._jpg(img[:32]),
                            alpha=(rng.random((32, 64)) * 255).astype(np.uint8),
                            meta=np.array(json.dumps(dict(base, kind="frame"))))


def test_train_split_export_and_compare(tmp_path, monkeypatch):
    """Two projects: one held out (fixed), partial labels per kind, a checked ONNX with three outputs that the app
    side reads, and the comparison runs on the held-out project."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from contour.learn.unified import bench as B
    from contour.learn.unified import net as N
    from contour.learn.unified import train as TR
    small = N.build
    monkeypatch.setattr(N, "build", lambda pretrained=True, width=128, face_classes=6:
                        small(pretrained=False, width=16, face_classes=face_classes))
    root, out = str(tmp_path / "teach"), str(tmp_path / "out")
    _fake_teach(root, "MV_A", "c1")
    _fake_teach(root, "MV_B", "c2")
    rec = TR.train(root, out, steps=3, batch=2, log=lambda *a: None)
    assert rec["onnx_check"] and os.path.exists(rec["onnx"])
    held = json.load(open(os.path.join(out, "val_fixed.json")))["projects"]
    assert held == rec["held"] and len(held) == 1
    again, _, h2 = TR.split(__import__("contour.learn.unified.teach", fromlist=["index"]).index(root), out)
    assert h2 == held
    st = B.Student(rec["onnx"])
    items = [it for it in __import__("contour.learn.unified.teach", fromlist=["index"]).index(root) if it[2] in held]
    res = B.compare(items, st, teachers=None, n=4)
    assert res["face"]["student_clean"]["n"] > 0 and res["alpha"]["student_clean"]["mad"] is not None
    assert res["flicker"]["face_pairs"] >= 1
    # the app side reads the same file
    rt = U.Runtime(__import__("pathlib").Path(rec["onnx"]))
    assert rt.face_logits(np.zeros((64, 64, 3), np.uint8)).shape == (6, 16, 16) and rt.face_classes == 6
    assert rt.alpha(np.zeros((90, 160, 3), np.uint8)).shape == (90, 160)


def test_commercial_face_model_uses_the_combined_model_taught_by_it(on, monkeypatch):
    """A combined model taught by MediaPipe multiclass (6 face classes) serves face_model 'mp'; one taught by the
    non-commercial parser (19) is never used for it."""
    from contour.track import faceparse as FP
    U.reset_session()
    _tiny_unified(U.path(), face_classes=6)
    mp = FP.MPFaceParser()
    assert mp.unified is not None
    pr = mp.probs(np.zeros((512, 512, 3), np.uint8))
    assert pr.shape == (6, FP.MP_SIZE, FP.MP_SIZE) and np.allclose(pr.sum(0), 1, atol=1e-4)
    U.reset_session()
    _tiny_unified(U.path(), face_classes=19)
    assert FP.MPFaceParser().unified is None


def test_teacher_samples_keep_each_teacher_and_the_silhouette_apart(tmp_path):
    """Ballin instruction: the SegFormer teachers (when their files are here) and the tracking's coarse silhouette
    are stored next to the commercial teachers' answers, each under its own name, with timecode and review state."""
    from types import SimpleNamespace as NS
    from contour.learn.unified import teach as T
    from contour.track import faceparse as FP
    from contour.track.store import TrackStore
    n = 6
    st = TrackStore(n, [3])
    lm = _lm()
    for t in range(n):
        st.present[0, t] = True
        st.boxes[0, t] = [60, 40, 260, 340]
        st.face[0, t] = lm
    st.has_face = True
    rgb = np.random.default_rng(2).integers(0, 255, (360, 320, 3)).astype(np.uint8)
    proxy = NS(frame=lambda t: rgb, sx=1.0, sy=1.0, w=320, h=360)
    lab = np.full((90, 80), -1, np.int16)
    lab[10:85, 15:65] = 0
    tracker = NS(store=st, proxy=proxy, matte=NS(w=80, h=90), labels=lambda t: lab)
    cut = NS(work_range=(0, n), name="c", id="cid", source="A_0006C009.mov")
    prj = NS(name="Ballin")

    class FaceSF:
        def parse(self, img, lm_):
            M_ = FP.FaceParser.crop_affine(lm_)
            z = np.zeros((FP.KEEP, FP.KEEP), np.float16)
            return dict(M=M_.astype(np.float32), skin=z + 1, neck=z, hair=z, cloth=z, feat=z)
    teachers = NS(matter=NS(reset=lambda: None, __call__=None), face_probs=lambda c: np.full((6, 128, 128), 0.5, np.float32),
                  cloth_probs=lambda c: np.zeros((18, 128, 128), np.float32), face_sf=FaceSF(),
                  cloth_sf=lambda crop: np.full(crop.shape[:2], 4, np.uint8))

    class Matter:
        def reset(self):
            pass

        def __call__(self, img):
            return np.full(img.shape[:2], 0.7, np.float32)
    teachers.matter = Matter()
    n_ = T.run_cut(prj, cut, tracker, teachers, tmp_path, frames=[2, 3], tc=lambda t: f"00:00:00:{t:02d}")
    assert n_ == dict(face=2, person=2, frame=2)
    with np.load(tmp_path / "Ballin" / "cid" / "face_00003_p3.npz") as d:
        assert d["face_sf"].shape == (5, 128, 128) and d["face_sf"][0].min() == 255      # SegFormer skin, apart
        assert d["face"].shape == (6, 128, 128)
        m = json.loads(str(d["meta"]))
        assert m["tc"] == "00:00:00:03" and m["review"] == "unreviewed" and m["pair_of"] == 2
        assert "segformer_b5_face" in m["teachers"] and m["labels"] == ["face_mediapipe", "face_sf"]
    with np.load(tmp_path / "Ballin" / "cid" / "person_00002_p3.npz") as d:
        assert (d["cloth_sf"] == 4).all() and d["silhouette"].shape == (512, 512)
        assert d["alpha"].min() == d["alpha"].max()           # RVM's soft alpha untouched by the silhouette
        assert set(np.unique(d["silhouette"]).tolist()) <= {0, 255}
    with np.load(tmp_path / "Ballin" / "cid" / "frame_00002.npz") as d:
        assert "silhouette" in d.files and "alpha" in d.files
