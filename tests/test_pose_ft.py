"""RTMW-x fine-tune helpers (2026-11-19): crops the way the pose model sees them, fresh degradations, SimCC
targets that decode back to the answer, the score (pck, left/right swaps) and the ONNX Clip fix onnx2torch needs
(PC j2771d: conversion stopped at a Clip with an empty bound)."""
import numpy as np
import pytest

from contour.learn import pose_ft as PF


def test_crop_matches_the_model_input_and_points_follow():
    rng = np.random.default_rng(0)
    rgb = (rng.random((720, 1280, 3)) * 255).astype(np.uint8)
    box = np.array([500, 100, 700, 650], np.float32)
    img, m = PF.crop(rgb, box)
    assert img.shape == (384, 288, 3)
    c = PF.to_crop(np.array([[600, 375]]), m)[0]
    assert abs(c[0] - 144) < 1 and abs(c[1] - 192) < 1


@pytest.mark.parametrize("kind", PF.KINDS)
def test_every_degradation_keeps_the_crop_shape(kind):
    rng = np.random.default_rng(1)
    img = (rng.random((384, 288, 3)) * 255).astype(np.uint8)
    out = PF.degrade(img, kind, rng)
    assert out.shape == img.shape and out.dtype == np.uint8
    if kind != "clean":
        assert not np.array_equal(out, img)


def test_targets_decode_back_to_the_answer():
    kp = np.array([[100.25, 200.5], [10.0, 380.0], [400.0, 10.0], [np.nan, 5.0]], np.float32)
    tx, ty, w = PF.simcc_targets(kp, np.ones(4))
    assert tx.shape == (4, 576) and ty.shape == (4, 768)
    assert list(w) == [1, 1, 0, 0]                        # outside the crop, unknown: no target
    p, s = PF.decode(tx[None], ty[None])
    assert np.allclose(p[0, :2], kp[:2], atol=0.5)


def test_score_counts_swapped_sides():
    gt = np.zeros((1, 17, 2), np.float32)
    gt[0, :, 0] = np.arange(17) * 30
    gt[0, :, 1] = 100
    pred = gt.copy()
    pred[0, [9, 10]] = gt[0, [10, 9]]                     # the wrists swapped
    s = PF.score(pred, gt, np.ones((1, 17), bool), np.array([300.0]))
    assert s["swap"] == pytest.approx(1 / 8) and s["pck05"] < 1


def test_kl_loss_is_lowest_at_the_target():
    torch = pytest.importorskip("torch")
    kp = np.array([[100.0, 200.0]], np.float32)
    tx, ty, w = PF.simcc_targets(kp, np.ones(1))
    T = lambda a: torch.from_numpy(a)[None]
    good = PF.kl_loss(T(tx), T(ty), T(tx), T(ty), T(w))
    bad = PF.kl_loss(T(np.roll(tx, 40, -1)), T(ty), T(tx), T(ty), T(w))
    assert float(good) < float(bad)


def test_clip_with_an_empty_bound_gets_one():
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto
    node = helper.make_node("Clip", ["x", "", "mx"], ["y"])
    g = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
                          [helper.make_tensor("mx", TensorProto.FLOAT, [], [6.0])])
    m = helper.make_model(g)
    assert PF.fix_onnx_clip(m) == 1 and m.graph.node[0].input[1] != ""
