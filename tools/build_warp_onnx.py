"""Builds assets/onnx/warp_plane.onnx: the full-resolution warp of one image plane on the GPU.

inputs   D (1,2,gh,gw) float32   coarse backward field in output px (dx, dy), cell-centred grid
         P (1,1,h,w)   float32   the plane (luma or a subsampled chroma plane)
         size (4,)     int64     [1, 2, H, W] full output size
         steps (2,)    int64     [ssy, ssx] plane subsampling
output   O (1,1,h,w)   float32

Resize (cubic, half pixel = OpenCV INTER_CUBIC on the coarse grid) -> co-sited slice for chroma ->
source coordinates -> GridSample (bicubic, border)."""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper as h

nodes = []
inits = []


def const(name, arr, dtype=None):
    arr = np.asarray(arr, dtype=dtype)
    inits.append(onnx.numpy_helper.from_array(arr, name))
    return name


const("i0", [0], np.int64)
const("i1", [1], np.int64)
const("i2", [2], np.int64)
const("i3", [3], np.int64)
const("big", [1 << 30, 1 << 30], np.int64)
const("ax23", [2, 3], np.int64)
const("st0", [0, 0], np.int64)
const("f0", 0.0, np.float32)
const("f1", 1.0, np.float32)
const("f2", 2.0, np.float32)
const("ax_last", [3], np.int64)

nodes += [
    h.make_node("Resize", ["D", "", "", "size"], ["Dfull"], mode="cubic", coordinate_transformation_mode="half_pixel",
                cubic_coeff_a=-0.75, exclude_outside=0, nearest_mode="floor"),
    h.make_node("Slice", ["Dfull", "st0", "big", "ax23", "steps"], ["Ds"]),          # (1,2,h,w) co-sited
    h.make_node("Shape", ["P"], ["pshape"]),
    h.make_node("Gather", ["pshape", "i2"], ["ph_"], axis=0),
    h.make_node("Gather", ["pshape", "i3"], ["pw_"], axis=0),
    h.make_node("Squeeze", ["ph_"], ["ph"]),
    h.make_node("Squeeze", ["pw_"], ["pw"]),
    h.make_node("Cast", ["ph"], ["phf"], to=TensorProto.FLOAT),
    h.make_node("Cast", ["pw"], ["pwf"], to=TensorProto.FLOAT),
    h.make_node("Range", ["f0", "phf", "f1"], ["ys"]),                                 # (h,)
    h.make_node("Range", ["f0", "pwf", "f1"], ["xs"]),                                 # (w,)
    h.make_node("Cast", ["steps"], ["stepsf"], to=TensorProto.FLOAT),
    h.make_node("Gather", ["stepsf", "i0"], ["ssy"], axis=0),
    h.make_node("Gather", ["stepsf", "i1"], ["ssx"], axis=0),
    h.make_node("Split", ["Ds"], ["Dx", "Dy"], axis=1, num_outputs=2),               # (1,1,h,w) each
    h.make_node("Div", ["Dx", "ssx"], ["dxp"]),
    h.make_node("Div", ["Dy", "ssy"], ["dyp"]),
    h.make_node("Unsqueeze", ["ys", "i1"], ["ys2"]),                                  # (h,1)
    h.make_node("Add", ["dxp", "xs"], ["xsrc"]),                                      # broadcast over w
    h.make_node("Add", ["dyp", "ys2"], ["ysrc"]),
    h.make_node("Sub", ["pwf", "f1"], ["wm1"]),
    h.make_node("Sub", ["phf", "f1"], ["hm1"]),
    h.make_node("Max", ["wm1", "f1"], ["wm1c"]),
    h.make_node("Max", ["hm1", "f1"], ["hm1c"]),
    h.make_node("Div", ["xsrc", "wm1c"], ["xn0"]),
    h.make_node("Div", ["ysrc", "hm1c"], ["yn0"]),
    h.make_node("Mul", ["xn0", "f2"], ["xn1"]),
    h.make_node("Mul", ["yn0", "f2"], ["yn1"]),
    h.make_node("Sub", ["xn1", "f1"], ["gx"]),                                        # (1,1,h,w)
    h.make_node("Sub", ["yn1", "f1"], ["gy"]),
    h.make_node("Concat", ["gx", "gy"], ["g2"], axis=1),                              # (1,2,h,w)
    h.make_node("Transpose", ["g2"], ["grid"], perm=[0, 2, 3, 1]),                    # (1,h,w,2)
    h.make_node("GridSample", ["P", "grid"], ["O"], mode="bicubic", padding_mode="border", align_corners=1),
]
graph = h.make_graph(
    nodes, "contour_warp_plane",
    [h.make_tensor_value_info("D", TensorProto.FLOAT, [1, 2, "gh", "gw"]),
     h.make_tensor_value_info("P", TensorProto.FLOAT, [1, 1, "h", "w"]),
     h.make_tensor_value_info("size", TensorProto.INT64, [4]),
     h.make_tensor_value_info("steps", TensorProto.INT64, [2])],
    [h.make_tensor_value_info("O", TensorProto.FLOAT, [1, 1, "h", "w"])],
    initializer=inits)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], producer_name="contour")
model.ir_version = 8
onnx.checker.check_model(model)
out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets" / "onnx" / "warp_plane.onnx")
out.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(out))
print("saved", out)
