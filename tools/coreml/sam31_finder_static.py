"""finder-gpu.onnx -> finder-coreml.onnx: the inputs that are always the same (text, text_pad: the phrase "person"; box_tokens: none)
put into the graph as constants and folded away (onnxsim), so only fpn0/1/2 remain and every shape is fixed."""
import onnx, numpy as np, os, sys, collections
from onnx import helper, numpy_helper
SRC = os.path.expanduser("~/mnt/_DIAP_FIX/iwaly_batch/data/models/onnx/seg_sam31_find/finder-gpu.onnx")
NPZ = os.path.expanduser("~/mnt/_DIAP_FIX/iwaly_batch/code_code_j2818a/contour/track/data/sam31_text_person.npz")
m = onnx.load(SRC); g = m.graph
txt = np.load(NPZ)
const = {"text": txt["text"].astype(np.float32), "text_pad": txt["text_pad"]}
print({k: (v.shape, v.dtype) for k, v in const.items()})
for k, v in const.items():
    g.initializer.append(numpy_helper.from_array(v, k))
# box_tokens: 0 tokens
nodes = []
for n in g.node:
    if n.op_type == "Shape" and n.input[0] == "box_tokens":
        g.initializer.append(numpy_helper.from_array(np.array([0], np.int64), n.output[0])); continue
    if n.op_type == "Concat" and "box_tokens" in n.input:
        rest = [x for x in n.input if x != "box_tokens"]
        nodes.append(helper.make_node("Identity", rest, list(n.output), name=n.name + "_nobox") if len(rest) == 1 else
                     helper.make_node("Concat", rest, list(n.output), name=n.name, axis=0)); continue
    nodes.append(n)
del g.node[:]; g.node.extend(nodes)
keep = [i for i in g.input if i.name not in ("text", "text_pad", "box_tokens")]
del g.input[:]; g.input.extend(keep)
print("inputs", [i.name for i in g.input])
import onnxsim
ms, ok = onnxsim.simplify(m)
print("onnxsim ok", ok)
c = collections.Counter(n.op_type for n in ms.graph.node)
print(len(ms.graph.node), c.most_common())
dyn = [v.name for v in ms.graph.value_info if any(d.dim_param or d.dim_value == 0 for d in v.type.tensor_type.shape.dim)]
print("dynamic value_info", len(dyn), dyn[:5])
onnx.save_model(ms, "out_f/finder-coreml.onnx", save_as_external_data=False)
print("saved", os.path.getsize("out_f/finder-coreml.onnx"))
