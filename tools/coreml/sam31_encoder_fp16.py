"""image-encoder-gpu.onnx -> image-encoder-coreml.onnx: fp16 throughout (the fp32 RoPE islands and their 199 Casts taken out,
the input and output Casts kept), the constant position-embedding chain (Slice/Reshape/Transpose/Tile/Slice/Slice/Transpose)
folded into one initializer, Gather(axis, scalar) written as Slice+Squeeze. Weights are the same numbers."""
import onnx, numpy as np, os, sys
from onnx import helper, numpy_helper, TensorProto
D = sys.argv[1]; SRC = os.path.join(D, "image-encoder-gpu.onnx"); DST = os.path.join(sys.argv[2], "image-encoder-coreml.onnx")
m = onnx.load(SRC, load_external_data=True); g = m.graph
init = {i.name: i for i in g.initializer}
def val(n): return numpy_helper.to_array(init[n])
out_names = {o.name for o in g.output}; in_names = {i.name for i in g.input}
prod = {o: n for n in g.node for o in n.output}
# 1) fold the position embedding chain ending at node_permute_2
chain = ["node_slice_2", "node_view", "node_permute_1", "node_tile", "node_slice_3", "node_slice_4", "node_permute_2"]
byname = {n.name: n for n in g.node}
env = {}
def get(x): return env[x] if x in env else val(x)
for nm in chain:
    n = byname[nm]; a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
    if n.op_type == "Slice":
        x = get(n.input[0]).astype(np.float32); st = get(n.input[1]); en = get(n.input[2])
        ax = get(n.input[3]) if len(n.input) > 3 and n.input[3] else np.arange(len(st))
        sp = get(n.input[4]) if len(n.input) > 4 and n.input[4] else np.ones(len(st), np.int64)
        sl = [slice(None)] * x.ndim
        for s, e, aa, p in zip(st, en, ax, sp): sl[int(aa)] = slice(int(s), int(min(e, 2**62)), int(p))
        y = x[tuple(sl)]
    elif n.op_type == "Reshape":
        x = get(n.input[0]); sh = [int(v) for v in get(n.input[1])]
        sh = [x.shape[i] if v == 0 else v for i, v in enumerate(sh)]; y = x.reshape(sh)
    elif n.op_type == "Transpose":
        y = get(n.input[0]).transpose(a["perm"])
    elif n.op_type == "Tile":
        y = np.tile(get(n.input[0]), [int(v) for v in get(n.input[1])])
    env[n.output[0]] = y
pe = env[byname["node_permute_2"].output[0]].astype(np.float16)
print("folded pos embed", pe.shape)
g.initializer.append(numpy_helper.from_array(pe, "pos_embed_folded"))
for n in g.node:
    for i, x in enumerate(n.input):
        if x == byname["node_permute_2"].output[0]: n.input[i] = "pos_embed_folded"
keep = [n for n in g.node if n.name not in chain]
# 2) Casts: drop all but the input cast and the casts that make the graph outputs
ren = {}
nodes = []
for n in keep:
    if n.op_type == "Cast" and n.input[0] not in in_names and n.output[0] not in out_names:
        ren[n.output[0]] = n.input[0]; continue
    nodes.append(n)
def r(x):
    while x in ren: x = ren[x]
    return x
for n in nodes:
    for i, x in enumerate(n.input): n.input[i] = r(x)
# fp32 initializers used inside (none expected) -> fp16
for i in list(g.initializer):
    if i.data_type == TensorProto.FLOAT:
        a_ = numpy_helper.to_array(i).astype(np.float16); i.CopyFrom(numpy_helper.from_array(a_, i.name)); print("init to fp16", i.name)
# 3) Gather(axis, scalar index) -> Slice + Squeeze
out = []
for n in nodes:
    if n.op_type == "Gather" and n.input[1] in init and val(n.input[1]).ndim == 0:
        ax = [helper.get_attribute_value(a) for a in n.attribute if a.name == "axis"][0]; k = int(val(n.input[1]))
        base = n.name + "_cm"
        g.initializer.extend([numpy_helper.from_array(np.array([k], np.int64), base + "_s"),
                              numpy_helper.from_array(np.array([k + 1], np.int64), base + "_e"),
                              numpy_helper.from_array(np.array([ax], np.int64), base + "_a")])
        out.append(helper.make_node("Slice", [n.input[0], base + "_s", base + "_e", base + "_a"], [base + "_sl"], name=base + "_slice"))
        out.append(helper.make_node("Squeeze", [base + "_sl", base + "_a"], [n.output[0]], name=base + "_sq"))
    else:
        out.append(n)
del g.node[:]; g.node.extend(out)
# drop initializers nobody uses
used = {x for n in g.node for x in n.input}
kept = [i for i in g.initializer if i.name in used]
del g.initializer[:]; g.initializer.extend(kept)
del g.value_info[:]
import collections
print("nodes", len(g.node), collections.Counter(n.op_type for n in g.node).most_common())
for f in (DST, DST + ".data"):
    if os.path.exists(f): os.rename(f, f + ".old")
onnx.save_model(m, DST, save_as_external_data=True, all_tensors_to_one_file=True, location="image-encoder-coreml.onnx.data", size_threshold=1024)
print("saved", os.path.getsize(DST), os.path.getsize(DST + ".data"))
