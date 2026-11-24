"""image-encoder-coreml.onnx (v1) -> image-encoder-coreml2.onnx: the windowed attention's rank-6 Reshape/Transpose pairs
written in rank 5 (the batch of 1 dropped: the same numbers in the same order), the input taken as fp16 (no Cast)."""
import onnx, numpy as np, os, collections
from onnx import numpy_helper, helper, TensorProto
SRC = "out/image-encoder-coreml.onnx"; DST = "out2/image-encoder-coreml2.onnx"
m = onnx.load(SRC, load_external_data=True); g = m.graph
init = {i.name: i for i in g.initializer}
cons = collections.defaultdict(list)
for n in g.node:
    for x in n.input: cons[x].append(n)
k = 0
for n in g.node:
    if n.op_type == "Reshape" and n.input[1] in init:
        sh = numpy_helper.to_array(init[n.input[1]])
        if len(sh) != 6: continue
        assert sh[0] == 1, sh
        c = cons[n.output[0]]
        assert len(c) == 1 and c[0].op_type == "Transpose", (n.name, [x.op_type for x in c])
        tp = c[0]; perm = [helper.get_attribute_value(a) for a in tp.attribute if a.name == "perm"][0]
        assert perm[0] == 0, perm
        nm = n.name + "_r5"
        g.initializer.append(numpy_helper.from_array(sh[1:].astype(np.int64), nm))
        n.input[1] = nm
        for a in tp.attribute:
            if a.name == "perm":
                del a.ints[:]; a.ints.extend([p - 1 for p in perm[1:]])
        k += 1
print("rewritten", k)
# input as fp16
cast = [n for n in g.node if n.op_type == "Cast" and n.input[0] == "image"][0]
for n in g.node:
    for i, x in enumerate(n.input):
        if x == cast.output[0]: n.input[i] = "image"
g.node.remove(cast)
g.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
used = {x for n in g.node for x in n.input}
kept = [i for i in g.initializer if i.name in used]; del g.initializer[:]; g.initializer.extend(kept)
del g.value_info[:]
os.makedirs("out2", exist_ok=True)
onnx.save_model(m, DST, save_as_external_data=True, all_tensors_to_one_file=True, location="image-encoder-coreml2.onnx.data", size_threshold=1024)
onnx.checker.check_model(DST)
print("saved", os.path.getsize(DST), os.path.getsize(DST + ".data"))
