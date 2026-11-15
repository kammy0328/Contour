"""Replace ai.onnx Attention (opset 23, no CUDA kernel in onnxruntime 1.24) by MatMul/Softmax/MatMul that CUDA runs.
Only the plain form the SAM 3.1 encoder uses: no mask, not causal, no softcap, default scale 1/sqrt(head_dim)."""
import sys, onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

def patch(src, dst):
    m = onnx.load(src, load_external_data=False)
    head_dim = {}
    for v in list(m.graph.value_info) + list(m.graph.input):
        d = [x.dim_value for x in v.type.tensor_type.shape.dim]
        if len(d) == 4: head_dim[v.name] = d[-1]
    out, k = [], 0
    for n in m.graph.node:
        if n.op_type != "Attention":
            out.append(n); continue
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        assert len(n.input) == 3 and not at.get("is_causal") and not at.get("softcap") and not at.get("qk_matmul_output_mode") \
            and "scale" not in at and "q_num_heads" not in at, (n.name, at, list(n.input))
        q, kk, v = n.input
        hd = head_dim.get(q) or head_dim.get(kk)
        assert hd, ("no head dim", n.name)
        p = f"attn_dec{k}_"; k += 1
        s = np.array(1.0 / np.sqrt(hd), np.float32)
        # the dtype of the tensors: take from value_info of q
        dt = TensorProto.FLOAT
        for vi in m.graph.value_info:
            if vi.name == q: dt = vi.type.tensor_type.elem_type
        sc = numpy_helper.from_array(s.astype(np.float16 if dt == TensorProto.FLOAT16 else np.float32), p + "scale")
        m.graph.initializer.append(sc)
        out += [helper.make_node("Mul", [q, p + "scale"], [p + "qs"], name=p + "mul"),
                helper.make_node("Transpose", [kk], [p + "kt"], perm=[0, 1, 3, 2], name=p + "tr"),
                helper.make_node("MatMul", [p + "qs", p + "kt"], [p + "sc"], name=p + "qk"),
                helper.make_node("Softmax", [p + "sc"], [p + "pr"], axis=-1, name=p + "sm"),
                helper.make_node("MatMul", [p + "pr", v], list(n.output), name=p + "pv")]
    del m.graph.node[:]
    m.graph.node.extend(out)
    onnx.save(m, dst)
    return k

if __name__ == "__main__":
    print(patch(sys.argv[1], sys.argv[2]), "attention nodes replaced")
