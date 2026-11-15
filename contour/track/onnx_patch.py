"""Replace ai.onnx Attention (opset 23; onnxruntime 1.24 has no CUDA kernel for it, so it runs on the CPU) by
Mul/MatMul/Softmax/MatMul, which CUDA runs. Handles the forms the SAM 3.1 graphs use: 4-D q, k, v, optional attention
mask (boolean: True = may attend; float: added to the scores), not causal, no softcap, default scale 1/sqrt(head_dim)."""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def patch(src, dst, load_external=False):
    m = onnx.load(src, load_external_data=load_external)
    try:
        inf = onnx.shape_inference.infer_shapes(m, strict_mode=False)
        vis = list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)
    except Exception:
        vis = list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)
    info = {}
    for v in vis:
        tt = v.type.tensor_type
        info[v.name] = (tt.elem_type, [d.dim_value for d in tt.shape.dim])
    for init in m.graph.initializer:
        info.setdefault(init.name, (init.data_type, list(init.dims)))
    out, k = [], 0
    for n in m.graph.node:
        if n.op_type != "Attention":
            out.append(n); continue
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        ins = [i for i in n.input]
        assert len(ins) in (3, 4) and not at.get("is_causal") and not at.get("softcap") \
            and not at.get("qk_matmul_output_mode") and "scale" not in at and "q_num_heads" not in at \
            and len([o for o in n.output if o]) == 1, (n.name, at, ins, list(n.output))
        q, kk, v = ins[:3]
        mask = ins[3] if len(ins) == 4 and ins[3] else None
        dt, shp = info.get(q, (0, []))
        dtk, shpk = info.get(kk, (0, []))
        hd = (shp[-1] if len(shp) == 4 and shp[-1] else 0) or (shpk[-1] if len(shpk) == 4 and shpk[-1] else 0)
        assert hd and dt in (TensorProto.FLOAT, TensorProto.FLOAT16), ("head dim / type unknown", n.name, dt, shp, shpk)
        npdt = np.float16 if dt == TensorProto.FLOAT16 else np.float32
        p = f"attn_dec{k}_"; k += 1
        m.graph.initializer.append(numpy_helper.from_array(np.array(1.0 / np.sqrt(hd), npdt), p + "scale"))
        nodes = [helper.make_node("Mul", [q, p + "scale"], [p + "qs"], name=p + "mul"),
                 helper.make_node("Transpose", [kk], [p + "kt"], perm=[0, 1, 3, 2], name=p + "tr"),
                 helper.make_node("MatMul", [p + "qs", p + "kt"], [p + "sc"], name=p + "qk")]
        sc = p + "sc"
        if mask is not None:
            mdt = info.get(mask, (0, []))[0]
            if mdt == TensorProto.BOOL:
                m.graph.initializer.append(numpy_helper.from_array(np.array(-1e4 if npdt == np.float16 else -1e9, npdt), p + "neg"))
                nodes.append(helper.make_node("Where", [mask, sc, p + "neg"], [p + "scm"], name=p + "where"))
            else:
                assert mdt in (TensorProto.FLOAT, TensorProto.FLOAT16), ("mask type", n.name, mdt)
                src_m = mask
                if mdt != dt:
                    nodes.append(helper.make_node("Cast", [mask], [p + "mc"], to=dt, name=p + "cast")); src_m = p + "mc"
                nodes.append(helper.make_node("Add", [sc, src_m], [p + "scm"], name=p + "add"))
            sc = p + "scm"
        nodes += [helper.make_node("Softmax", [sc], [p + "pr"], axis=-1, name=p + "sm"),
                  helper.make_node("MatMul", [p + "pr", v], [o for o in n.output if o], name=p + "pv")]
        out += nodes
    del m.graph.node[:]
    m.graph.node.extend(out)
    onnx.save(m, dst)
    return k


if __name__ == "__main__":
    print(patch(sys.argv[1], sys.argv[2], len(sys.argv) > 3), "attention nodes replaced")
