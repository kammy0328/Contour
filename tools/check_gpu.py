"""Prints CUDA / DML / COREML / CPU / NONE: what ONNX Runtime really ran a model on, on this computer.

    python tools/check_gpu.py [--path DIR] [--json]

A model that comes with the app is run once on each accelerator the installed runtime offers, and the runtime's own
record of the run is read: the answer is the accelerator only when operators ran on it (a session that lists
CUDA and runs on the CPU is CPU). No download: it works without a connection. The last line is the answer.
--path DIR puts DIR first on the module path: the installer tries another runtime there before it changes the one
that is installed (setup_env.py). NONE: onnxruntime itself could not be loaded."""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
MODEL = HERE / "contour" / "track" / "data" / "spotnet_v1.onnx"          # convolutions: needs the accelerator's own libraries
TAGS = (("CUDAExecutionProvider", "CUDA"), ("DmlExecutionProvider", "DML"), ("CoreMLExecutionProvider", "COREML"))


def run_on(ort, prov: str) -> dict:
    """One run of the model with this provider first: {'nodes': operators that ran on it, 'cpu': on the CPU}."""
    import numpy as np
    from contour.track import backend as BK
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(tempfile.gettempdir(), f"diap_check_{os.getpid()}")
    s = ort.InferenceSession(str(MODEL), sess_options=so, providers=[prov, "CPUExecutionProvider"])
    listed = prov in s.get_providers()
    i = s.get_inputs()[0]
    s.run(None, {i.name: np.zeros([1, 4, 96, 96], np.float32)})
    f = s.end_profiling()
    place = BK.placement(f)
    try:
        os.remove(f)
    except OSError:
        pass
    return dict(listed=listed, nodes=int((place.get(prov) or {}).get("nodes", 0)),
                cpu=int((place.get("CPUExecutionProvider") or {}).get("nodes", 0)))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    if "--path" in argv:
        sys.path.insert(0, argv[argv.index("--path") + 1])
    out = dict(answer="NONE", tried={}, error="")
    try:
        import onnxruntime as ort
    except Exception as e:
        out["error"] = str(e)[:300]
        print("NONE", e)
        if as_json:
            print(json.dumps(out, ensure_ascii=False))
        return 0
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    out.update(version=str(ort.__version__), available=list(ort.get_available_providers()), answer="CPU")
    for prov, tag in TAGS:
        if prov not in out["available"]:
            continue
        try:
            r = run_on(ort, prov)
            out["tried"][tag] = r
            if r["nodes"] > 0:
                out["answer"] = tag
                break
            print(f"{tag}_NOT_USED listed={r['listed']} operators on it 0, on the CPU {r['cpu']}", file=sys.stderr)
        except Exception as e:
            out["tried"][tag] = dict(error=str(e)[:300])
            print(f"{tag}_FAIL", e, file=sys.stderr)
    if as_json:
        print(json.dumps(out, ensure_ascii=False))
    print(out["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
