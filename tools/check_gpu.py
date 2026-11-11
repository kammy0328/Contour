"""Prints CUDA / DML / CPU depending on what ONNX Runtime can really use on this machine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    try:
        import onnxruntime as ort
    except Exception as e:
        print("NONE", e)
        return
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    avail = ort.get_available_providers()
    from contour.track import models as M
    try:
        path = M.ensure_model("face_yunet")
    except Exception as e:
        print("CPU", "test model download failed:", e)
        return
    for prov, tag in (("CUDAExecutionProvider", "CUDA"), ("DmlExecutionProvider", "DML")):
        if prov in avail:
            try:
                s = ort.InferenceSession(str(path), providers=[prov, "CPUExecutionProvider"])
                if prov in s.get_providers():
                    print(tag)
                    return
            except Exception as e:
                print(f"{tag}_FAIL", e, file=sys.stderr)
    print("CPU")


if __name__ == "__main__":
    main()
