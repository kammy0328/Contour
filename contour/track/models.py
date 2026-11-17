"""Model registry, downloads and ONNX Runtime sessions (CUDA -> DirectML -> CPU).

Pose/detector weights are the OpenMMLab RTMPose / RTMW / YOLOX ONNX exports
(Apache-2.0). The matte model is Robust Video Matting (GPL-3.0, fine for
in-house use). Face detection uses OpenCV YuNet (MIT). Person segmentation uses SAM 3.1 (Meta, SAM License; smile-forge ONNX export), SAM 2.1
(Apache-2.0, onnx-community export) as the fallback.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..core.settings import settings

OPENMMLAB = "https://download.openmmlab.com/mmpose/v1/projects/"
HF_MIRROR = "https://huggingface.co/Tau-J/RTMPose/resolve/main/"
HF_SAM2 = "https://huggingface.co/onnx-community/sam2.1-hiera-{m}-ONNX/resolve/main/onnx/"
SAM2_FILES = ["vision_encoder.onnx", "vision_encoder.onnx_data",
              "prompt_encoder_mask_decoder.onnx", "prompt_encoder_mask_decoder.onnx_data"]
HF_SAM31 = "https://huggingface.co/smile-forge/sam3.1-onnx/resolve/main/"
SAM31_FILES = ["image-encoder/image-encoder.onnx", "image-encoder/image-encoder.onnx.data", "click-decoder/click-decoder.onnx"]
HF_SAM2V = "https://huggingface.co/square-zero-labs/sam2.1-tiny-video-onnx/resolve/main/"
SAM2V_FILES = ["onnx/vision_encoder.onnx", "onnx/mask_decoder.onnx", "onnx/memory_encoder.onnx",
               "onnx/memory_attention.onnx", "onnx/pointer_tpos.onnx", "constants.json"]

MODELS: Dict[str, dict] = {
    "det_yolox_m": dict(kind="det", label="YOLOX-m 인물 검출", input=(640, 640),
                        url=OPENMMLAB + "rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip", size_mb=95),
    "det_yolox_x": dict(kind="det", label="YOLOX-x 인물 검출", input=(640, 640),
                        url=OPENMMLAB + "rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip", size_mb=380),
    "pose_rtmpose_x": dict(kind="pose", label="RTMPose-x (몸 17점, 384px)", input=(288, 384), kpts=17,
                           url=OPENMMLAB + "rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
                           size_mb=190),
    "pose_rtmw_x": dict(kind="pose", label="RTMW-x (전신 133점, 얼굴 포함)", input=(288, 384), kpts=133,
                        url=OPENMMLAB + "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
                        size_mb=230),
    "pose_rtmpose_m": dict(kind="pose", label="RTMPose-m (몸 17점, 256px)", input=(192, 256), kpts=17,
                           url=OPENMMLAB + "rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
                           size_mb=55),
    "pose_dwpose_l": dict(kind="pose", label="DWPose-L (전신 133점, 384px)", input=(288, 384), kpts=133,
                          url="https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx", size_mb=135),
    "face_rtm106": dict(kind="face_lm", label="RTMPose-m 얼굴 106점 (256px)", input=(256, 256), kpts=106,
                        url=OPENMMLAB + "rtmposev1/onnx_sdk/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.zip",
                        size_mb=55),
    "hand_rtm21": dict(kind="hand", label="RTMPose-m 손 21점 (256px)", input=(256, 256), kpts=21,
                       url=OPENMMLAB + "rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
                       size_mb=55),
    "face_yunet": dict(kind="face", label="YuNet 얼굴 검출", input=(320, 320),
                       url="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
                       size_mb=1),
    # SAM 2.1 (Meta, Apache-2.0), ONNX export by onnx-community: image encoder + prompt decoder, each with an
    # external weights file. Prompted with each tracked person's box and joints -> one exact mask per person.
    "seg_sam2_s": dict(kind="seg", label="SAM 2.1 small 인물 분할", input=(1024, 1024), size_mb=185,
                       files=[HF_SAM2.format(m="small") + f for f in SAM2_FILES]),
    "seg_sam2_b": dict(kind="seg", label="SAM 2.1 base+ 인물 분할", input=(1024, 1024), size_mb=330,
                       files=[HF_SAM2.format(m="base-plus") + f for f in SAM2_FILES]),
    # SAM 3.1 (Meta, SAM License), ONNX export by smile-forge: image encoder + click decoder. The encoder's opset 23
    # Attention operator has no CUDA kernel, so a copy of its graph with MatMul/Softmax (track/data) is put next to it.
    "seg_sam31": dict(kind="seg", label="SAM 3.1 인물 분할", input=(1008, 1008), size_mb=940,
                      files=[HF_SAM31 + f for f in SAM31_FILES],
                      bundle={"sam31_encoder_gpu.onnx": "image-encoder-gpu.onnx"}),
    # SAM 3.1 finder (same export): every "person" in the picture as an instance mask. Its Attention operators are
    # rewritten on this computer (onnx_patch, needs the onnx package) into finder-gpu.onnx, which CUDA runs.
    "seg_sam31_find": dict(kind="seg", label="SAM 3.1 인물 찾기", input=(1008, 1008), size_mb=120,
                           files=[HF_SAM31 + "finder/finder.onnx"], patch={"finder.onnx": "finder-gpu.onnx"}),
    # SAM 2.1 tiny video tracker (Meta, Apache-2.0), ONNX export with the memory modules by square-zero-labs
    # (Apache-2.0): follows each dancer by their own pixels through crossings (group-dance identities)
    "track_sam2v": dict(kind="track", label="SAM 2.1 비디오 추적 (tiny)", input=(1024, 1024), size_mb=190,
                        files=[HF_SAM2V + f for f in SAM2V_FILES]),
    # clothing parsing (SegFormer-B2 fine-tuned on ATR by mattmdjaga): skin / hair / clothes per pixel
    "parse_clothes": dict(kind="parse", label="옷 영역 분석 (SegFormer-B2)", input=(512, 512), size_mb=110,
                          url="https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx"),
    # face parsing (SegFormer-B5 on CelebAMask-HQ by jonathandinu, non-commercial): skin / neck / hair per pixel
    # around each face, for the retouch masks
    "face_parse": dict(kind="face_parse", label="SegFormer 얼굴 영역 분석", input=(512, 512), size_mb=340,
                       url="https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model.onnx"),
    # face recognition (OpenCV Zoo SFace, Apache-2.0): telling members apart by face
    "face_sface": dict(kind="face_id", label="SFace 얼굴 인식", input=(112, 112), size_mb=37,
                       url="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"),
    "matte_rvm": dict(kind="matte", label="RVM 인물 매트", input=None,
                      url="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
                      size_mb=15),
}

POSE_MODEL_KEYS = ["pose_rtmpose_x", "pose_rtmw_x", "pose_rtmpose_m", "pose_dwpose_l"]


def model_path(key: str) -> Path:
    """Single file models: the .onnx file. Multi file models (spec 'files'): their folder."""
    if key in MODELS and MODELS[key].get("files"):
        return settings().models_path / "onnx" / key
    return settings().models_path / "onnx" / f"{key}.onnx"


def is_downloaded(key: str) -> bool:
    p = model_path(key)
    files = MODELS.get(key, {}).get("files")
    if files:
        return all(_usable(p / Path(u).name) for u in files) and \
            all((p / n).is_file() for n in list((MODELS[key].get("bundle") or {}).values())
                + list((MODELS[key].get("patch") or {}).values()))
    return _usable(p)


MIN_MODEL_BYTES = 1024           # no model (nor weights file) is smaller: an empty or error page download is


def _usable(p: Path) -> bool:
    """present, readable by this user (a file can exist and still be closed to us: see _set_aside) and plausibly a
    model: not empty or a few bytes, and an .onnx file is not text (an error page saved in its place).
    A file that passes and still does not load is set aside by session() and downloaded again."""
    if not p.is_file():
        return False
    try:
        if p.stat().st_size < MIN_MODEL_BYTES:
            return False
        with open(p, "rb") as fh:
            head = fh.read(16)
    except PermissionError:
        return not _set_aside(p)
    except OSError:
        return False
    # an error page saved in place of the model (HTML, JSON, XML): never the start of a protobuf (the first byte of a
    # model is a field tag; any field order is accepted, only text is turned away)
    if p.suffix.lower() == ".onnx" and head.lstrip(b"\xef\xbb\xbf \t\r\n")[:1] in (b"<", b"{", b"["):
        return False
    return True


def _set_aside(p: Path) -> bool:
    """1.9.0's installer downloaded models elevated through tempfile.mkdtemp, whose owner-only ACL (Administrators)
    stayed on the files: the normal user could not open them and the analysis ran without them, silently.
    Such a file is renamed (the folder still lets the user do that) so it is downloaded again."""
    import time
    try:
        p.rename(p.with_name(p.name + f".noaccess{int(time.time())}"))
        return True
    except OSError:
        return False


def _download(url: str, dst: Path, progress: Optional[Callable[[float, str], None]], label: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Contour/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".part")
        got = 0
        try:
            with os.fdopen(fd, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if progress and total:
                        progress(got / total, f"{label} 다운로드 {got / 1e6:.0f}/{total / 1e6:.0f} MB")
            if (total and got != total) or got < MIN_MODEL_BYTES:
                # the connection ended early (no error is raised for that): never kept as if complete
                raise IOError(f"{label}: 다운로드가 중간에 끊겼습니다 ({got / 1e6:.1f}/{total / 1e6:.1f} MB). 다시 시도하세요.")
            shutil.move(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def ensure_model(key: str, progress: Optional[Callable[[float, str], None]] = None) -> Path:
    """Download (once) and return the local ONNX path (or the folder of a multi file model)."""
    out = model_path(key)
    spec = MODELS[key]
    if spec.get("files"):
        out.mkdir(parents=True, exist_ok=True)
        files = spec["files"]
        for i, url in enumerate(files):
            dst = out / Path(url).name
            if _usable(dst):
                continue

            def prog(f, msg, i=i):
                if progress:
                    progress((i + f) / len(files), msg)
            _download(url, dst, prog, spec["label"])
        for src_name, dst_name in (spec.get("patch") or {}).items():
            if not (out / dst_name).is_file():
                from .onnx_patch import patch as _patch          # ImportError without onnx: the caller falls back
                tmp = out / (dst_name + ".part")
                _patch(str(out / src_name), str(tmp), True)
                os.replace(tmp, out / dst_name)
        for src_name, dst_name in (spec.get("bundle") or {}).items():
            src, dst = Path(__file__).parent / "data" / src_name, out / dst_name
            if src.is_file() and (not dst.is_file() or dst.stat().st_size != src.stat().st_size):
                shutil.copyfile(src, dst)
        return out
    if _usable(out):
        return out
    url = spec["url"]
    out.parent.mkdir(parents=True, exist_ok=True)
    # a plain folder, not tempfile.mkdtemp: mkdtemp's owner-only ACL would stay on the model (see _set_aside)
    tmpdir = out.parent / f".download_{key}_{os.getpid()}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        fname = tmpdir / Path(url.split("?")[0]).name
        try:
            _download(url, fname, progress, spec["label"])
        except Exception:
            if url.startswith(OPENMMLAB):
                _download(HF_MIRROR + url[len(OPENMMLAB):], fname, progress, spec["label"])
            else:
                raise
        if fname.suffix == ".zip":
            with zipfile.ZipFile(fname) as z:
                names = [n for n in z.namelist() if n.endswith("end2end.onnx")] or \
                        [n for n in z.namelist() if n.endswith(".onnx")]
                if not names:
                    raise RuntimeError(f"{key}: 압축 파일에 ONNX가 없습니다")
                with z.open(names[0]) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            shutil.move(str(fname), out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out


# --------------------------------------------------------------------------- ONNX Runtime

_provider_lock = threading.Lock()
_providers_cache: Optional[List] = None
_device_label = "CPU"


def _nvidia_dll_dirs() -> List[str]:
    """pip's nvidia-* wheels put nvrtc, cudnn, cublas ... in site-packages\\nvidia\\*\\bin; Windows does not look there on its own."""
    import glob
    bases: List[str] = []
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        bases += list(getattr(spec, "submodule_search_locations", None) or [])
    except Exception:
        pass
    for p in sys.path:
        d = os.path.join(p, "nvidia")
        if os.path.isdir(d) and d not in bases:
            bases.append(d)
    out: List[str] = []
    for b in bases:
        for d in glob.glob(os.path.join(b, "*", "bin")):
            if os.path.isdir(d) and d not in out:
                out.append(d)
    return out


def _preload_cuda() -> None:
    if os.name == "nt":
        try:
            dirs = _nvidia_dll_dirs()
            for d in dirs:
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
            if dirs:
                os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass


def providers() -> List:
    global _providers_cache, _device_label
    with _provider_lock:
        if _providers_cache is not None:
            return _providers_cache
        import onnxruntime as ort
        want = settings().gpu
        avail = ort.get_available_providers()
        chosen: List = []
        if want in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
            _preload_cuda()
            opts = {"cudnn_conv_algo_search": "HEURISTIC"}
            if os.environ.get("DIAP_ARENA", ARENA) == "same":
                # the GPU memory pool grows by what is asked, not to the next power of two (SAM 3.1's attention asks
                # for buffers of a few hundred MB: doubling them filled a 12 GB card)
                opts["arena_extend_strategy"] = "kSameAsRequested"
            chosen.append(("CUDAExecutionProvider", opts))
            _device_label = "CUDA"
        elif want in ("auto", "dml") and "DmlExecutionProvider" in avail:
            chosen.append("DmlExecutionProvider")
            _device_label = "DirectML"
        else:
            _device_label = "CPU"
        chosen.append("CPUExecutionProvider")
        _providers_cache = chosen
        return chosen


def device_label() -> str:
    providers()
    return _device_label


def reset_providers() -> None:
    global _providers_cache
    _providers_cache = None


# session settings (measured on the 4070 Ti, j2682); DIAP_ARENA / DIAP_SPIN / DIAP_THREADS override them
ARENA = "default"           # "same": the CUDA memory pool grows by what is asked (kSameAsRequested)
SPIN = "1"                  # "0": idle threads do not spin

# ---------------------------------------------------------------------------------------------- GPU memory
OOM_MARKERS = ("out of memory", "failed to allocate", "cudnn_status_alloc_failed", "cudaerrormemoryallocation",
               "cuda_error_out_of_memory", "bfcarena", "e_outofmemory", "887a0005", "887a0006")


class GpuMemoryError(RuntimeError):
    pass


GPU_OOM_TEXT = ("GPU 메모리가 부족합니다. GPU를 쓰는 다른 프로그램(다빈치 리졸브 등)이나 함께 돌고 있는 분석·렌더를 "
                "끝낸 뒤 다시 하세요.")
_release_hooks: List = []
cpu_fallbacks: List = []          # (model file, reason): models that run on the CPU although a GPU was chosen


def is_oom(e: BaseException) -> bool:
    t = str(e).lower()
    return isinstance(e, (GpuMemoryError, MemoryError)) or any(m in t for m in OOM_MARKERS)


def on_low_gpu_memory(fn) -> None:
    """fn() is called when the GPU runs out of memory: drop what can be made again later (cached sessions...)."""
    if fn not in _release_hooks:
        _release_hooks.append(fn)


def free_gpu_memory() -> None:
    import gc
    for fn in list(_release_hooks):
        try:
            fn()
        except Exception:
            pass
    gc.collect()
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class Session:
    """An onnxruntime session whose run() gets over a moment without GPU memory: what can be dropped is dropped and
    the run is tried again (another process, the preview or a render often frees its memory a moment later). Still
    without memory, a GpuMemoryError with a message for the user. Everything else is the session's own."""

    RETRIES = 2

    def __init__(self, sess, path):
        self._s = sess
        self._path = str(path)

    def run(self, *a, **k):
        for attempt in range(self.RETRIES + 1):
            try:
                return self._s.run(*a, **k)
            except Exception as e:
                if not is_oom(e):
                    raise
                sys.stderr.write(f"GPU OOM {Path(self._path).name} (try {attempt + 1}): {str(e)[:300]}\n")
                if attempt >= self.RETRIES:
                    raise GpuMemoryError(GPU_OOM_TEXT) from e
                free_gpu_memory()
                time.sleep(1.5 * (attempt + 1))

    def __getattr__(self, name):
        return getattr(self._s, name)


def take_cpu_fallbacks() -> List:
    out = list(cpu_fallbacks)
    cpu_fallbacks.clear()
    return out


def _session_raw(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = int(os.environ.get("DIAP_ORT_LOG", "3"))
    # leave two cores to the rest of the computer (the app's own preview, other programs): an analysis used
    # every core and made the whole PC sluggish
    n = os.cpu_count() or 4
    so.intra_op_num_threads = int(os.environ.get("DIAP_THREADS", "0") or 0) or max(1, n - 2)
    so.inter_op_num_threads = 1
    if os.environ.get("DIAP_SPIN", SPIN) == "0":
        # an idle session's threads stop at once instead of spinning: with a dozen models loaded, their spinning
        # threads took the CPU from the one that was working
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    if "DmlExecutionProvider" in [p if isinstance(p, str) else p[0] for p in providers()]:
        # DirectML requires sequential execution without memory patterns
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    prov = providers()
    on_gpu = any((p if isinstance(p, str) else p[0]) != "CPUExecutionProvider" for p in prov)
    err = None
    for attempt in range(2 if on_gpu else 1):
        try:
            return Session(ort.InferenceSession(str(path), sess_options=so, providers=prov), path)
        except Exception as e:
            err = e
            if not (on_gpu and is_oom(e)) or attempt:
                break
            # out of GPU memory while loading: free what can be freed and try once more on the GPU
            sys.stderr.write(f"GPU OOM loading {Path(path).name}: {str(e)[:300]}\n")
            free_gpu_memory()
            time.sleep(1.0)
    # A broken GPU setup should never stop the app: fall back to CPU (much slower: noted and shown to the user)
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    if on_gpu:
        why = "GPU 메모리 부족" if err is not None and is_oom(err) else "GPU에서 열지 못함"
        sys.stderr.write(f"MODEL ON CPU {Path(path).name} ({why}): {str(err)[:300]}\n")
        cpu_fallbacks.append((Path(path).name, why))
    return Session(sess, path)


class ModelFileError(RuntimeError):
    pass


# what onnxruntime says of a file that is not a whole model (not of one it could not open: locked, no access)
BROKEN_MARKERS = ("invalid_protobuf", "protobuf parsing failed", "unexpected end")


def _broken(e: BaseException) -> bool:
    t = str(e).lower()
    return any(m in t for m in BROKEN_MARKERS)


def session(path: Path):
    """Load an ONNX model. If the file itself cannot be opened (another process holds it, a scanner locks it, a sharing
    violation, errno 13), load a private copy from the temp folder: a silent loss of a model (the face parser) costs
    mask quality without any message. A model file that is not a model (cut off, damaged) is set aside so the next
    ensure_model downloads it again, and a ModelFileError says so."""
    try:
        return _session_raw(path)
    except Exception as first:
        p = Path(path)
        if not p.is_file():
            raise
        if _broken(first) and not isinstance(first, PermissionError):
            import time as _t
            try:
                p.rename(p.with_name(p.name + f".broken{int(_t.time())}"))
            except OSError:
                pass
            sys.stderr.write(f"MODEL BROKEN {p.name}: {str(first)[:300]}\n")
            raise ModelFileError(f"모델 파일이 손상되었습니다 ({p.name}). 다음에 필요할 때 자동으로 다시 받습니다. "
                                 "인터넷 연결을 확인하고 다시 실행하세요.") from first
        import atexit
        d = Path(tempfile.gettempdir()) / "diap_model_cache" / str(os.getpid())
        try:
            d.mkdir(parents=True, exist_ok=True)
            for f in p.parent.iterdir():                       # the model and its external weight files (name.onnx.data)
                if f.name == p.name or f.name.startswith(p.name + "."):
                    shutil.copyfile(f, d / f.name)
            sys.stderr.write(f"MODEL FALLBACK {p.name}: loaded from a copy after: {str(first)[:200]}\n")
            atexit.register(lambda: shutil.rmtree(d, ignore_errors=True))
            return _session_raw(d / p.name)
        except Exception as second:
            sys.stderr.write(f"MODEL FALLBACK FAILED {p.name}: {str(second)[:200]}\n")
            raise first


def gpu_report() -> str:
    try:
        import onnxruntime as ort
        return f"onnxruntime {ort.__version__} · 사용 가능: {', '.join(ort.get_available_providers())} · 선택: {device_label()}"
    except Exception as e:
        return f"onnxruntime 오류: {e}"
