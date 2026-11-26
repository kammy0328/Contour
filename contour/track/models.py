"""Model registry, downloads and ONNX Runtime sessions (Windows CUDA -> DirectML -> CPU, macOS Core ML -> CPU;
which of them a model gets, what it really ran on and why: track/backend.py).

Pose/detector weights are the OpenMMLab RTMPose / RTMW / YOLOX ONNX exports
(Apache-2.0). The matte model is Robust Video Matting (GPL-3.0, fine for
in-house use). Face detection uses OpenCV YuNet (MIT). Person segmentation uses SAM 3.1 (Meta, SAM License; smile-forge ONNX export), SAM 2.1
(Apache-2.0, onnx-community export) as the fallback.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
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


def legacy_path(key: str) -> Path:
    """Where a model downloaded by the app itself (before model packages) is. Single file models: the .onnx file.
    Multi file models (spec 'files'): their folder."""
    if key in MODELS and MODELS[key].get("files"):
        return settings().models_path / "onnx" / key
    return settings().models_path / "onnx" / f"{key}.onnx"


def packaged(key: str) -> Optional[Path]:
    """The model as an installed package (modelpkg): the version this work is pinned to, else the one in use.
    None when the model is not a package on this computer."""
    try:
        from . import modelpkg as MP
        return MP.resolve(key)
    except Exception:
        return None


def pinned_missing(key: str) -> str:
    """The version this work is pinned to when that version is not installed ('' otherwise): the model is then not
    opened from anywhere else."""
    try:
        from . import modelpkg as MP
        pins = MP.current_pins()
        if pins and key in pins and MP.store().path(key, pins[key]) is None:
            return pins[key]
    except Exception:
        pass
    return ""


def model_path(key: str) -> Path:
    """Where the model is opened from: its installed package (a version that never changes), else where the app
    itself downloaded it (legacy_path)."""
    return packaged(key) or legacy_path(key)


# models an analysis runs with (recorded on the cut, so a result can be told apart from one made with other files)
ANALYSIS_KEYS = ("det_yolox_m", "det_yolox_x", "pose_rtmw_x", "pose_rtmpose_x", "pose_rtmpose_m", "pose_dwpose_l",
                 "face_rtm106", "hand_rtm21", "face_yunet", "face_sface", "matte_rvm", "seg_sam31", "seg_sam31_find",
                 "face_parse")


def fingerprint(key: str) -> str:
    """First 16 hex digits of the SHA-256 of the model's file(s) ('' when not downloaded). Hashed once per file version:
    cached by name, size and modification time in models/hashes.json (a 940 MB model takes a few seconds once)."""
    import hashlib
    try:
        from . import modelpkg as MP
        ver = MP.version_in_use(key)
        pkg = MP.store().package(key, ver) if ver else None
        if pkg is not None:
            # a package's content never changes: from its own list, not read again. The same figure the app's own
            # download of the same files has (below), so a cut analysed before its model came as a package and one
            # analysed after name the same files
            fs = sorted((Path(f.path), f.sha256) for f in pkg.files
                        if not f.path.endswith((".part", ".json")) and ".broken" not in Path(f.path).name
                        and ".noaccess" not in Path(f.path).name)
            if fs:
                h_pkg = hashlib.sha256()
                for fp_, hx in fs:
                    h_pkg.update(f"{fp_.name}:{hx};".encode())
                return h_pkg.hexdigest()[:16]
            return pkg.sha256[:16]
    except Exception:
        pass
    p = legacy_path(key)
    files = sorted(f for f in p.rglob("*") if f.is_file() and not f.name.endswith((".part", ".json"))
                   and ".broken" not in f.name and ".noaccess" not in f.name) if p.is_dir() else ([p] if p.is_file() else [])
    if not files:
        return ""
    cache_f = settings().models_path / "hashes.json"
    try:
        cache = json.loads(cache_f.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    h_all = hashlib.sha256()
    changed = False
    for f in files:
        st = f.stat()
        ck = f"{f.relative_to(settings().models_path).as_posix()}|{st.st_size}|{int(st.st_mtime)}"
        hx = cache.get(ck)
        if hx is None:
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            hx = cache[ck] = h.hexdigest()
            changed = True
        h_all.update(f"{f.name}:{hx};".encode())
    if changed:
        try:
            cache_f.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        except OSError:
            pass
    return h_all.hexdigest()[:16]


def fingerprints(keys=ANALYSIS_KEYS) -> dict:
    out = {}
    keys = list(keys)
    try:
        from . import unified as U
        if keys == list(ANALYSIS_KEYS) and U.switched_on() and U.path().is_file():
            keys.append(U.KEY)              # which model made the face, clothing and matte masks
    except Exception:
        pass
    for k in keys:
        try:
            fp = fingerprint(k)
        except Exception:
            fp = ""
        if fp:
            out[k] = fp
    return out


def is_downloaded(key: str) -> bool:
    if packaged(key) is not None:
        return True
    p = legacy_path(key)
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
    """Download (once) and return the local ONNX path (or the folder of a multi file model). A model installed as
    a package is returned as it is (every file it needs is in the package; nothing is downloaded or made beside
    it). Work pinned to a package version that is not on this computer gets an error that names it, never another
    version in its place."""
    pk = packaged(key)
    if pk is not None:
        return pk
    want = pinned_missing(key)
    if want:
        raise RuntimeError(f"{MODELS.get(key, {}).get('label', key)}: 이 작업을 시작한 모델 버전 {want}이(가) 설치되어 있지 "
                           f"않습니다. 모델 관리에서 그 버전을 설치하거나 다시 분석하세요.")
    out = legacy_path(key)
    spec = MODELS[key]
    try:
        return _ensure_legacy(key, out, spec, progress)
    except OSError as e:
        # no connection, a name that does not resolve, a download cut short (URLError, timeouts and the like are
        # all OSError): what is missing, how to get it without a connection, and what goes on working
        raise ModelMissing(key, str(getattr(e, "reason", e))[:160]) from e


def _ensure_legacy(key: str, out: Path, spec: dict, progress) -> Path:
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
        elif want in ("auto", "coreml") and "CoreMLExecutionProvider" in avail and sys.platform == "darwin":
            # which models are given to Core ML is decided per model (backend.for_model): with '자동' only those
            # checked on a Mac, so until one is the label stays CPU
            from . import backend as BK
            chosen.append(("CoreMLExecutionProvider", dict(BK.COREML_OPTIONS)))
            _device_label = "Core ML" if (want == "coreml" or BK.COREML_OK) else "CPU"
        else:
            _device_label = "CPU"
        chosen.append("CPUExecutionProvider")
        _providers_cache = chosen
        return chosen


def device_label() -> str:
    """The device for the status line: what was decided for this system, corrected by what the models opened so
    far really ran on (backend.summary_label)."""
    providers()
    try:
        from . import backend as BK
        return BK.summary_label(_device_label)
    except Exception:
        return _device_label


def reset_providers() -> None:
    global _providers_cache
    _providers_cache = None


# session settings (measured on the 4070 Ti, j2682); DIAP_ARENA / DIAP_SPIN / DIAP_THREADS override them
# j2730 (2026-11-18, cut 024 of 2611, 12 frames, RTX 4070 Ti 12 GB): with the default pools the SAM 3.1 encoder held
# 8.4 GB after its first run, the GPU sat at 11.9 of 12 GB, the encoder took 9 s a call (median; 108 s mean) and the
# whole computer stuttered (Chrome, the desktop). Pools that grow by what is asked and runs that give spare memory
# back: 0.41 s a call, 1249 s -> 58 s for the analysis, 2.0 GB median and 9.8 GB peak, the matte identical (IoU 1.0,
# mean difference 0). Both on by default.
RUN_SHRINK = "1"            # "1": a model's run gives its arena's spare memory back (not the SAM models)
SHARE_ARENA = "0"           # "1": every session takes its GPU memory from one shared pool
_shared_arena = []


def _share_cuda_arena(so) -> None:
    """One CUDA memory pool for all sessions (each had its own, every one grown to its own peak)."""
    try:
        import onnxruntime as ort
        if not _shared_arena:
            mi = ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
            cfg = ort.OrtArenaCfg(0, 1, -1, -1)           # no cap, grow by what is asked
            ort.create_and_register_allocator(mi, cfg)
            _shared_arena.append(1)
        so.add_session_config_entry("session.use_env_allocators", "1")
    except Exception as e:
        sys.stderr.write(f"shared GPU pool not used: {str(e)[:200]}\n")


ARENA = "same"              # "same": the CUDA memory pool grows by what is asked (kSameAsRequested); "default": doubling
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


def _no_pool(e: BaseException) -> bool:
    """onnxruntime's answer to a run that asks a memory pool the session does not have to give memory back
    ('Did not find an arena based allocator registered for device-id combination in the memory arena shrink list:
    gpu:0'): every run of a model that CUDA was asked for and the runtime opened on the CPU by itself ended with it
    (CUDA not starting: j2805)."""
    t = str(e).lower()
    return "arena shrink list" in t or "arena based allocator" in t


class Session:
    """An onnxruntime session whose run() gets over a moment without GPU memory: what can be dropped is dropped and
    the run is tried again (another process, the preview or a render often frees its memory a moment later). Still
    without memory, a GpuMemoryError with a message for the user. Everything else is the session's own."""

    RETRIES = 2

    def __init__(self, sess, path, shrink: bool = False, verify: bool = False, asked=None, rtx: bool = False):
        self._s = sess
        self._path = str(path)
        self._ro = None
        self._verify = bool(verify)          # the first run's record tells which provider ran each operator
        self._ran = False
        self._asked = list(asked or [])
        # (1.27) a session on TensorRT RTX: its runs one at a time (the provider's own rule for one session is not written
        # down; a run is 0.08 s), and a run it cannot make moves the model to CUDA for the rest of the process
        self._rtx = bool(rtx)
        self._rtx_lock = threading.Lock() if rtx else None
        self.last_used = time.monotonic()
        # the name of the input whose first dimension is a batch of items that do not depend on each other (set
        # by the model's owner, e.g. the pose crops of one frame): with no GPU memory left the batch is halved
        self.batch_input: Optional[str] = None
        if shrink:
            try:
                import onnxruntime as ort
                self._ro = ort.RunOptions()
                # the arena gives back what a run needed above what it keeps: a small model's one big buffer was
                # held for the rest of the analysis, in the memory the SAM encoder needs
                self._ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
            except Exception:
                self._ro = None

    def run(self, *a, **k):
        ours = self._ro is not None and "run_options" not in k and len(a) < 3
        if ours:
            k["run_options"] = self._ro
        self.last_used = time.monotonic()
        for attempt in range(self.RETRIES + 1):
            try:
                t0 = time.perf_counter()
                lk = self._rtx_lock
                if lk is not None:
                    with lk:
                        out = self._s.run(*a, **k)
                else:
                    out = self._s.run(*a, **k)
                if not self._ran:
                    self._first_run(time.perf_counter() - t0)
                return out
            except Exception as e:
                if self._rtx and not is_oom(e) and self._to_cuda(e):
                    return self.run(*a, **k)
                if ours and _no_pool(e):
                    # the session has no memory pool of the kind the option names (opened on the CPU alone, or an
                    # accelerator with a pool of another kind): the run is made without it, from now on
                    self._ro = None
                    k.pop("run_options", None)
                    return self.run(*a, **k)
                if not is_oom(e):
                    twin = self._cpu_twin(e)
                    if twin is not None:
                        k.pop("run_options", None)
                        return twin.run(*a, **k)
                    raise
                sys.stderr.write(f"GPU OOM {Path(self._path).name} (try {attempt + 1}): {str(e)[:300]}\n")
                if attempt >= self.RETRIES:
                    return self._without_memory(e, a, k)
                free_gpu_memory()
                time.sleep(1.5 * (attempt + 1))

    def serialize(self) -> "Session":
        """The runs of this session one at a time from now on (SAM 3.1's encoder: a second run beside the first took its GPU
        memory twice, 10 -> 11.8 GB on a 12 GB card; the frames encoded ahead and a close-up's encoding now take turns)."""
        if self._rtx_lock is None:
            self._rtx_lock = threading.Lock()
        return self

    def _to_cuda(self, err) -> bool:
        """(TensorRT RTX) a run the provider could not make: the model is opened on CUDA, as it was before 1.27, and stays there
        for the rest of this process (said in the diagnostics, with the error). False when that did not work either."""
        from . import backend as BK
        name = Path(self._path).name
        sys.stderr.write(f"TENSORRT RTX {name}: run failed, going on with CUDA: {str(err)[:300]}\n")
        try:
            _rtx_failed.add(str(self._path))
            s2 = _session_raw(Path(self._path))
        except Exception as e:
            sys.stderr.write(f"TENSORRT RTX {name}: CUDA session failed too: {str(e)[:300]}\n")
            return False
        self._s, self._ro, self._verify, self._asked = s2._s, s2._ro, s2._verify, s2._asked
        self._rtx, self._ran = False, False             # (the lock stays: runs one at a time on CUDA too)
        BK.note(self._path, reasons=[f"TensorRT RTX 실행 오류로 CUDA로 전환 ({str(err)[:120]})"])
        return True

    def _cpu_twin(self, err):
        """(Mac, Core ML) a run Core ML cannot make (a picture with no detection gives a tensor with no elements, which the
        Core ML provider refuses) is made by a CPU session of the same file; None when the error is not Core ML's."""
        try:
            if "CoreML" not in str(err) or "CoreMLExecutionProvider" not in list(self._s.get_providers()):
                return None
            if getattr(self, "_twin", None) is None:
                import onnxruntime as ort
                self._twin = ort.InferenceSession(self._path, sess_options=_options(False), providers=["CPUExecutionProvider"])
                sys.stderr.write(f"COREML FALLBACK {Path(self._path).name}: {str(err)[:200]}\n")
            return self._twin
        except Exception:
            return None

    def _first_run(self, seconds: float) -> None:
        """After the first run: which provider ran the operators (the session's profile), kept with the reasons."""
        self._ran = True
        from . import backend as BK
        place = {}
        if self._verify:
            self._verify = False
            try:
                f = self._s.end_profiling()
                place = BK.placement(f)
                try:
                    os.remove(f)
                except OSError:
                    pass
            except Exception:
                place = {}
        try:
            got = [p for p in self._s.get_providers()]
        except Exception:
            got = []
        if place:
            backend, cpu_share = BK.ran_on(place)
        else:
            # no profile (a CPU session, or profiling not available): the session's own first provider
            backend, cpu_share = (BK.BACKEND.get(got[0], "") if got else ""), 0.0
        why = []
        asked_acc = [b for b in self._asked if b != "cpu"]
        name = Path(self._path).name
        if asked_acc and backend == "cpu":
            why.append("가속 장치를 요청했지만 연산이 모두 CPU에서 실행됨")
            if not any(n == name for n, _ in cpu_fallbacks):
                cpu_fallbacks.append((name, "GPU 실행 장치를 쓰지 못함"))
        elif asked_acc and place and cpu_share > BK.CPU_HEAVY:
            why.append(f"연산 시간의 {cpu_share * 100:.0f}%가 CPU에서 실행됨 (가속 장치에 없는 연산)")
        BK.note(self._path, ran=True, backend=backend, placement=place, cpu_share=round(cpu_share, 3),
                first_ms=round(seconds * 1000.0, 1), session=got, reasons=why)

    def _without_memory(self, err, a, k):
        """The GPU has no memory for this run even after what could be dropped was dropped: a batch is run in
        halves; a model the CPU can run in reasonable time goes on there (said once, with the reason); the others
        end with the message that says what to close."""
        from . import backend as BK
        name = Path(self._path).name
        feed = a[1] if len(a) > 1 else k.get("input_feed")
        bi = self.batch_input
        if bi and isinstance(feed, dict) and bi in feed and len(feed[bi]) > 1:
            n = len(feed[bi])
            h = n // 2
            BK.note(self._path, reasons=[f"GPU 메모리 부족: 배치 {n}개를 나눠 실행"])
            sys.stderr.write(f"GPU OOM {name}: batch {n} -> {h} + {n - h}\n")
            outs = []
            for part in (slice(0, h), slice(h, n)):
                f2 = dict(feed)
                f2[bi] = feed[bi][part]
                outs.append(self.run(a[0] if a else k.get("output_names"), f2))
            return [np.concatenate([x[i] for x in outs]) if getattr(outs[0][i], "shape", ())[:1] == (h,)
                    else outs[0][i] for i in range(len(outs[0]))]
        if any(name.startswith(x) for x in BK.CPU_TOO_SLOW) or os.environ.get("DIAP_OOM_CPU", "1") == "0":
            raise GpuMemoryError(GPU_OOM_TEXT) from err
        try:
            import onnxruntime as ort
            self._s = ort.InferenceSession(self._path, sess_options=_options(False), providers=["CPUExecutionProvider"])
        except Exception:
            raise GpuMemoryError(GPU_OOM_TEXT) from err
        self._ro = None
        self.batch_input = bi
        sys.stderr.write(f"MODEL ON CPU {name} (GPU 메모리 부족)\n")
        cpu_fallbacks.append((name, "GPU 메모리 부족"))
        BK.note(self._path, backend="cpu", session=["CPUExecutionProvider"],
                reasons=["GPU 메모리 부족: 이 모델을 CPU로 전환 (다른 GPU 프로그램을 끝내고 다시 시작하면 GPU로 돌아갑니다)"])
        return self._s.run(*a, **{kk: v for kk, v in k.items() if kk != "run_options"})

    def __getattr__(self, name):
        return getattr(self._s, name)


def _check_actual(sess, prov, path) -> None:
    """onnxruntime can open a session on the CPU by itself (the GPU provider's libraries do not load) and say nothing:
    what it actually runs on is compared with what was asked, the display and the warning follow the truth."""
    global _device_label
    want = [p if isinstance(p, str) else p[0] for p in prov]
    gpu = [p for p in want if p != "CPUExecutionProvider"]
    if not gpu:
        return
    try:
        got = list(sess.get_providers())
    except Exception:
        return
    if not any(p in got for p in gpu):
        sys.stderr.write(f"MODEL ON CPU {Path(path).name} (onnxruntime chose {got} over {gpu})\n")
        cpu_fallbacks.append((Path(path).name, "GPU 실행 장치를 쓰지 못함"))
        if not _device_label.endswith("일부 CPU)"):
            _device_label = f"{_device_label} (일부 CPU)" if _device_label != "CPU" else _device_label


def take_cpu_fallbacks() -> List:
    out = list(cpu_fallbacks)
    cpu_fallbacks.clear()
    return out


_prof_n = [0]
_verified: set = set()          # model files whose first run was looked at in this process (once is enough)
_rtx_failed: set = set()        # model files TensorRT RTX could not open or run in this process: CUDA from then on


def rtx_cache_dir() -> Path:
    """Where TensorRT RTX keeps what it compiled for this GPU. Beside the models, on this computer; made again by itself
    when it is gone."""
    if os.environ.get("DIAP_RTX_CACHE"):                    # (a test's own folder)
        d = Path(os.environ["DIAP_RTX_CACHE"])
    else:
        try:
            d = settings().models_path / "rtx_cache"
        except Exception:
            d = Path(tempfile.gettempdir()) / "diap_rtx_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


RTX_COMPILED = True     # a model's compiled copy is kept and opened the next time (0.9 s against 8 to 13: the batch analysis opens every model once a cut)
RTX_COMPILED_MIN_FREE = 3 << 30      # free space the drive keeps after a compiled copy (SAM 3.1's encoder: 0.9 GB) is written


def _rtx_compiled_dir(path: Path) -> Path:
    """The folder of this model file's compiled copy: one for the file as it is now, this GPU, its driver and this version of
    the plugin (any of them changed: compiled again, the old copy removed)."""
    import hashlib
    from . import backend as BK
    p = Path(path)
    try:
        st = p.stat()
        ident = f"{p.name}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        ident = p.name
    g = BK.gpu()
    r = BK.rtx()
    key = hashlib.sha256(f"{ident}|{g.get('name')}|{g.get('driver')}|{r.get('version')}".encode()).hexdigest()[:12]
    return rtx_cache_dir() / f"{p.stem}.{key}"


def _rtx_open(path: Path, devices: list, cache: Path, write_to: Optional[Path] = None):
    import onnxruntime as ort
    so = _options(False)
    if write_to is not None:
        # onnxruntime's "EP context" model: the provider's compiled engine written beside a small model that names it
        so.add_session_config_entry("ep.context_enable", "1")
        so.add_session_config_entry("ep.context_file_path", str(write_to))
        so.add_session_config_entry("ep.context_embed_mode", "0")
    so.add_provider_for_devices(devices, {"nv_runtime_cache_path": str(cache)})
    return ort.InferenceSession(str(path), sess_options=so)


def _rtx_compile(path: Path, devices: list, d: Path, ctx_name: str) -> bool:
    """Writes the model's compiled copy into folder d (whole or not at all): compiled by a session of its own that is closed
    before the folder gets its name (an open session holds the engine file). False when there is no room for it, the
    provider wrote none, or anything failed: the model is then opened as it is."""
    import gc
    name = Path(path).name
    root = rtx_cache_dir()
    for old in root.glob("*.tmp*"):                         # left by a process that ended while compiling
        try:
            if time.time() - old.stat().st_mtime > 600:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    tmp = d.with_name(d.name + f".tmp{os.getpid()}")
    try:
        free = shutil.disk_usage(root).free
        need = Path(path).stat().st_size + sum(f.stat().st_size for f in Path(path).parent.glob(name + ".*") if f.is_file())
        if free < need + RTX_COMPILED_MIN_FREE:
            sys.stderr.write(f"TENSORRT RTX {name}: no room for a compiled copy ({free / 1e9:.1f} GB free), compiled every time\n")
            return False
        for old in root.glob(Path(path).stem + ".*"):       # this model's copies for another driver or plugin version
            if old.is_dir() and old != d and ".tmp" not in old.name:
                shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        s0 = _rtx_open(Path(path), devices, root, write_to=tmp / ctx_name)
        del s0
        gc.collect()
        if not (tmp / ctx_name).is_file():
            try:
                d.with_name(d.name + ".none").write_text("the provider wrote no compiled copy", encoding="utf-8")   # not tried again
            except OSError:
                pass
            return False
        for k in range(5):
            try:
                tmp.rename(d)
                return True
            except OSError:
                if d.exists():
                    return (d / ctx_name).is_file()         # another process put its own there meanwhile
                time.sleep(0.3)
        return False
    except Exception as e:
        sys.stderr.write(f"TENSORRT RTX {name}: no compiled copy kept: {str(e)[:200]}\n")
        return False
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def _rtx_session(path: Path, devices: list, want: str):
    """The model's session on TensorRT RTX (backend.rtx_for said it may), or None with the reason noted: the caller goes on
    with CUDA. The provider compiles the model for this GPU when a session is made (SAM 3.1's encoder: 13 s); the compiled
    copy is kept beside the models (rtx_cache, 0.9 GB for the encoder) and opened from then on (0.9 s; j2891)."""
    from . import backend as BK
    name = Path(path).name
    t0 = time.perf_counter()
    try:
        sess, how = None, "compiled now"
        d = _rtx_compiled_dir(Path(path)) if RTX_COMPILED and os.environ.get("DIAP_RTX_COMPILED", "1") != "0" else None
        if d is not None:
            ctx = d / (Path(path).stem + "_ctx.onnx")
            if d.exists() and not ctx.is_file():
                shutil.rmtree(d, ignore_errors=True)
            made = ctx.is_file() or (not d.with_name(d.name + ".none").exists() and _rtx_compile(Path(path), devices, d, ctx.name))
            if made:
                try:
                    sess, how = _rtx_open(ctx, devices, rtx_cache_dir()), "compiled copy"
                    if BK.PROVIDER["rtx"] not in list(sess.get_providers()):
                        raise RuntimeError("the compiled copy did not open on TensorRT RTX")
                except Exception as e:
                    # a copy that does not open (damaged, of another build): removed; compiled again the next time
                    sys.stderr.write(f"TENSORRT RTX {name}: compiled copy not usable: {str(e)[:200]}\n")
                    sess = None
                    shutil.rmtree(d, ignore_errors=True)
        if sess is None:
            sess, how = _rtx_open(Path(path), devices, rtx_cache_dir()), "compiled now"
        got = list(sess.get_providers())
        if BK.PROVIDER["rtx"] not in got:
            raise RuntimeError(f"세션이 {got}(으)로 열림")
        BK.note(path, asked=["rtx", "cuda", "cpu"], session=got, setting=want, opened=how,
                open_ms=round((time.perf_counter() - t0) * 1000.0, 1))
        return Session(sess, path, asked=["rtx", "cuda", "cpu"], rtx=True)
    except Exception as e:
        _rtx_failed.add(str(path))
        sys.stderr.write(f"TENSORRT RTX {name}: not used, going on with CUDA: {str(e)[:300]}\n")
        BK.note(path, reasons=[f"TensorRT RTX로 열지 못해 CUDA로 실행 ({str(e)[:120]})"])
        return None


def _options(dml: bool):
    """The session settings every model gets (threads, spinning; DirectML's own needs)."""
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
    if dml:
        # DirectML requires sequential execution without memory patterns
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def _why_not_gpu(err) -> str:
    """Why a model that was asked to run on an accelerator was opened on the CPU, for the person."""
    from . import backend as BK
    if err is not None and is_oom(err):
        return "GPU 메모리 부족"
    state, text = BK.driver_state()
    if state == "old":
        return text
    t = str(err or "").lower()
    if "cudnn" in t or "cublas" in t or "cufft" in t or "nvrtc" in t or ".dll" in t or "libcu" in t or "load library" in t:
        return ("CUDA 라이브러리를 불러오지 못함 (" + str(err)[:120] + ")")
    return "GPU에서 열지 못함" + (f" ({str(err)[:120]})" if err is not None else "")


def _session_raw(path: Path):
    import onnxruntime as ort
    from . import backend as BK
    base = providers()
    names = [p if isinstance(p, str) else p[0] for p in base]
    so = _options("DmlExecutionProvider" in names)
    try:
        want = settings().gpu
    except Exception:
        want = "auto"
    # the system's list narrowed for this model (a package that is for other backends, Core ML only for the
    # models checked on it), with why
    prov, why = BK.for_model(path, base, want)
    asked = [BK.BACKEND.get(p if isinstance(p, str) else p[0], "") for p in prov]
    on_gpu = any((p if isinstance(p, str) else p[0]) != "CPUExecutionProvider" for p in prov)
    err = None
    name = Path(path).name
    if "CUDAExecutionProvider" in [p if isinstance(p, str) else p[0] for p in prov] and str(path) not in _rtx_failed:
        # (1.27) a model checked on TensorRT RTX is opened there first, on a GPU the plugin runs on; anything else: CUDA below
        try:
            allow = getattr(settings(), "rtx", "off")
        except Exception:
            allow = "off"
        try:
            devs, rwhy = BK.rtx_for(path, want, allow)
        except Exception as e:
            devs, rwhy = [], f"TensorRT RTX 확인 실패 ({str(e)[:120]})"
        if devs:
            s2 = _rtx_session(Path(path), devs, want)
            if s2 is not None:
                return s2
        elif rwhy:
            why = why + ["TensorRT RTX: " + rwhy]
    # (the GPU memory pool settings are CUDA's and DirectML's, as before; Core ML has no pool of ours)
    pooled = on_gpu and "CoreMLExecutionProvider" not in names
    shrink = pooled and os.environ.get("DIAP_RUN_SHRINK", RUN_SHRINK) == "1" and not name.startswith(
        ("image-encoder", "finder", "click-decoder"))
    if pooled and os.environ.get("DIAP_SHARE_ARENA", SHARE_ARENA) == "1":
        _share_cuda_arena(so)
    # the first run of a model on an accelerator writes which provider ran each operator (once per model file in
    # this process): the name of a provider in the session's list is not taken for acceleration
    verify = on_gpu and os.environ.get("DIAP_VERIFY_BACKEND", "1") != "0" and str(path) not in _verified
    if verify:
        try:
            _prof_n[0] += 1
            so.enable_profiling = True
            so.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"diap_place_{os.getpid()}_{_prof_n[0]}")
        except Exception:
            verify = False
    for attempt in range(2 if on_gpu else 1):
        try:
            sess = ort.InferenceSession(str(path), sess_options=so, providers=prov)
            _check_actual(sess, prov, path)
            if verify:
                _verified.add(str(path))
            try:
                got = list(sess.get_providers())
            except Exception:
                got = []
            if on_gpu and not any(BK.BACKEND.get(g, "cpu") != "cpu" for g in got):
                # opened, but on the CPU alone (the accelerator's libraries did not load): with the reason
                why = why + [_why_not_gpu(None) if BK.driver_state()[0] == "old" else
                             "가속 장치를 열지 못해 CPU로 실행 (드라이버·CUDA 라이브러리 확인)"]
            BK.note(path, asked=asked, session=got, reasons=why, setting=want)
            # (the run's option that gives the pool's spare memory back is CUDA's: by what the session has, not by
            # what was asked for. A session the runtime opened on the CPU by itself has no such pool, and every run
            # ended with an error: j2805)
            return Session(sess, path, shrink and "CUDAExecutionProvider" in got, verify=verify, asked=asked)
        except Exception as e:
            err = e
            if not (on_gpu and is_oom(e)) or attempt:
                break
            # out of GPU memory while loading: free what can be freed and try once more on the GPU
            sys.stderr.write(f"GPU OOM loading {name}: {str(e)[:300]}\n")
            free_gpu_memory()
            time.sleep(1.0)
    # A broken GPU setup should never stop the app: fall back to CPU (much slower: noted and shown to the user)
    sess = ort.InferenceSession(str(path), sess_options=_options(False), providers=["CPUExecutionProvider"])
    if on_gpu:
        short = "GPU 메모리 부족" if err is not None and is_oom(err) else "GPU에서 열지 못함"
        sys.stderr.write(f"MODEL ON CPU {name} ({short}): {str(err)[:300]}\n")
        cpu_fallbacks.append((name, short))
        why = why + [_why_not_gpu(err)]
    BK.note(path, asked=asked, session=["CPUExecutionProvider"], reasons=why, setting=want)
    return Session(sess, path, asked=asked)


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
            own = _package_of(p)
            if own is not None:
                # a file of an installed package: never renamed or fetched again on its own. When its files are
                # no longer what the list said, the version is taken out of use (the one in use before it comes
                # back) and the message says which package to install again
                key, ver = own
                from . import modelpkg as MP
                now = None
                bad = not MP.store().verify(key, ver)
                if bad:
                    now = MP.store().set_aside(key, ver)
                sys.stderr.write(f"MODEL PACKAGE BROKEN {key} {ver} ({p.name}) verified={not bad}: {str(first)[:300]}\n")
                raise ModelFileError(
                    f"{MODELS.get(key, {}).get('label', key)} {ver}: 모델 파일을 열지 못했습니다 ({p.name}). "
                    + ((f"설치된 파일이 손상되어 사용을 멈췄습니다. 이전 버전 {now}을(를) 씁니다. " if now else
                        "설치된 파일이 손상되어 사용을 멈췄습니다. ") + "모델 관리에서 이 패키지를 다시 설치하세요."
                       if bad else "파일은 목록과 같습니다: 이 버전의 DIAP·ONNX Runtime과 맞는 패키지인지 확인하세요.")) from first
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


# ---------------------------------------------------------------------------------------------- sessions kept between uses
KEEP_S = 120.0          # seconds a session opened with session_kept stays loaded after its last run
_kept: Dict[tuple, "Session"] = {}
_kept_lock = threading.Lock()
_kept_sweeper: List = [None]


def _kept_key(path) -> tuple:
    p = Path(path)
    try:
        st = p.stat()
        return (str(p), st.st_size, st.st_mtime_ns)
    except OSError:
        return (str(p), 0, 0)


def _drop_kept() -> None:
    with _kept_lock:
        _kept.clear()


def _sweep_once(now: Optional[float] = None) -> int:
    """Lets go of the kept sessions not run for KEEP_S seconds; how many are still kept."""
    now = time.monotonic() if now is None else now
    with _kept_lock:
        for k in [k for k, s in _kept.items() if now - getattr(s, "last_used", now) > KEEP_S]:
            _kept.pop(k, None)
        return len(_kept)


def _sweep_kept() -> None:
    while True:
        time.sleep(15.0)
        if _sweep_once() == 0:
            with _kept_lock:
                if not _kept:
                    _kept_sweeper[0] = None
                    return


def session_kept(path: Path):
    """session(path), the same one for everybody in this process while it is in use (1.27). SAM 3.1's encoder, decoder and
    finder were opened by the person matte, by the clothes pass and by every matte made again after an edit, each for
    itself: 1 to 3 s on CUDA and about 10 on TensorRT RTX every time, and the encoder's GPU memory twice where two of them
    met. A session here is let go KEEP_S seconds after its last run (its GPU memory with it, once nobody holds it) and
    at once when the GPU runs out of memory; a model file that was replaced gets a session of its own."""
    if os.environ.get("DIAP_KEEP_SESSIONS", "1") == "0":
        return session(path)
    key = _kept_key(path)
    with _kept_lock:
        s = _kept.get(key)
        if s is not None:
            s.last_used = time.monotonic()
            return s
    s = session(path)                       # (not under the lock: opening takes seconds)
    with _kept_lock:
        s0 = _kept.get(key)
        if s0 is not None:
            return s0                       # opened by another thread meanwhile: that one for both
        _kept[key] = s
        try:
            s.last_used = time.monotonic()
        except Exception:
            pass
        if _kept_sweeper[0] is None:
            on_low_gpu_memory(_drop_kept)
            th = threading.Thread(target=_sweep_kept, name="model-keeper", daemon=True)
            _kept_sweeper[0] = th
            th.start()
    return s


def _package_of(path) -> Optional[tuple]:
    try:
        from . import modelpkg as MP
        return MP.store().owns(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------- what work needs
# Which models a kind of work opens (the work itself asks for them with ensure_model; this is for telling the
# person, before they start and without a connection, what is missing and what goes on working without it).
# Retouch, preview and render run on models that are part of the app (track/data, assets/onnx): never listed here.
TASKS = (
    ("prescan", "빠른 검사 (인물 없는 컷 건너뛰기)", "없으면 모든 컷을 일반 분석으로 넘깁니다"),
    ("analyze", "분석 (추적·인물 매트)", "없으면 새 분석·다시 분석·매트 갱신을 할 수 없습니다"),
    ("clothes", "옷 영역", "없으면 옷 영역을 새로 만들 수 없습니다 (저장된 옷 영역은 그대로 씁니다)"),
    ("face", "얼굴·목 영역", "없으면 얼굴·목 영역을 새로 만들 수 없습니다 (저장된 영역은 그대로 씁니다)"),
)
ALWAYS = "프로젝트 열기·컷 전환·보정값 편집·기존 분석 결과 보기·미리보기·렌더는 모델 다운로드 없이 됩니다."


def task_models(task: str, accuracy: str = "") -> List[str]:
    """The keys the task opens with the settings as they are (accuracy 'max' adds its extra models)."""
    if task == "prescan":
        return ["det_yolox_m"]
    if task == "analyze":
        keys = ["det_yolox_m", "pose_rtmpose_x", "pose_rtmw_x", "pose_rtmpose_m", "face_yunet", "face_rtm106",
                "hand_rtm21", "face_sface", "matte_rvm", "seg_sam31", "seg_sam31_find"]
        if accuracy == "max":
            keys += ["det_yolox_x", "pose_dwpose_l"]
        return keys + [k for k in task_models("face") if k not in keys]
    if task == "clothes":
        try:
            from . import clothing as CL
            return ["parse_clothes"] if CL.cloth_model() == "parse_clothes" else ["seg_sam31", "seg_sam31_find"]
        except Exception:
            return ["seg_sam31", "seg_sam31_find"]
    if task == "face":
        try:
            from . import faceparse as FP
            return [] if FP.face_model() == "mp" else [FP.MODEL_KEY]         # 'mp' is part of the app
        except Exception:
            return []
    return []


def missing(task: str, accuracy: str = "") -> List[str]:
    return [k for k in task_models(task, accuracy) if not is_downloaded(k)]


def availability(accuracy: str = "") -> List[dict]:
    """For every kind of work: [{task, label, ok, missing: [{key, label, size_mb}], limit}]."""
    out = []
    for task, label, limit in TASKS:
        miss = missing(task, accuracy)
        out.append(dict(task=task, label=label, ok=not miss, limit="" if not miss else limit,
                        missing=[dict(key=k, label=MODELS.get(k, {}).get("label", k),
                                      size_mb=MODELS.get(k, {}).get("size_mb", 0)) for k in miss]))
    return out


class ModelMissing(RuntimeError):
    """A model the work needs is not on this computer and could not be had (no connection, no package)."""

    def __init__(self, key: str, why: str = ""):
        spec = MODELS.get(key, {})
        super().__init__(f"{spec.get('label', key)} 모델이 이 컴퓨터에 없고 받지 못했습니다"
                         + (f" ({why})" if why else "") + ". 인터넷에 연결한 뒤 다시 하거나, 설정의 모델 관리에서 모델 패키지 "
                         "파일을 가져오세요. " + ALWAYS)
        self.key = key


def gpu_report() -> str:
    """One line for the settings: the runtime, what it offers, the device read from the system, the choice."""
    try:
        import onnxruntime as ort
        from . import backend as BK
        g = BK.gpu()
        dev = (f" · {g['name']} {g.get('memory_mb', 0) // 1024} GB · 드라이버 {g.get('driver')}" if g.get("vendor") == "nvidia"
               else f" · {g['name']}" if g.get("vendor") == "apple" else "")
        return (f"onnxruntime {ort.__version__} · 사용 가능: {', '.join(ort.get_available_providers())}{dev}"
                f" · 선택: {device_label()}")
    except Exception as e:
        return f"onnxruntime 오류: {e}"
