"""Model registry, downloads and ONNX Runtime sessions (CUDA -> DirectML -> CPU).

Pose/detector weights are the OpenMMLab RTMPose / RTMW / YOLOX ONNX exports
(Apache-2.0). The matte model is Robust Video Matting (GPL-3.0, fine for
in-house use). Face detection uses OpenCV YuNet (MIT). Person segmentation uses SAM 2.1 (Meta,
Apache-2.0) in the onnx-community ONNX export.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
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
        return all((p / Path(u).name).exists() for u in files)
    return p.exists()


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
            if dst.exists():
                continue

            def prog(f, msg, i=i):
                if progress:
                    progress((i + f) / len(files), msg)
            _download(url, dst, prog, spec["label"])
        return out
    if out.exists():
        return out
    url = spec["url"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(dir=str(out.parent)))
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


def _preload_cuda() -> None:
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
            chosen.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
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


def session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    # leave two cores to the rest of the computer (the app's own preview, other programs): an analysis used
    # every core and made the whole PC sluggish
    n = os.cpu_count() or 4
    so.intra_op_num_threads = max(1, n - 2)
    so.inter_op_num_threads = 1
    if "DmlExecutionProvider" in [p if isinstance(p, str) else p[0] for p in providers()]:
        # DirectML requires sequential execution without memory patterns
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        return ort.InferenceSession(str(path), sess_options=so, providers=providers())
    except Exception:
        # A broken GPU setup should never stop the app: fall back to CPU.
        return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def gpu_report() -> str:
    try:
        import onnxruntime as ort
        return f"onnxruntime {ort.__version__} · 사용 가능: {', '.join(ort.get_available_providers())} · 선택: {device_label()}"
    except Exception as e:
        return f"onnxruntime 오류: {e}"
