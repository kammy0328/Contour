# DIAP third-party notices

DIAP runs these programs and models. The models are downloaded on first use (설정 > 모델 관리), except the ones marked
"shipped", which come with the app. Each is under its own licence; the full texts of the ones that ask for it are in
`contour/track/data/` and in this file.

## Programs

| Component | Used for | Licence |
|---|---|---|
| FFmpeg (gyan.dev "essentials" build, shipped in `ffmpeg\bin`) | reading and writing video | GPL-3.0 (the build includes x264 / x265). Run as a separate program. Source: https://ffmpeg.org and https://github.com/GyanD/codexffmpeg; the build's own licence text: `ffmpeg\LICENSE.txt` |
| Python, NumPy, SciPy, OpenCV, OpenColorIO, ONNX Runtime, ONNX | the app itself | PSF, BSD-3-Clause, Apache-2.0, MIT (each package's own files) |
| PySide6 (Qt) | the window | LGPL-3.0 |
| PyTorch, torchvision (installed with the learning features) | GPU retouch steps, learning | BSD-3-Clause |

## Models used by default

| Model | Used for | Licence |
|---|---|---|
| RTMPose / RTMW / YOLOX (OpenMMLab) | people, poses, faces, hands | Apache-2.0 |
| YuNet, SFace (OpenCV Zoo) | faces, telling members apart | Apache-2.0 / MIT |
| OSNet-IBN re-identification (shipped, `assets/onnx`) | telling people apart | MIT (torchreid) |
| MediaPipe Selfie Multiclass (shipped, `contour/track/data`) | face parsing (skin, hair) | Apache-2.0 (`MEDIAPIPE_LICENSE.txt`) |
| SAM 3.1 (Meta; ONNX export by smile-forge; the GPU graph and text encodings shipped in `contour/track/data`) | people's masks, clothes, body skin | SAM License (`SAM_LICENSE.txt`): commercial use allowed; the licence goes with any copy; no use for ITAR / trade-controlled ends |
| SAM 2.1 (Meta; ONNX exports by onnx-community, square-zero-labs) | fallback masks, video tracking | Apache-2.0 |
| Robust Video Matting (PeterL1n) | soft hair edges of the person matte | GPL-3.0 (its terms apply when DIAP is handed to others; DIAP is used in house) |
| DIAP spot detector (shipped, `spotnet_v1.onnx`) | blemish finding | DIAP's own |

## Not used unless switched on by hand (non-commercial)

These are off in the app and need `DIAP_NONCOMMERCIAL=1` to be used at all. Their licences allow research and education only.

| Model | Licence |
|---|---|
| SegFormer face parsing (jonathandinu/face-parsing, CelebAMask-HQ) | non-commercial (CelebAMask-HQ) |
| SegFormer-B2 clothes (mattmdjaga/segformer_b2_clothes, ATR) | NVIDIA Source Code License (non-commercial) |
