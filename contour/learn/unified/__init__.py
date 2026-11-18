"""DIAP's own combined model (experiment, off by default): face parsing, clothing and person alpha from one network.

One shared encoder (ResNet-34, ImageNet start) with three heads, taught on crops of the user's own analysed
originals by teachers whose terms allow commercial use (2026-11-19; the first version learnt from the
non-commercial SegFormer parsers):
- face: Google MediaPipe Selfie Multiclass (Apache-2.0): background, hair, body skin, face skin, clothes, others;
- clothing: SAM 3.1's finder with garment phrases (SAM License), in the app's clothes classes;
- alpha: Robust Video Matting (GPL-3.0: its answers are not under the GPL; the model itself is not shipped).
- teach.py   labels from the teachers, exactly as the app feeds them (face crop / person crop / whole frame)
- net.py     the network and its ONNX export (one file, three outputs, any size divisible by 32)
- train.py   training where a sample only teaches the heads it has labels for; whole projects held out
- bench.py   the teachers and the combined model on the held-out project: agreement, edges, flicker, time, VRAM
The app side is contour/track/unified.py (used only when switched on, falls back to the three models).

Licence note: a student taught by a teacher is bound by what the teacher's licence allows its use for, so only
teachers that allow commercial use are used here.
"""
