"""DIAP's own combined model (experiment, off by default): face parsing, clothing and person alpha from one network.

One shared encoder (ResNet-34, ImageNet start) with three heads, taught by the three models the app uses today
(face_parse: SegFormer-B5 face parser, parse_clothes: SegFormer-B2 clothes, matte_rvm: Robust Video Matting) on
crops of the user's own analysed originals:
- teach.py   labels from the teachers, exactly as the app feeds them (face crop / person crop / whole frame)
- net.py     the network and its ONNX export (one file, three outputs, any size divisible by 32)
- train.py   training where a sample only teaches the heads it has labels for; whole projects held out
- bench.py   the teachers and the combined model on the held-out project: agreement, edges, flicker, time, VRAM
The app side is contour/track/unified.py (used only when switched on, falls back to the three models).

Licence note: the student is taught from the teachers' answers. Teaching does not lift the teachers' terms
(face_parse and parse_clothes are non-commercial), so the student is for in-house use like them.
"""
