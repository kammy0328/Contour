"""The combined network: shared encoder, three heads (face classes, clothing classes, person alpha)."""
from __future__ import annotations

FACE_CLASSES = 19          # CelebAMask-HQ, as face_parse
CLOTH_CLASSES = 18         # ATR, as parse_clothes
OUTPUTS = ("face", "cloth", "alpha")


def build(pretrained: bool = True, width: int = 128):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    class Head(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.b = nn.Sequential(nn.Conv2d(cin, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(True),
                                   nn.Conv2d(256, cout, 1))

        def forward(self, x):
            return self.b(x)

    class Unified(nn.Module):
        """image (N,3,H,W) normalised like ImageNet, H and W divisible by 32 ->
        face logits (N,19,H/4,W/4), cloth logits (N,18,H/4,W/4), alpha (N,1,H,W) in 0..1."""

        def __init__(self):
            super().__init__()
            w = None
            if pretrained:
                try:
                    w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
                except Exception:
                    w = None
            try:
                r = torchvision.models.resnet34(weights=w)
            except Exception as e:
                print("ImageNet 가중치를 받지 못해 처음부터 학습합니다:", e)
                r = torchvision.models.resnet34(weights=None)
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
            self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
            C = width
            self.lat = nn.ModuleList([nn.Conv2d(c, C, 1) for c in (64, 128, 256, 512)])
            self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.GroupNorm(8, C), nn.ReLU(True))
                                         for _ in range(4)])
            self.face = Head(4 * C, FACE_CLASSES)
            self.cloth = Head(4 * C, CLOTH_CLASSES)
            # alpha: coarse at 1/4, refined with the picture itself at 1/2 and full size (hair, motion blur)
            self.a4 = nn.Sequential(nn.Conv2d(4 * C, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(True))
            self.i2 = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(True))
            self.a2 = nn.Sequential(nn.Conv2d(64 + 16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(True))
            self.a1 = nn.Sequential(nn.Conv2d(32 + 3, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 1, 3, padding=1))

        def forward(self, image):
            x = self.stem(image)
            f1 = self.l1(x)
            f2 = self.l2(f1)
            f3 = self.l3(f2)
            f4 = self.l4(f3)
            p = None
            outs = []
            for i, f in zip((3, 2, 1, 0), (f4, f3, f2, f1)):
                q = self.lat[i](f)
                if p is not None:
                    q = q + F.interpolate(p, size=q.shape[-2:], mode="bilinear", align_corners=False)
                p = q
                outs.append(F.interpolate(self.smooth[i](q), size=f1.shape[-2:], mode="bilinear", align_corners=False))
            z = torch.cat(outs, 1)
            a = self.a4(z)
            a = F.interpolate(a, scale_factor=2.0, mode="bilinear", align_corners=False)
            a = self.a2(torch.cat([a, self.i2(image)], 1))
            a = F.interpolate(a, scale_factor=2.0, mode="bilinear", align_corners=False)
            a = torch.sigmoid(self.a1(torch.cat([a, image], 1)))
            return self.face(z), self.cloth(z), a

    return Unified()


def export(model, path: str, size=(512, 512)) -> dict:
    """One ONNX file, input 'image' of any height and width divisible by 32, outputs face, cloth, alpha.
    Checked (loads, runs, shapes, finite, agrees with torch) on a square crop and a 16:9 frame."""
    import numpy as np
    import torch
    from ..safe_install import check_onnx
    model = model.float().eval().cpu()
    x = torch.randn(1, 3, *size)
    kw = dict(input_names=["image"], output_names=list(OUTPUTS), opset_version=17,
              dynamic_axes={"image": {2: "h", 3: "w"}, "face": {2: "h4", 3: "w4"}, "cloth": {2: "h4", 3: "w4"},
                            "alpha": {2: "h", 3: "w"}})
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False
    torch.onnx.export(model, x, path, **kw)
    res = {}
    for name, (h, w) in (("square", size), ("frame", (288, 512))):
        xi = torch.randn(1, 3, h, w)
        with torch.no_grad():
            ref = model(xi)
        for k, (o, sh) in enumerate(zip(ref, ((1, FACE_CLASSES, h // 4, w // 4), (1, CLOTH_CLASSES, h // 4, w // 4),
                                               (1, 1, h, w)))):
            r = _check_output(path, xi.numpy(), k, o.numpy(), sh)
            res[f"{name}_{OUTPUTS[k]}"] = r
    res["ok"] = all(v["ok"] for k_, v in res.items() if isinstance(v, dict))
    return res


def _check_output(path, x, k, ref, shape):
    """check_onnx on output k (check_onnx looks at the first output only)."""
    import numpy as np
    out = dict(ok=False, reason="", shape=None, max_diff=None)
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        o = s.run(None, {"image": x.astype(np.float32)})[k]
    except Exception as e:
        out["reason"] = f"ONNX 로드·추론 실패: {e}"[:200]
        return out
    out["shape"] = list(o.shape)
    if tuple(o.shape) != tuple(shape):
        out["reason"] = f"출력 형태 {o.shape} != {shape}"
        return out
    if not np.isfinite(o).all():
        out["reason"] = "NaN/Inf"
        return out
    d = float(np.abs(o - ref).max())
    out["max_diff"] = d
    tol = 1e-2 if k < 2 else 2e-3
    if d > tol:
        out["reason"] = f"차이 {d:.3g} > {tol}"
        return out
    out["ok"] = True
    return out
