"""Step training of DIAP's OWN face parser on one collected project (run by the developer with step_train.bat).

Cloud\\data\\<project>\\<cut id>\\*.jpg are the face crops the app gathered while analysing (originals only, no
labels). The parser DIAP ships today (SegFormer, non-commercial licence) is used as a TEACHER only: it looks at
clean, normalised and mirrored views of a crop and its averaged answer is the label. The STUDENT is a different
network that shares no weights with it (ResNet-34 encoder from torchvision + a small decoder, same input and
output as the parser, so the app can load it in its place). It learns the teacher's answer on the crop as it
looks in hard shots. Crops of projects trained earlier (Cloud\\_trained) are mixed in, so a new project does
not push the earlier ones out; trained projects are archived there, never deleted unless asked.
The student is installed for the app (face_parse_own.onnx) once it agrees with the teacher well enough.

  python step_train.py                 choose a project, steps by its size
  python step_train.py <project> [steps] [--delete|--keep]
"""
import os, sys, json, math, random, time, zlib, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "app")
if os.path.isdir(APP):
    sys.path.insert(0, APP)
import numpy as np, cv2, torch, torch.nn.functional as F

DATA = os.path.join(HERE, "data")
DONE = os.path.join(HERE, "_trained")
MODEL = os.path.join(HERE, "model")
BATCH = int(os.environ.get("DIAP_BATCH", "8"))
ACCUM = int(os.environ.get("DIAP_ACCUM", "1"))
LR = 6e-4
INSTALL_AT = 0.90                # agreement with the teacher (clean crops) from which the app uses the student
SIZE = 512
REPLAY = 0.3                     # share of every batch taken from projects trained before
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FLIP = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
HF = os.environ.get("DIAP_HF", "jonathandinu/face-parsing")
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = random.Random(7)


def imread(path):
    try:
        im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)      # (paths with Korean names)
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def to_t(x):
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() / 255.0
    return (t - MEAN) / STD


def load_items(folder):
    """every crop under one project folder: (full path, sampling weight, meta)"""
    items = []
    for cut in sorted(os.listdir(folder)):
        mf = os.path.join(folder, cut, "meta.json")
        if not os.path.exists(mf):
            continue
        try:
            meta = json.load(open(mf, encoding="utf-8"))["items"]
        except Exception:
            continue
        for m in meta:
            w = 1.0
            if m["lum"] < 0.25 or m["lum"] > 0.78:
                w += 2.0
            if m["vis"] < 0.5:
                w += 1.0
            if m["wf"] < 90:
                w += 1.0
            items.append((os.path.join(folder, cut, m["f"]), w, m))
    return items


def projects():
    out = []
    if os.path.isdir(DATA):
        for n in sorted(os.listdir(DATA)):
            p = os.path.join(DATA, n)
            if os.path.isdir(p):
                k = len(load_items(p))
                if k:
                    out.append((n, k))
    return out


class Student(torch.nn.Module):
    """ResNet-34 encoder (torchvision, ImageNet) + FPN-style decoder. 512 px in, 19 class logits at 128 px out."""

    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        w = None
        if pretrained:
            try:
                w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            except Exception:
                w = None
        try:
            r = torchvision.models.resnet34(weights=w)
        except Exception as e:                       # no network: start from scratch, say so
            print("ImageNet 가중치를 받지 못해 처음부터 학습합니다:", e)
            r = torchvision.models.resnet34(weights=None)
        self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        C = 128
        self.lat = torch.nn.ModuleList([torch.nn.Conv2d(c, C, 1) for c in (64, 128, 256, 512)])
        self.smooth = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(C, C, 3, padding=1), torch.nn.GroupNorm(8, C), torch.nn.ReLU(True)) for _ in range(4)])
        self.head = torch.nn.Sequential(torch.nn.Conv2d(4 * C, 256, 3, padding=1), torch.nn.GroupNorm(8, 256), torch.nn.ReLU(True),
                                        torch.nn.Conv2d(256, 19, 1))

    def forward(self, pixel_values):
        x = self.stem(pixel_values)
        f1 = self.l1(x); f2 = self.l2(f1); f3 = self.l3(f2); f4 = self.l4(f3)
        p = None
        outs = []
        for i, f in zip((3, 2, 1, 0), (f4, f3, f2, f1)):
            q = self.lat[i](f)
            if p is not None:
                q = q + F.interpolate(p, size=q.shape[-2:], mode="bilinear", align_corners=False)
            p = q
            outs.append(F.interpolate(self.smooth[i](q), size=f1.shape[-2:], mode="bilinear", align_corners=False))
        return self.head(torch.cat(outs, 1))


def logits_of(model, x):
    o = model(pixel_values=x)
    return o.logits if hasattr(o, "logits") else o


def norm_view(x):
    """x uint8 RGB -> brightness lifted and contrast evened (CLAHE on L, gamma to a mid grey face)."""
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    med = float(np.median(L[128:384, 128:384])) / 255.0
    g = math.log(0.5) / math.log(max(0.05, min(0.95, med)))
    L = 255.0 * (L / 255.0) ** g
    lab[..., 0] = np.clip(L, 0, 255).astype(np.uint8)
    lab[..., 0] = cv2.createCLAHE(2.0, (4, 4)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def geo_aug(x):
    """small shifts / scale / turn, sometimes the face pushed against the border (half out of the frame)."""
    s = rng.uniform(0.85, 1.2)
    a = rng.uniform(-12, 12)
    tx, ty = rng.uniform(-0.08, 0.08) * SIZE, rng.uniform(-0.08, 0.08) * SIZE
    if rng.random() < 0.25:
        tx += rng.choice([-1, 1]) * rng.uniform(0.15, 0.30) * SIZE
    Mx = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), a, s)
    Mx[:, 2] += (tx, ty)
    return cv2.warpAffine(x, Mx, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def degrade(x):
    """how a hard shot looks: dark, blown out, soft, small, tinted, noisy, compressed."""
    f = x.astype(np.float32) / 255.0
    r = rng.random()
    if r < 0.30:
        f = f ** rng.uniform(1.4, 2.6) * rng.uniform(0.35, 0.8)                 # dark
    elif r < 0.55:
        f = np.clip(f * rng.uniform(1.3, 2.2) + rng.uniform(0.0, 0.15), 0, 1) ** rng.uniform(0.6, 0.9)  # washed out
    if rng.random() < 0.5:
        f = f * np.array([rng.uniform(0.8, 1.2), rng.uniform(0.85, 1.1), rng.uniform(0.8, 1.25)], np.float32)
    f = np.clip(f, 0, 1)
    if rng.random() < 0.6:                                                       # small / soft
        k = rng.uniform(0.18, 0.7)
        sm = cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        f = cv2.resize(sm, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.8, 3.0))
    if rng.random() < 0.5:
        f = f + np.random.normal(0, rng.uniform(0.004, 0.03), f.shape).astype(np.float32)
    u = np.clip(f * 255, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, b = cv2.imencode(".jpg", cv2.cvtColor(u, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 80)])
        u = cv2.cvtColor(cv2.imdecode(b, 1), cv2.COLOR_BGR2RGB)
    return u


def to_t(x):
    t = torch.from_numpy(x).permute(2, 0, 1).float()[None] / 255.0
    return (t - MEAN) / STD


@torch.no_grad()
def teacher_probs(teacher, x):
    """probabilities (19, S/4, S/4) averaged over clean / normalised / mirrored views."""
    views = [x, norm_view(x)]
    acc = 0
    n = 0
    for v in views:
        for flip in (False, True):
            vv = v[:, ::-1].copy() if flip else v
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                lo = teacher(pixel_values=to_t(vv).to(dev)).logits.float()
            p = F.softmax(lo, 1)
            if flip:
                p = p.flip(3)[:, FLIP]
            acc = acc + p
            n += 1
    return (acc / n)[0]


def batch(items, weights, teacher):
    xs, ts = [], []
    while len(xs) < BATCH:
        f, _, _ = rng.choices(items, weights)[0]
        im = imread(f)
        if im is None or im.shape[:2] != (SIZE, SIZE):
            continue
        g = geo_aug(im)
        if rng.random() < 0.5:
            g = g[:, ::-1].copy()
        t = teacher_probs(teacher, g)
        s = degrade(g) if rng.random() < 0.85 else g
        xs.append(to_t(s)[0]); ts.append(t)
    return torch.stack(xs).to(dev), torch.stack(ts)


def loss_fn(logits, tprob):
    """distillation: soft cross-entropy against the teacher on every pixel (surer pixels count more), face
    classes weighted up."""
    conf, _ = tprob.max(1)
    pw = 0.3 + 0.7 * conf
    w = torch.ones(19, device=logits.device)
    w[[1, 2, 17]] = 2.0                          # skin, nose, neck: what the retouch needs
    lp = F.log_softmax(logits, 1)
    ce = -(tprob * lp * w.view(1, 19, 1, 1)).sum(1)
    return (ce * pw).sum() / pw.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, teacher, items, n=60, clean=False):
    """agreement (IoU of skin+neck) of model(degraded) with teacher(clean views), on held-out crops."""
    r = random.Random(99)
    ious = []
    for f, _, m in r.sample(items, min(n, len(items))):
        im = imread(f)
        if im is None:
            continue
        t = teacher_probs(teacher, im)
        tm = (t[[1, 2, 17]].sum(0) > 0.5)
        rng.seed(zlib.crc32(f.encode()))
        d = im if clean else degrade(im)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            lo = logits_of(model, to_t(d).to(dev)).float()
        p = F.softmax(lo, 1)[0]
        pm = (p[[1, 2, 17]].sum(0) > 0.5)
        u = (tm | pm).sum().item()
        ious.append((tm & pm).sum().item() / max(1, u))
    return float(np.mean(ious))


def set_train(model):
    """train mode, but the BatchNorm layers keep the statistics they have (a small batch of degraded crops would
    pull them away from clean faces)."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def export(model, path):
    from transformers import SegformerForSemanticSegmentation
    model = model.float().eval().cpu()

    class W(torch.nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, pixel_values):
            return logits_of(s.m, pixel_values)
    x = torch.randn(1, 3, SIZE, SIZE)
    wm = W(model).eval()
    torch.onnx.export(wm, x, path, input_names=["pixel_values"], output_names=["logits"], opset_version=18,
                      dynamic_axes=None, do_constant_folding=True)
    try:                                  # one single file (the exporter may leave the weights in a .data file)
        import onnx
        m_ = onnx.load(path)
        onnx.save(m_, path, save_as_external_data=False)
        if os.path.exists(path + ".data"):
            os.remove(path + ".data")
    except Exception as e:
        print("single-file save skipped:", e)
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        o = s.run(None, {"pixel_values": x.numpy()})[0]
        with torch.no_grad():
            ref = wm(x).numpy()
        print("onnx check: max diff", float(np.abs(o - ref).max()), "shape", o.shape)
    except Exception as e:
        print("onnx check skipped:", e)


def models_dir():
    try:
        from contour.core.settings import settings
        return os.path.join(str(settings().data_path()), "models", "onnx")
    except Exception:
        return MODEL


def main():
    from transformers import SegformerForSemanticSegmentation
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    prjs = projects()
    if not prjs:
        print("학습할 프로젝트가 없습니다 (Cloud\\data 가 비어 있음). DIAP에서 영상을 분석하면 여기에 쌓입니다."); return 0
    name = args[0] if args else None
    if name is None:
        print("\n수집된 프로젝트:")
        for i, (n, k) in enumerate(prjs, 1):
            print(f"  {i}. {n}   얼굴 {k}장")
        s = input("\n학습할 프로젝트 번호 (Enter = 취소): ").strip()
        if not s.isdigit() or not 1 <= int(s) <= len(prjs):
            print("취소"); return 0
        name = prjs[int(s) - 1][0]
    folder = os.path.join(DATA, name)
    items = load_items(folder)
    if len(items) < 30:
        print(f"{name}: 얼굴 {len(items)}장은 너무 적습니다 (30장 이상 필요)."); return 1
    steps = int(args[1]) if len(args) > 1 else int(min(3000, max(300, len(items) * 1.3)))
    old = []
    if os.path.isdir(DONE):
        for n in sorted(os.listdir(DONE)):
            old += load_items(os.path.join(DONE, n))
    os.makedirs(MODEL, exist_ok=True)
    print(f"프로젝트 {name}: 얼굴 {len(items)}장, 이전 프로젝트 {len(old)}장 섞음, {steps}스텝, {dev}", flush=True)
    random.Random(3).shuffle(items)
    k = max(10, len(items) // 10)
    val, tr = items[:k], items[k:]
    val_old = random.Random(5).sample(old, min(60, len(old))) if old else []
    weights = [w for _, w, _ in tr]
    w_old = [w for _, w, _ in old]
    teacher = SegformerForSemanticSegmentation.from_pretrained(HF).to(dev).eval()
    tuned = os.path.join(MODEL, "face_parse_diap.pt")
    if os.path.exists(tuned):                     # the teacher as adapted to this user's footage, when there is one
        teacher.load_state_dict(torch.load(tuned, map_location=dev))
    for p in teacher.parameters():
        p.requires_grad_(False)
    state = os.path.join(MODEL, "face_parse_own.pt")
    model = Student(pretrained=not os.path.exists(state)).to(dev)
    if os.path.exists(state):
        model.load_state_dict(torch.load(state, map_location=dev))
        print("이전까지 학습된 자체 모델에서 이어서 학습합니다.")
    else:
        print("자체 모델을 새로 시작합니다 (ResNet-34, ImageNet 초기값).")
    model.eval()
    base = evaluate(model, teacher, val, clean=True)
    base_old = evaluate(model, teacher, val_old, clean=True) if val_old else None
    print(f"학습 전 교사와의 일치율: 이 프로젝트 {base:.3f}" + (f", 이전 프로젝트 {base_old:.3f}" if base_old is not None else ""), flush=True)
    model.train()
    enc = [p for n_, p in model.named_parameters() if n_.startswith(("stem", "l1", "l2", "l3", "l4"))]
    dec = [p for n_, p in model.named_parameters() if not n_.startswith(("stem", "l1", "l2", "l3", "l4"))]
    opt = torch.optim.AdamW([dict(params=enc, lr=LR * 0.3), dict(params=dec, lr=LR)], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / max(1, steps)))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    t0 = time.time()
    run = 0.0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(ACCUM):
            if old and rng.random() < REPLAY:
                x, tp = batch(old, w_old, teacher)
            else:
                x, tp = batch(tr, weights, teacher)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                lo = logits_of(model, x)
            l = loss_fn(lo.float(), tp) / ACCUM
            scaler.scale(l).backward()
            tot += float(l.detach())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.95 * run + 0.05 * tot if step else tot
        if step % 25 == 0:
            el = time.time() - t0
            print(f"\r스텝 {step}/{steps}  loss {run:.4f}  {el / 60:.1f}분 경과  남은 시간 약 {el / max(1, step) * (steps - step) / 60:.0f}분   ", end="", flush=True)
    print()
    model.eval()
    after = evaluate(model, teacher, val, clean=True)
    hard = evaluate(model, teacher, val)
    after_old = evaluate(model, teacher, val_old, clean=True) if val_old else None
    print(f"학습 후 교사와의 일치율: 이 프로젝트 {after:.3f} (전 {base:.3f}), 어렵게 만든 화면 {hard:.3f}"
          + (f", 이전 프로젝트 {after_old:.3f} (전 {base_old:.3f})" if after_old is not None else ""), flush=True)
    hist_f = os.path.join(HERE, "history.json")
    hist = json.load(open(hist_f, encoding="utf-8")) if os.path.exists(hist_f) else []
    ok = after >= base - 0.003 and (after_old is None or after_old >= base_old - 0.02)
    use = ok and after >= INSTALL_AT and (after_old is None or after_old >= INSTALL_AT - 0.03)
    rec = dict(project=name, time=time.strftime("%Y-%m-%d %H:%M:%S"), crops=len(items), steps=steps, model="own", before=round(base, 4),
               after=round(after, 4), hard=round(hard, 4), before_old=None if base_old is None else round(base_old, 4),
               after_old=None if after_old is None else round(after_old, 4), kept=ok, installed=use)
    hist.append(rec)
    json.dump(hist, open(hist_f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not ok:
        print("좋아지지 않아 모델을 바꾸지 않았습니다. 데이터는 그대로 둡니다."); return 2
    torch.save(model.state_dict(), state + ".tmp")
    os.replace(state + ".tmp", state)
    tmp = os.path.join(MODEL, "face_parse_own.onnx")
    export(model, tmp)
    if use:
        md = models_dir()
        os.makedirs(md, exist_ok=True)
        shutil.copyfile(tmp, os.path.join(md, "face_parse_own.onnx"))
        print(f"자체 모델을 앱에 설치했습니다 (일치율 {after:.3f} >= {INSTALL_AT}):", os.path.join(md, "face_parse_own.onnx"))
    else:
        print(f"자체 모델은 저장했지만 아직 앱에 넣지 않았습니다 (일치율 {after:.3f} < {INSTALL_AT}). 프로젝트가 더 쌓이면 올라갑니다.")
    how = "delete" if "--delete" in flags else "keep" if "--keep" in flags else None
    if how is None:
        s = input("학습한 데이터: [K] 보관 (권장: 다음 학습에 섞여 이전 프로젝트를 잊지 않게 하고, 나중에 새 모델을 처음부터 만들 때 필요) / [D] 삭제 : ").strip().lower()
        how = "delete" if s == "d" else "keep"
    if how == "delete":
        shutil.rmtree(folder, ignore_errors=True)
        print("삭제했습니다:", folder)
    else:
        os.makedirs(DONE, exist_ok=True)
        dst = os.path.join(DONE, name)
        if os.path.exists(dst):
            dst = dst + time.strftime("_%Y%m%d_%H%M%S")
        shutil.move(folder, dst)
        print("보관했습니다:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
