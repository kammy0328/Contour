"""Model packages: the publisher's tool (instruction 2026-11-20 §4).

    python tools/modelpkg.py keygen --out DIR [--id NAME]
    python tools/modelpkg.py pack --registry DIR --key KEY --version VER --from PATH [--main FILE] [...]
    python tools/modelpkg.py pack-installed --registry DIR --version VER [--keys K,K]
    python tools/modelpkg.py sign --registry DIR --secret FILE [--id NAME]
    python tools/modelpkg.py verify --registry DIR [--pub HEX --id NAME]
    python tools/modelpkg.py zip --registry DIR --out FILE [--keys K,K]
    python tools/modelpkg.py install --from DIR|ZIP|URL [--keys K,K] [--pub HEX --id NAME]
    python tools/modelpkg.py list

A registry is a folder (or the same layout on any plain file host):

    manifest.json   manifest.sig   <key>/<version>/<files...>

- keygen makes the publishing key (Ed25519). The .secret file stays with the publisher and is never put in the
  repository or on the file host; the line it prints goes into contour/track/modelpkg.py TRUSTED, and from the
  release that carries it the app takes lists signed with that key.
- pack copies a model's files into the registry and writes them into the list with their sizes and SHA-256, the
  inputs and outputs read from the ONNX file, and the app / runtime versions and backends it is for. A version that
  is in the list already is never written again with other content (a version never changes): give a new one.
- sign writes manifest.sig for the list as it is. Packing after signing needs signing again.
- zip makes one file of the list, its signature and the chosen packages, for a computer that is never online
  (설정 > 모델 관리 > 패키지 가져오기, or `install --from that.zip`).

Nothing here uploads: putting the registry folder on a file host is the publisher's own step, with the host's own
tool (the storage provider is not decided)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour.core import ed25519 as ED  # noqa: E402
from contour.track import modelpkg as MP  # noqa: E402


def say(*a):
    print(*a, flush=True)


def _read_list(reg: Path) -> list:
    f = reg / "manifest.json"
    if not f.is_file():
        return []
    return MP.parse_manifest(f.read_bytes())


def _write_list(reg: Path, pkgs: list) -> None:
    reg.mkdir(parents=True, exist_ok=True)
    tmp = reg / "manifest.json.tmp"
    tmp.write_bytes(MP.make_manifest(sorted(pkgs, key=lambda p: (p.key, MP._ver(p.version), p.version))))
    os.replace(tmp, reg / "manifest.json")
    sig = reg / "manifest.sig"
    if sig.exists():
        sig.unlink()                  # the signature was of the list before this change
        say("  (목록이 바뀌어 이전 서명을 지웠습니다: sign을 다시 실행하세요)")


def onnx_io(path: Path) -> dict:
    """Names, types and shapes of a model's inputs and outputs, read from the file's header (the weights are not
    loaded). {} when neither the onnx package nor onnxruntime can read it."""
    def dims(t):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]
    try:
        import onnx
        m = onnx.load(str(path), load_external_data=False)
        init = {i.name for i in m.graph.initializer}
        T = onnx.TensorProto.DataType.Name
        return dict(inputs=[dict(name=i.name, type=T(i.type.tensor_type.elem_type).lower(), shape=dims(i.type.tensor_type))
                            for i in m.graph.input if i.name not in init],
                    outputs=[dict(name=o.name, type=T(o.type.tensor_type.elem_type).lower(), shape=dims(o.type.tensor_type))
                             for o in m.graph.output],
                    opset=max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=0))
    except Exception:
        pass
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return dict(inputs=[dict(name=i.name, type=i.type, shape=list(i.shape)) for i in s.get_inputs()],
                    outputs=[dict(name=o.name, type=o.type, shape=list(o.shape)) for o in s.get_outputs()])
    except Exception:
        return {}


def _io_for(key: str, folder: Path, main: str) -> dict:
    from contour.track import models as M
    spec = M.MODELS.get(key, {})
    io = {}
    if spec.get("input"):
        io["input"] = list(spec["input"])
    if spec.get("kpts"):
        io["kpts"] = int(spec["kpts"])
    graphs = [main] if main else sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.onnx"))
    files = {}
    for g in graphs:
        d = onnx_io(folder / g)
        if d:
            files[g] = d
    if files:
        io["graphs"] = files
    return io


def pack(reg: Path, key: str, version: str, src: Path, main: str = "", label: str = "", app_min: str = "",
         app_max: str = "", ort_min: str = "", ort_max: str = "", backends=(), io: dict = None) -> "MP.Package":
    from contour.track import models as M
    src = Path(src)
    if not src.exists():
        raise SystemExit(f"없는 경로: {src}")
    dst = reg / key / version
    have = [p for p in _read_list(reg) if (p.key, p.version) == (key, version)]
    stage = reg / key / f".pack-{version}-{os.getpid()}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        if src.is_file():
            shutil.copyfile(src, stage / src.name)
            main = main or src.name
        else:
            for f in sorted(p for p in src.rglob("*") if p.is_file()):
                n = f.name
                # not part of a model: unfinished downloads, files set aside, the app's own notes
                if n.endswith((".part", ".tmp")) or ".broken" in n or ".noaccess" in n or n in ("package.json",):
                    continue
                out = stage / f.relative_to(src)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, out)
        more = dict(main=main, label=label or M.MODELS.get(key, {}).get("label", ""))
        if app_min or app_max:
            more["app"] = {k: v for k, v in (("min", app_min), ("max", app_max)) if v}
        if ort_min or ort_max:
            more["runtime"] = {"onnxruntime": {k: v for k, v in (("min", ort_min), ("max", ort_max)) if v}}
        if backends:
            more["backends"] = list(backends)
        more["io"] = io if io is not None else _io_for(key, stage, main)
        pkg = MP.describe(stage, key, version, **{k: v for k, v in more.items() if v})
        if have:
            if have[0].sha256 != pkg.sha256:
                raise SystemExit(f"{key} {version}은(는) 목록에 다른 내용으로 이미 있습니다. 버전은 내용이 바뀌지 않습니다: "
                                 "새 버전 이름으로 올리세요.")
            say(f"  [있음] {key} {version}")
            return have[0]
        if dst.exists():
            shutil.rmtree(dst)
        os.rename(stage, dst)
        _write_list(reg, [p for p in _read_list(reg)] + [pkg])
        say(f"  [추가] {key} {version}: 파일 {len(pkg.files)}개, {pkg.size / 1e6:.1f} MB, sha256 {pkg.sha256[:16]}…")
        return pkg
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def cmd_keygen(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kid = a.id
    sec_f, pub_f = out / f"{kid}.secret", out / f"{kid}.pub"
    if sec_f.exists():
        raise SystemExit(f"이미 있습니다 (덮어쓰지 않습니다): {sec_f}")
    seed = os.urandom(32)
    pub = ED.public_key(seed).hex()
    fd = os.open(str(sec_f), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(seed.hex() + "\n")
    pub_f.write_text(pub + "\n", encoding="utf-8")
    say(f"비밀 키: {sec_f}  (배포자만 보관: 저장소·파일 호스트·메일에 올리지 않습니다. 잃으면 새 키와 새 앱 릴리스가 필요합니다)")
    say(f"공개 키: {pub_f}")
    say("contour/track/modelpkg.py 의 TRUSTED 에 다음 줄을 넣은 릴리스부터 이 키로 서명한 목록을 받습니다:")
    say(f'    "{kid}": "{pub}",')
    return 0


def cmd_pack(a) -> int:
    io = json.loads(Path(a.io).read_text(encoding="utf-8")) if a.io else None
    pack(Path(a.registry), a.key, a.version, Path(getattr(a, "from")), a.main, a.label, a.app_min, a.app_max,
         a.ort_min, a.ort_max, [b for b in a.backends.split(",") if b], io)
    return 0


def cmd_pack_installed(a) -> int:
    """The models this computer has (downloaded by the app) as packages of one version: the first publication."""
    from contour.track import models as M
    keys = [k for k in a.keys.split(",") if k] if a.keys else list(M.MODELS)
    n = 0
    for k in keys:
        if k not in M.MODELS:
            raise SystemExit(f"모르는 모델 키: {k}")
        p = M.legacy_path(k)
        if not p.exists():
            say(f"  [없음] {k}: 이 컴퓨터에 없어 건너뜁니다")
            continue
        pack(Path(a.registry), k, a.version, p, "", "", a.app_min, "", a.ort_min, "",
             [b for b in a.backends.split(",") if b])
        n += 1
    say(f"{n}개 모델을 목록에 넣었습니다: {Path(a.registry) / 'manifest.json'}")
    return 0


def cmd_sign(a) -> int:
    reg = Path(a.registry)
    data = (reg / "manifest.json").read_bytes()
    MP.parse_manifest(data)
    seed = bytes.fromhex(Path(a.secret).read_text(encoding="utf-8").strip())
    kid = a.id or Path(a.secret).stem
    (reg / "manifest.sig").write_bytes(MP.sign_manifest(data, seed, kid))
    say(f"서명함: {reg / 'manifest.sig'} (키 {kid}, 공개 키 {ED.public_key(seed).hex()})")
    return 0


def _trusted(a):
    if getattr(a, "pub", ""):
        return {a.id or "cli": a.pub.strip()}
    return None


def cmd_verify(a) -> int:
    reg = Path(a.registry)
    try:
        data, sig = MP.DirRegistry(reg).manifest()
        pkgs = MP.verify_manifest(data, sig, _trusted(a))
    except MP.PackageError as e:
        say(f"실패 ({e.why}): {e}")
        return 1
    bad = 0
    for p in pkgs:
        ok = MP.describe(reg / p.key / p.version, p.key, p.version).sha256 == p.sha256 \
            if (reg / p.key / p.version).is_dir() else None
        say(f"  {'[맞음]' if ok else '[파일 없음]' if ok is None else '[다름]'} {p.key} {p.version} "
            f"{p.size / 1e6:.1f} MB")
        bad += ok is False
    say("서명과 목록이 맞습니다" if not bad else f"목록과 다른 패키지 {bad}개")
    return 1 if bad else 0


def cmd_zip(a) -> int:
    reg = Path(a.registry)
    pkgs = _read_list(reg)
    if not (reg / "manifest.sig").is_file():
        raise SystemExit("서명이 없습니다: sign을 먼저 실행하세요")
    keys = set(k for k in a.keys.split(",") if k) if a.keys else {p.key for p in pkgs}
    newest = {}
    for p in pkgs:
        if p.key in keys and (p.key not in newest or MP._ver(p.version) > MP._ver(newest[p.key].version)):
            newest[p.key] = p
    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        z.write(reg / "manifest.json", "manifest.json")
        z.write(reg / "manifest.sig", "manifest.sig")
        for p in newest.values():
            for f in p.files:
                z.write(reg / p.key / p.version / f.path, f"{p.key}/{p.version}/{f.path}")
    say(f"{out}: 패키지 {len(newest)}개, {out.stat().st_size / 1e6:.1f} MB")
    return 0


def cmd_install(a) -> int:
    last = [-1]

    def prog(f, m):
        pct = int(f * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r  {pct:3d}% {m}      ", end="", flush=True)
    try:
        res = MP.bring_in(getattr(a, "from"), [k for k in a.keys.split(",") if k] if a.keys else None, prog,
                          trusted=_trusted(a))
    except MP.PackageError as e:
        say(f"\n실패 ({e.why}): {e}")
        return 1
    print()
    for k, v in res["installed"]:
        say(f"  [설치] {k} {v}")
    for k, v in res["kept"]:
        say(f"  [있음] {k} {v}")
    for k, t in res["failed"].items():
        say(f"  [실패] {k}: {t}")
    return 1 if res["failed"] else 0


def cmd_list(a) -> int:
    for r in MP.listing():
        src = {"package": "패키지", "app": "앱 다운로드", "none": "없음"}[r["source"]]
        say(f"  {r['key']:<18} {src:<8} {r['version'] or '-':<14} {r['size'] / 1e6:8.0f} MB  {r['label']}"
            + (f"  (설치된 버전 {', '.join(r['versions'])})" if len(r["versions"]) > 1 else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python tools/modelpkg.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("keygen")
    p.add_argument("--out", required=True)
    p.add_argument("--id", default="diap-models-1")
    p.set_defaults(fn=cmd_keygen)
    p = sub.add_parser("pack")
    p.add_argument("--registry", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--from", required=True)
    p.add_argument("--main", default="")
    p.add_argument("--label", default="")
    p.add_argument("--app-min", default="")
    p.add_argument("--app-max", default="")
    p.add_argument("--ort-min", default="")
    p.add_argument("--ort-max", default="")
    p.add_argument("--backends", default="")
    p.add_argument("--io", default="", help="입력·출력 규격 JSON 파일 (없으면 ONNX 파일에서 읽음)")
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("pack-installed")
    p.add_argument("--registry", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--keys", default="")
    p.add_argument("--app-min", default="")
    p.add_argument("--ort-min", default="")
    p.add_argument("--backends", default="")
    p.set_defaults(fn=cmd_pack_installed)
    p = sub.add_parser("sign")
    p.add_argument("--registry", required=True)
    p.add_argument("--secret", required=True)
    p.add_argument("--id", default="")
    p.set_defaults(fn=cmd_sign)
    p = sub.add_parser("verify")
    p.add_argument("--registry", required=True)
    p.add_argument("--pub", default="")
    p.add_argument("--id", default="")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("zip")
    p.add_argument("--registry", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keys", default="")
    p.set_defaults(fn=cmd_zip)
    p = sub.add_parser("install")
    p.add_argument("--from", required=True)
    p.add_argument("--keys", default="")
    p.add_argument("--pub", default="")
    p.add_argument("--id", default="")
    p.set_defaults(fn=cmd_install)
    p = sub.add_parser("list")
    p.set_defaults(fn=cmd_list)
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
