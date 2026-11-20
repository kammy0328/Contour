"""Model packages: models handed out as files with a signed list, installed whole or not at all, and used by the
version a piece of work began with (instruction 2026-11-20 §4).

What comes from outside is the list (manifest.json), its signature (manifest.sig) and the files the list names;
nothing is ever sent (a registry is read with GET only: no footage, frame, crop, mask or project file has a way
out through here). Reading originals, proxies, inference, analysis, retouch and render all stay on this computer.

- The list names each package by its key (the app's own name for the model, as in models.MODELS) and a version that
  never changes its content, the app and runtime versions it fits, its inputs and outputs, the backends it runs on,
  and for every file its size and SHA-256, with one SHA-256 over all of them. A model whose weights are in a second
  file is one package: all its files or none.
- The list is taken only with a signature by a key the app trusts (Ed25519), and only the files it names are
  installed, each checked against its size and hash while it is written.
- A package is written beside the ones in use and put in use by one rename when every file is checked; cut short,
  with a wrong hash or a full disk, what was in use stays in use. Two installs of one package do not run together.
- Work that has begun keeps the versions it began with (pinned), also when a newer one is put in use meanwhile; a
  cut remembers the versions it was analysed with, and the keys of results kept in memory carry them.
- With the packages installed nothing here opens a connection. A package can be brought in from a folder or a zip
  with the same checks (a computer that is never online).

Until a list is published there is none: models are found where they always were (models.model_path) and every
function here says 'not installed' for them. The storage provider is not decided; HttpRegistry reads the layout
below from any plain file host, DirRegistry from a folder (the tests' registry, and the offline import).

    manifest.json   manifest.sig   <key>/<version>/<file path> ..."""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from ..core import ed25519

SCHEMA = 1
# key id -> Ed25519 public key (hex) of who may publish model lists for this app. Empty: no list is taken (the
# publishing key is made and kept by the publisher: tools/modelpkg.py keygen). Tests pass their own.
TRUSTED: Dict[str, str] = {}
BACKENDS = ("cuda", "coreml", "dml", "cpu")
LOCK_STALE = 120.0          # s without a sign of life: the install that held the lock is gone
LOCK_BEAT = 2.0             # s between signs of life while files are written
SPACE_MARGIN = 64 << 20     # bytes kept free beyond the package's own size
CHUNK = 1 << 20
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_VER_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,63}$")


class PackageError(Exception):
    """why: signature | list | compat | missing | size | hash | interrupted | space | busy | cancelled | io"""

    def __init__(self, why: str, text: str):
        super().__init__(text)
        self.why = why


# ------------------------------------------------------------------------------------------------- the list
@dataclass(frozen=True)
class FileSpec:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Package:
    key: str
    version: str
    files: Tuple[FileSpec, ...]
    sha256: str                               # over all files (digest())
    main: str = ""                            # the file a single-file model is opened by ('' = the package's folder)
    app_min: str = ""
    app_max: str = ""
    ort_min: str = ""
    ort_max: str = ""
    backends: Tuple[str, ...] = ()
    io: dict = field(default_factory=dict, compare=False, hash=False)
    label: str = ""

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    def digest(self) -> str:
        return files_digest(self.files)

    def fits(self, app_version: str = "", ort_version: str = "") -> str:
        """'' when this app and runtime can use the package, else why not."""
        if app_version:
            if self.app_min and _ver(app_version) < _ver(self.app_min):
                return f"DIAP {self.app_min} 이상 필요 (지금 {app_version})"
            if self.app_max and _ver(app_version) > _ver(self.app_max):
                return f"DIAP {self.app_max} 이하에서만 사용 (지금 {app_version})"
        if ort_version:
            if self.ort_min and _ver(ort_version) < _ver(self.ort_min):
                return f"ONNX Runtime {self.ort_min} 이상 필요 (지금 {ort_version})"
            if self.ort_max and _ver(ort_version) > _ver(self.ort_max):
                return f"ONNX Runtime {self.ort_max} 이하에서만 사용 (지금 {ort_version})"
        return ""

    def to_dict(self) -> dict:
        d = dict(key=self.key, version=self.version, sha256=self.sha256,
                 files=[dict(path=f.path, size=f.size, sha256=f.sha256) for f in self.files])
        for k in ("main", "label"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.app_min or self.app_max:
            d["app"] = {k: v for k, v in (("min", self.app_min), ("max", self.app_max)) if v}
        if self.ort_min or self.ort_max:
            d["runtime"] = {"onnxruntime": {k: v for k, v in (("min", self.ort_min), ("max", self.ort_max)) if v}}
        if self.backends:
            d["backends"] = list(self.backends)
        if self.io:
            d["io"] = self.io
        return d


def io_mismatch(pkg: Package) -> str:
    """'' when the package's inputs and outputs (the list's io) are what this app feeds the model of that key and
    reads from it, else why not: a model with another input size or number of points is not put in use under a key
    whose code was written for the other."""
    try:
        from . import models as M
        spec = M.MODELS.get(pkg.key) or {}
    except Exception:
        return ""
    io = pkg.io or {}
    want = spec.get("input")
    got = io.get("input")
    if want and got and [int(v) for v in got] != [int(v) for v in want]:
        return f"입력 크기 {list(got)} (이 버전의 DIAP은 {list(want)})"
    if spec.get("kpts") and io.get("kpts") and int(io["kpts"]) != int(spec["kpts"]):
        return f"출력 점 {io['kpts']}개 (이 버전의 DIAP은 {spec['kpts']}개)"
    return ""


def _ver(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.+-]", str(v))[:4])


def files_digest(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda f: f.path):
        h.update(f"{f.path}\n{f.size}\n{f.sha256}\n".encode("utf-8"))
    return h.hexdigest()


def _safe_path(p) -> str:
    """A file's place inside its package as the list may name it: forward slashes, no way out of the folder."""
    if not isinstance(p, str) or not p or len(p) > 240:
        raise PackageError("list", "파일 경로가 올바르지 않습니다")
    parts = p.split("/")
    if p.startswith("/") or "\\" in p or ":" in p or any(x in ("", ".", "..") for x in parts) or "\x00" in p:
        raise PackageError("list", f"파일 경로가 올바르지 않습니다: {p!r}")
    if any(x.startswith(".stage") or x in ("package.json", "active.json", ".lock") for x in parts[:1]):
        raise PackageError("list", f"쓸 수 없는 파일 이름입니다: {p!r}")
    return p


def _hex64(s) -> str:
    if not isinstance(s, str) or not re.fullmatch(r"[0-9a-f]{64}", s):
        raise PackageError("list", "SHA-256 값이 올바르지 않습니다")
    return s


def package_from(d: dict) -> Package:
    try:
        key, version = d["key"], d["version"]
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise PackageError("list", f"모델 키가 올바르지 않습니다: {key!r}")
        if not isinstance(version, str) or not _VER_RE.match(version):
            raise PackageError("list", f"{key}: 버전이 올바르지 않습니다: {version!r}")
        files = []
        seen = set()
        for f in d["files"]:
            path = _safe_path(f["path"])
            if path.lower() in seen:
                raise PackageError("list", f"{key}: 같은 파일이 두 번 있습니다: {path}")
            seen.add(path.lower())
            size = f["size"]
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise PackageError("list", f"{key}: 파일 크기가 올바르지 않습니다: {path}")
            files.append(FileSpec(path, size, _hex64(f["sha256"])))
        if not files:
            raise PackageError("list", f"{key}: 파일이 없습니다")
        app, rt = d.get("app") or {}, (d.get("runtime") or {}).get("onnxruntime") or {}
        backends = tuple(d.get("backends") or ())
        if any(b not in BACKENDS for b in backends):
            raise PackageError("list", f"{key}: 알 수 없는 백엔드 {backends}")
        main = d.get("main") or ""
        if main and main not in [f.path for f in files]:
            raise PackageError("list", f"{key}: main 파일이 목록에 없습니다")
        p = Package(key=key, version=version, files=tuple(files), sha256=_hex64(d["sha256"]), main=main,
                    app_min=str(app.get("min") or ""), app_max=str(app.get("max") or ""),
                    ort_min=str(rt.get("min") or ""), ort_max=str(rt.get("max") or ""), backends=backends,
                    io=dict(d.get("io") or {}), label=str(d.get("label") or ""))
    except PackageError:
        raise
    except Exception as e:
        raise PackageError("list", f"모델 목록의 항목을 읽지 못했습니다: {e}") from e
    if p.digest() != p.sha256:
        raise PackageError("list", f"{p.key} {p.version}: 전체 SHA-256이 파일 목록과 맞지 않습니다")
    return p


def parse_manifest(data: bytes) -> List[Package]:
    try:
        d = json.loads(data.decode("utf-8"))
    except Exception as e:
        raise PackageError("list", f"모델 목록을 읽지 못했습니다: {e}") from e
    if not isinstance(d, dict) or d.get("schema") != SCHEMA or not isinstance(d.get("packages"), list):
        raise PackageError("list", "모델 목록의 형식이 이 버전의 DIAP과 맞지 않습니다")
    out = [package_from(x) for x in d["packages"]]
    ids = [(p.key, p.version) for p in out]
    if len(set(ids)) != len(ids):
        raise PackageError("list", "모델 목록에 같은 키와 버전이 두 번 있습니다")
    return out


def verify_manifest(data: bytes, sig: bytes, trusted: Optional[Dict[str, str]] = None) -> List[Package]:
    """The packages of a list signed by a trusted key. The signature is of the list's exact bytes."""
    keys = TRUSTED if trusted is None else trusted
    try:
        s = json.loads(sig.decode("utf-8"))
        kid, alg, raw = s["key"], s.get("alg", "ed25519"), base64.b64decode(s["sig"], validate=True)
    except Exception as e:
        raise PackageError("signature", "모델 목록의 서명을 읽지 못했습니다") from e
    pub = keys.get(kid)
    if alg != "ed25519" or pub is None:
        raise PackageError("signature", "모델 목록이 이 앱이 신뢰하는 키로 서명되지 않았습니다")
    try:
        pub_b = bytes.fromhex(pub)
    except ValueError:
        pub_b = b""
    if not ed25519.verify(pub_b, data, raw):
        raise PackageError("signature", "모델 목록의 서명이 맞지 않습니다 (바뀌었거나 손상된 목록)")
    return parse_manifest(data)


def make_manifest(packages: List[Package]) -> bytes:
    return json.dumps(dict(schema=SCHEMA, packages=[p.to_dict() for p in packages]), ensure_ascii=False,
                      indent=1, sort_keys=True).encode("utf-8")


def sign_manifest(data: bytes, secret: bytes, key_id: str) -> bytes:
    """manifest.sig for these exact bytes (the publishing tool and the tests)."""
    return json.dumps(dict(key=key_id, alg="ed25519",
                           sig=base64.b64encode(ed25519.sign(secret, data)).decode("ascii"))).encode("utf-8")


def describe(folder: Path, key: str, version: str, **more) -> Package:
    """The package a folder of files is (sizes and hashes measured): for the publishing tool and the tests."""
    folder = Path(folder)
    files = []
    for f in sorted(p for p in folder.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(CHUNK), b""):
                h.update(b)
        files.append(FileSpec(f.relative_to(folder).as_posix(), f.stat().st_size, h.hexdigest()))
    return package_from(dict(key=key, version=version, sha256=files_digest(files),
                             files=[dict(path=f.path, size=f.size, sha256=f.sha256) for f in files], **more))


# ------------------------------------------------------------------------------------------------- registries
class Registry:
    """Where a list and its files are read from. Reading only."""

    def manifest(self) -> Tuple[bytes, bytes]:
        raise NotImplementedError

    @contextlib.contextmanager
    def open(self, pkg: Package, f: FileSpec):
        raise NotImplementedError
        yield                                                    # pragma: no cover


class DirRegistry(Registry):
    """A folder with the layout of the module text (the tests' registry; a package brought in by hand)."""

    def __init__(self, root):
        self.root = Path(root)

    def manifest(self) -> Tuple[bytes, bytes]:
        try:
            return (self.root / "manifest.json").read_bytes(), (self.root / "manifest.sig").read_bytes()
        except OSError as e:
            raise PackageError("missing", f"모델 목록 파일이 없습니다: {self.root}") from e

    @contextlib.contextmanager
    def open(self, pkg: Package, f: FileSpec):
        p = self.root / pkg.key / pkg.version / f.path
        try:
            fh = open(p, "rb")
        except OSError as e:
            raise PackageError("missing", f"{pkg.key} {pkg.version}: 파일이 없습니다: {f.path}") from e
        try:
            yield fh
        finally:
            fh.close()


class HttpRegistry(Registry):
    """The same layout under a base URL on any plain file host. GET only: nothing is sent but the request for a
    file the list names."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def _get(self, rel: str):
        from urllib.parse import quote
        req = urllib.request.Request(self.base + quote(rel), headers={"User-Agent": "DIAP-models/1"}, method="GET")
        return urllib.request.urlopen(req, timeout=self.timeout)

    def manifest(self) -> Tuple[bytes, bytes]:
        try:
            with self._get("manifest.json") as r:
                data = r.read()
            with self._get("manifest.sig") as r:
                sig = r.read()
        except Exception as e:
            raise PackageError("io", f"모델 목록을 받지 못했습니다: {e}") from e
        return data, sig

    @contextlib.contextmanager
    def open(self, pkg: Package, f: FileSpec):
        try:
            r = self._get(f"{pkg.key}/{pkg.version}/{f.path}")
        except Exception as e:
            raise PackageError("io", f"{pkg.key} {pkg.version}: 받지 못했습니다: {f.path} ({e})") from e
        try:
            yield r
        finally:
            r.close()


@contextlib.contextmanager
def zip_registry(path) -> Iterator[DirRegistry]:
    """A package file (.zip of the registry layout) as a registry, unpacked beside itself for the time of the
    import. Its members are read by name only when they stay inside the folder."""
    import tempfile
    path = Path(path)
    tmp = Path(tempfile.mkdtemp(prefix=".modelpkg_", dir=str(path.parent)))
    try:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith("/"):
                    continue
                parts = n.split("/")
                if n.startswith("/") or "\\" in n or ":" in n or any(x in ("", ".", "..") for x in parts):
                    raise PackageError("list", f"패키지 파일 안의 경로가 올바르지 않습니다: {n!r}")
                dst = tmp.joinpath(*parts)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
        yield DirRegistry(tmp)
    except zipfile.BadZipFile as e:
        raise PackageError("list", "패키지 파일을 열지 못했습니다 (zip이 아니거나 손상됨)") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------- what is installed
class Store:
    """The packages on this computer: <root>/<key>/<version>/ (a folder that exists is complete: it is renamed into
    place after its last check) and <root>/<key>/active.json (which version is in use, and the ones before it)."""

    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ reading
    def _kdir(self, key: str) -> Path:
        if not _KEY_RE.match(key):
            raise PackageError("list", f"모델 키가 올바르지 않습니다: {key!r}")
        return self.root / key

    def versions(self, key: str) -> List[str]:
        d = self._kdir(key)
        try:
            return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")
                          and (p / "package.json").is_file())
        except OSError:
            return []

    def _read_active(self, key: str) -> dict:
        try:
            return json.loads((self._kdir(key) / "active.json").read_text(encoding="utf-8"))
        except Exception:
            return {}

    def active(self, key: str) -> Optional[str]:
        """The version in use, or None. A record that points at a version that is not there (its folder taken
        away) falls back to the newest of the ones before it that is."""
        a = self._read_active(key)
        have = set(self.versions(key))
        for v in [a.get("version")] + list(reversed(a.get("previous") or [])):
            if v and v in have:
                return v
        return None

    def active_versions(self) -> Dict[str, str]:
        out = {}
        try:
            keys = sorted(p.name for p in self.root.iterdir() if p.is_dir() and _KEY_RE.match(p.name))
        except OSError:
            return out
        for k in keys:
            v = self.active(k)
            if v:
                out[k] = v
        return out

    def package(self, key: str, version: Optional[str] = None) -> Optional[Package]:
        v = version or self.active(key)
        if not v:
            return None
        try:
            return package_from(json.loads((self._kdir(key) / v / "package.json").read_text(encoding="utf-8")))
        except Exception:
            return None

    def folder(self, key: str, version: Optional[str] = None) -> Optional[Path]:
        v = version or self.active(key)
        if not v or v not in self.versions(key):
            return None
        return self._kdir(key) / v

    def path(self, key: str, version: Optional[str] = None) -> Optional[Path]:
        """What the model is opened by: its main file, or its folder (several files and no main)."""
        d = self.folder(key, version)
        if d is None:
            return None
        p = self.package(key, d.name)
        if p is None:
            return None
        return d / p.main if p.main else d

    def verify(self, key: str, version: Optional[str] = None) -> bool:
        """Every file of an installed version is still what the list said (sizes and hashes read again)."""
        d, p = self.folder(key, version), self.package(key, version)
        if d is None or p is None:
            return False
        for f in p.files:
            fp = d / f.path
            try:
                if fp.stat().st_size != f.size:
                    return False
                h = hashlib.sha256()
                with open(fp, "rb") as fh:
                    for b in iter(lambda: fh.read(CHUNK), b""):
                        h.update(b)
            except OSError:
                return False
            if h.hexdigest() != f.sha256:
                return False
        return True

    # ------------------------------------------------------------------ one install at a time for a package
    @contextlib.contextmanager
    def _held(self, key: str, wait: float):
        d = self._kdir(key)
        d.mkdir(parents=True, exist_ok=True)
        lock = d / ".lock"
        t0 = time.time()
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()}".encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > LOCK_STALE:
                        lock.unlink()                       # the install that held it is gone
                        continue
                except OSError:
                    continue
                if time.time() - t0 >= wait:
                    raise PackageError("busy", f"{key}: 다른 설치가 진행 중입니다")
                time.sleep(0.05)
        beat = dict(t=time.time())

        def alive():
            if time.time() - beat["t"] >= LOCK_BEAT:
                beat["t"] = time.time()
                try:
                    os.utime(lock, None)
                except OSError:
                    pass
        try:
            yield alive
        finally:
            try:
                lock.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------ installing
    def install(self, reg: Registry, key: str, version: Optional[str] = None,
                progress: Optional[Callable[[float, str], None]] = None,
                cancel: Optional[Callable[[], bool]] = None, trusted: Optional[Dict[str, str]] = None,
                app_version: Optional[str] = None, ort_version: Optional[str] = None, wait: float = 600.0,
                activate: bool = True) -> Package:
        """The package from the registry, checked and put in use. version None: the newest this app can use.
        Raises PackageError; whatever happens, what was in use before is in use after unless this succeeded."""
        data, sig = reg.manifest()
        pkgs = [p for p in verify_manifest(data, sig, trusted) if p.key == key]
        if app_version is None:
            from ..core.settings import APP_VERSION as app_version
        if ort_version is None:
            ort_version = _ort_version()
        if version is not None:
            pkgs = [p for p in pkgs if p.version == version]
            if not pkgs:
                raise PackageError("missing", f"{key} {version}: 모델 목록에 없습니다")
            why = pkgs[0].fits(app_version, ort_version) or io_mismatch(pkgs[0])
            if why:
                raise PackageError("compat", f"{key} {version}: {why}")
            pkg = pkgs[0]
        else:
            ok = [p for p in pkgs if not (p.fits(app_version, ort_version) or io_mismatch(p))]
            if not ok:
                why = (pkgs[-1].fits(app_version, ort_version) or io_mismatch(pkgs[-1])) if pkgs else "모델 목록에 없습니다"
                raise PackageError("compat" if pkgs else "missing", f"{key}: {why}")
            pkg = sorted(ok, key=lambda p: _ver(p.version))[-1]
        with self._held(key, wait) as alive:
            if pkg.version in self.versions(key):
                have = self.package(key, pkg.version)
                if have is None or have.sha256 != pkg.sha256:
                    raise PackageError("hash", f"{key} {pkg.version}: 설치된 같은 버전의 내용이 목록과 다릅니다")
            else:
                self._fetch(reg, pkg, progress, cancel, alive)
            if activate:
                self._activate(key, pkg.version)
        return pkg

    def _fetch(self, reg: Registry, pkg: Package, progress, cancel, alive) -> None:
        kd = self._kdir(pkg.key)
        try:
            free = shutil.disk_usage(kd).free
        except OSError:
            free = None
        if free is not None and free < pkg.size + SPACE_MARGIN:
            raise PackageError("space", f"{pkg.key}: 디스크 공간이 부족합니다 ({pkg.size / 1e6:.0f} MB 필요, "
                                        f"{free / 1e6:.0f} MB 남음)")
        stage = kd / f".stage-{pkg.version}-{os.getpid()}-{threading.get_ident()}"
        shutil.rmtree(stage, ignore_errors=True)
        done = 0
        try:
            stage.mkdir(parents=True)
            for f in pkg.files:
                dst = stage.joinpath(*f.path.split("/"))
                dst.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                got = 0
                with reg.open(pkg, f) as src, open(dst, "wb") as out:
                    while True:
                        if cancel and cancel():
                            raise PackageError("cancelled", "취소됨")
                        b = src.read(CHUNK)
                        if not b:
                            break
                        got += len(b)
                        if got > f.size:
                            raise PackageError("size", f"{pkg.key}: {f.path}가 목록보다 큽니다")
                        h.update(b)
                        out.write(b)
                        alive()
                        if progress and pkg.size:
                            progress((done + got) / pkg.size,
                                     f"{pkg.label or pkg.key} 받는 중 {(done + got) / 1e6:.0f}/{pkg.size / 1e6:.0f} MB")
                    out.flush()
                    try:
                        os.fsync(out.fileno())
                    except OSError:
                        pass
                if got != f.size:
                    raise PackageError("interrupted", f"{pkg.key}: {f.path}를 받다가 끊겼습니다 "
                                                      f"({got / 1e6:.1f}/{f.size / 1e6:.1f} MB)")
                if h.hexdigest() != f.sha256:
                    raise PackageError("hash", f"{pkg.key}: {f.path}의 SHA-256이 목록과 다릅니다 (손상되었거나 바뀐 파일)")
                done += got
            # the record of what this folder is goes in last; then the folder takes its name in one step
            (stage / "package.json").write_text(json.dumps(pkg.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            os.rename(stage, kd / pkg.version)
        except PackageError:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        except OSError as e:
            shutil.rmtree(stage, ignore_errors=True)
            import errno
            if e.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
                raise PackageError("space", f"{pkg.key}: 디스크가 가득 찼습니다") from e
            raise PackageError("io", f"{pkg.key}: 파일을 쓰지 못했습니다: {e}") from e
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    def _activate(self, key: str, version: str) -> None:
        if version not in self.versions(key):
            raise PackageError("missing", f"{key} {version}: 설치되어 있지 않습니다")
        kd = self._kdir(key)
        a = self._read_active(key)
        prev = [v for v in (a.get("previous") or []) if v != version]
        if a.get("version") and a["version"] != version:
            prev.append(a["version"])
        tmp = kd / f"active.json.tmp{os.getpid()}"
        tmp.write_text(json.dumps(dict(version=version, previous=prev[-8:], since=time.time())), encoding="utf-8")
        os.replace(tmp, kd / "active.json")

    def activate(self, key: str, version: str) -> None:
        """An installed version put in use by hand (going back to an older one)."""
        with self._held(key, 30.0):
            self._activate(key, version)

    def remove(self, key: str, version: str) -> bool:
        """An installed version that is not in use and not pinned by work going on."""
        with self._held(key, 30.0):
            if version == self.active(key) or (key, version) in pinned_now():
                return False
            d = self._kdir(key) / version
            gone = d.with_name(f".gone-{version}-{os.getpid()}")
            try:
                os.rename(d, gone)
            except OSError:
                return False
            shutil.rmtree(gone, ignore_errors=True)
            return True

    def set_aside(self, key: str, version: str) -> Optional[str]:
        """An installed version whose files are no longer what the list said (a disk fault, a file changed by hand)
        is taken out of use: the version in use before it is in use again (active()), and it can be installed
        again. Returns the version now in use (None: none)."""
        with self._held(key, 30.0):
            d = self._kdir(key) / version
            try:
                os.rename(d, d.with_name(f".bad-{version}-{int(time.time())}"))
            except OSError:
                return self.active(key)
            a = self._read_active(key)
            if a.get("version") == version:
                now = self.active(key)
                if now:
                    self._activate(key, now)
                else:
                    try:
                        (self._kdir(key) / "active.json").unlink()
                    except OSError:
                        pass
            return self.active(key)

    def owns(self, path) -> Optional[Tuple[str, str]]:
        """(key, version) when the file or folder is inside an installed package of this store."""
        try:
            rel = Path(path).resolve().relative_to(self.root.resolve())
        except (ValueError, OSError):
            return None
        parts = rel.parts
        if len(parts) >= 2 and _KEY_RE.match(parts[0]) and parts[1] in self.versions(parts[0]):
            return parts[0], parts[1]
        return None

    def keys(self) -> List[str]:
        try:
            return sorted(p.name for p in self.root.iterdir() if p.is_dir() and _KEY_RE.match(p.name) and self.versions(p.name))
        except OSError:
            return []

    def recover(self) -> List[str]:
        """At the start of the app: what an install that never finished left behind is taken away (its lock gone
        stale, its staging folders). What is in use is what was last put in use whole. The names removed."""
        out = []
        try:
            kds = [p for p in self.root.iterdir() if p.is_dir() and _KEY_RE.match(p.name)]
        except OSError:
            return out
        for kd in kds:
            lock = kd / ".lock"
            try:
                busy = lock.exists() and time.time() - lock.stat().st_mtime <= LOCK_STALE
            except OSError:
                busy = False
            if busy:
                continue
            for p in list(kd.iterdir()):
                if p.name.startswith((".stage-", ".gone-", ".bad-")) or p.name.startswith("active.json.tmp") \
                        or p.name == ".lock":
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            p.unlink()
                        except OSError:
                            pass
                    out.append(f"{kd.name}/{p.name}")
        return out


def _ort_version() -> str:
    try:
        import onnxruntime as ort
        return str(ort.__version__)
    except Exception:
        return ""


_store: Optional[Store] = None
_store_lock = threading.Lock()


def store() -> Store:
    """The app's own packages: <models folder>/packages."""
    global _store
    from ..core.settings import settings
    root = settings().models_path / "packages"
    with _store_lock:
        if _store is None or _store.root != root:
            _store = Store(root)
        return _store


# ------------------------------------------------------------------------------------------------- pinned versions
_pins_lock = threading.Lock()
_pins: List[Dict[str, str]] = []           # every piece of work going on in this process, with its versions
_tls = threading.local()


def pinned_now() -> set:
    with _pins_lock:
        return {(k, v) for d in _pins for k, v in d.items()}


@contextlib.contextmanager
def pinned(versions: Optional[Dict[str, str]] = None, keys=None):
    """Work that begins now runs to its end with these versions (default: the ones in use at this moment), also in
    the threads it starts; putting a newer version in use meanwhile does not change what it reads, and its
    versions are not removed. Yields the versions."""
    st = store()
    v = dict(versions) if versions is not None else st.active_versions()
    if keys is not None:
        v = {k: x for k, x in v.items() if k in set(keys)}
    with _pins_lock:
        _pins.append(v)
    old = getattr(_tls, "pins", None)
    _tls.pins = v
    try:
        yield v
    finally:
        _tls.pins = old
        with _pins_lock:
            for i, d in enumerate(_pins):
                if d is v:
                    del _pins[i]
                    break


def current_pins() -> Optional[Dict[str, str]]:
    """The versions this thread's work runs with: its own (pinned / adopt), else those of the pinned work going on
    in this process (an analysis starts threads of its own, and each of them opens models: they are its threads;
    a thread of something else that opens a model meanwhile gets the same installed versions), else None."""
    p = getattr(_tls, "pins", None)
    if p is not None:
        return p
    with _pins_lock:
        return _pins[-1] if _pins else None


@contextlib.contextmanager
def adopt(pins: Optional[Dict[str, str]]):
    """In a thread that goes on after the work that pinned them (or beside other pinned work): these versions."""
    old = getattr(_tls, "pins", None)
    _tls.pins = pins
    try:
        yield
    finally:
        _tls.pins = old


def resolve(key: str) -> Optional[Path]:
    """Where the model is opened from when it is installed as a package: the version this work is pinned to, else
    the one in use. None: not a package on this computer (models.model_path's own place applies), or the pinned
    version is not installed (never another one in its place: needs() tells which)."""
    st = store()
    pins = current_pins()
    if pins is not None and key in pins:
        return st.path(key, pins[key])
    return st.path(key)


class VersionNeeded(PackageError):
    """Work on a cut needs package versions that are not on this computer. needs: [{key, want, have, installed}]"""

    def __init__(self, miss: List[dict]):
        from . import models as M
        names = ", ".join(f"{M.MODELS.get(n['key'], {}).get('label', n['key'])} {n['want']}"
                          + (f" (설치된 버전 {n['have']})" if n["have"] else " (설치 안 됨)") for n in miss)
        super().__init__("version", "이 컷은 다른 모델 버전으로 분석되었습니다: " + names
                         + ". 모델 관리에서 그 버전을 설치하거나, 컷 전체를 지금 버전으로 다시 분석하세요.")
        self.needs = miss


@contextlib.contextmanager
def for_cut(cut):
    """Work on a cut that is analysed already (its matte made again for some frames, its clothing or face masks,
    a stretch analysed again) runs with the package versions the cut was analysed with: one cut is never made of
    two versions of a model. A version it needs that is not on this computer stops the work with VersionNeeded;
    installing that version, or analysing the whole cut again with the one in use, is the person's choice and is
    not taken for them. Inside work that is pinned already (the whole analysis) nothing changes."""
    if getattr(_tls, "pins", None) is not None:
        yield _tls.pins
        return
    want = {k: v for k, v in (getattr(cut, "model_versions", None) or {}).items() if v}
    if not want:
        with pinned() as p:
            yield p
        return
    miss = [n for n in needs(want) if not n["installed"]]
    if miss:
        raise VersionNeeded(miss)
    pins = store().active_versions()
    pins.update(want)
    with pinned(pins) as p:
        yield p


def cut_models(fn):
    """A function whose first argument is a cut's tracker runs inside for_cut(tracker.cut)."""
    import functools

    @functools.wraps(fn)
    def inner(tracker, *a, **k):
        with for_cut(tracker.cut):
            return fn(tracker, *a, **k)
    return inner


def version_in_use(key: str) -> str:
    pins = current_pins()
    if pins is not None and key in pins:
        return pins[key]
    return store().active(key) or ""


def needs(model_versions: Dict[str, str]) -> List[dict]:
    """For a cut analysed with these package versions: the ones this computer does not have in use.
    [{key, want, have, installed}]: installed True = the wanted version is here (it is used for this cut), False =
    it has to be installed, or the cut analysed again with `have` by the user's own choice."""
    st = store()
    out = []
    for k, want in sorted((model_versions or {}).items()):
        have = st.active(k) or ""
        if want and want != have:
            out.append(dict(key=k, want=want, have=have, installed=want in st.versions(k)))
    return out


# ------------------------------------------------------------------------------------------------- bringing in
def registry_at(where) -> "contextlib.AbstractContextManager":
    """A registry by where it is: an http(s) address, a folder, or a .zip of the layout (the offline import).
    Used as `with registry_at(x) as reg:`."""
    w = str(where)
    if re.match(r"^https?://", w, re.I):
        return contextlib.nullcontext(HttpRegistry(w))
    p = Path(w)
    if p.is_file():
        return zip_registry(p)
    if p.is_dir():
        # a folder that holds the layout, or one level above it (a zip unpacked by hand keeps its top folder)
        if not (p / "manifest.json").is_file():
            inner = [d for d in p.iterdir() if d.is_dir() and (d / "manifest.json").is_file()]
            if len(inner) == 1:
                p = inner[0]
        return contextlib.nullcontext(DirRegistry(p))
    raise PackageError("missing", f"모델 패키지를 찾지 못했습니다: {w}")


def offered(reg: Registry, trusted: Optional[Dict[str, str]] = None, app_version: Optional[str] = None,
            ort_version: Optional[str] = None) -> List[dict]:
    """What a registry's signed list offers this app: for every key the newest version it can use, with what is
    in use here. [{key, label, version, size, have, installed, why}] (why: '' or the reason no version fits)."""
    data, sig = reg.manifest()
    pkgs = verify_manifest(data, sig, trusted)
    if app_version is None:
        from ..core.settings import APP_VERSION as app_version
    if ort_version is None:
        ort_version = _ort_version()
    st = store()
    out = []
    for key in sorted({p.key for p in pkgs}):
        mine = sorted((p for p in pkgs if p.key == key), key=lambda p: _ver(p.version))
        ok = [p for p in mine if not (p.fits(app_version, ort_version) or io_mismatch(p))]
        best = ok[-1] if ok else mine[-1]
        out.append(dict(key=key, label=best.label, version=best.version, size=best.size,
                        have=st.active(key) or "", installed=best.version in st.versions(key),
                        why="" if ok else (best.fits(app_version, ort_version) or io_mismatch(best))))
    return out


def bring_in(where, keys=None, progress: Optional[Callable[[float, str], None]] = None,
             cancel: Optional[Callable[[], bool]] = None, trusted: Optional[Dict[str, str]] = None,
             versions: Optional[Dict[str, str]] = None, activate: bool = True, st: Optional[Store] = None) -> dict:
    """Installs from a registry (address, folder or zip) the newest usable version of every key it offers (or of
    `keys`; `versions` names exact ones), each with the full checks. A package that fails leaves the others and
    what was in use alone. -> {"installed": [(key, version)], "kept": [(key, version)], "failed": {key: text}}"""
    st = st or store()
    res = dict(installed=[], kept=[], failed={})
    with registry_at(where) as reg:
        data, sig = reg.manifest()
        pkgs = verify_manifest(data, sig, trusted)            # an unsigned or changed list: nothing is installed
        want = sorted({p.key for p in pkgs} if keys is None else set(keys))
        total = len(want) or 1
        for i, key in enumerate(want):
            if cancel and cancel():
                raise PackageError("cancelled", "취소됨")
            ver = (versions or {}).get(key)
            had = set(st.versions(key))

            def prog(f, msg, i=i):
                if progress:
                    progress((i + f) / total, msg)
            try:
                pkg = st.install(reg, key, ver, progress=prog, cancel=cancel, trusted=trusted, activate=activate)
                (res["kept"] if pkg.version in had else res["installed"]).append((key, pkg.version))
            except PackageError as e:
                if e.why == "cancelled":
                    raise
                res["failed"][key] = str(e)
    if progress:
        progress(1.0, "모델 패키지 확인 끝")
    return res


def listing(st: Optional[Store] = None) -> List[dict]:
    """Every model the app knows and every package on this computer, for the model list:
    [{key, label, source, version, versions, size, pinned}] with source 'package' (installed as a package),
    'app' (downloaded by the app itself, before packages), 'none' (not on this computer)."""
    from . import models as M
    st = st or store()
    pins = pinned_now()
    out = []
    for key in list(M.MODELS) + [k for k in st.keys() if k not in M.MODELS]:
        ver = st.active(key)
        pkg = st.package(key, ver) if ver else None
        spec = M.MODELS.get(key, {})
        if pkg is not None:
            src, size = "package", pkg.size
        elif key in M.MODELS and M.is_downloaded(key):
            src, size = "app", int(spec.get("size_mb", 0)) << 20
        else:
            src, size = "none", int(spec.get("size_mb", 0)) << 20
        out.append(dict(key=key, label=(pkg.label if pkg and pkg.label else spec.get("label", key)), source=src,
                        version=ver or "", versions=st.versions(key), size=size,
                        pinned=sorted(v for k, v in pins if k == key)))
    return out
