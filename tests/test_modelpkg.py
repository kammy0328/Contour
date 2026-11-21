"""Model packages (instruction 2026-11-20 §4-1 .. §4-3): a signed list, files checked against it, a package put in
use whole or not at all, one install of a package at a time, work pinned to the versions it began with, everything
usable with the network gone, a package brought in from a folder or a zip, and nothing ever sent."""
import errno
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import pytest

from contour.core import ed25519 as ED
from contour.track import modelpkg as MP

SECRET = bytes(range(32))
KEYS = {"test-1": ED.public_key(SECRET).hex()}


def _files(folder: Path, files: dict) -> Path:
    for name, data in files.items():
        p = folder.joinpath(*name.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return folder


def make_registry(root: Path, packages, secret=SECRET, key_id="test-1") -> MP.DirRegistry:
    """packages: [(key, version, {path: bytes}, {more of the list's fields})]"""
    pk = []
    for key, version, files, more in packages:
        _files(root / key / version, files)
        pk.append(MP.describe(root / key / version, key, version, **more))
    data = MP.make_manifest(pk)
    (root / "manifest.json").write_bytes(data)
    (root / "manifest.sig").write_bytes(MP.sign_manifest(data, secret, key_id))
    return MP.DirRegistry(root)


A1 = ("pose_x", "2026.11.1", {"model.onnx": b"A" * 5000, "model.onnx.data": b"a" * 300000}, dict(main="model.onnx"))
A2 = ("pose_x", "2026.12.1", {"model.onnx": b"B" * 5200, "model.onnx.data": b"b" * 310000}, dict(main="model.onnx"))


@pytest.fixture
def st(tmp_path):
    return MP.Store(tmp_path / "packages")


def _install(st, reg, key="pose_x", **kw):
    kw.setdefault("trusted", KEYS)
    kw.setdefault("app_version", "1.23.0")
    kw.setdefault("ort_version", "1.20.0")
    return st.install(reg, key, **kw)


# ------------------------------------------------------------------------------------------------- the signature
def test_ed25519_rfc8032_vectors():
    V = [("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
          "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
          "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
         ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
          "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
          "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
         ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
          "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
          "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
         ("833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42",
          "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf",
          "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
          "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b58909351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704")]
    for sk, pk, msg, sig in V:
        sk, pk, msg, sig = map(bytes.fromhex, (sk, pk, msg, sig))
        assert ED.public_key(sk) == pk and ED.sign(sk, msg) == sig and ED.verify(pk, msg, sig)
        assert not ED.verify(pk, msg + b"\x00", sig)
        bad = bytearray(sig)
        bad[40] ^= 0x10
        assert not ED.verify(pk, msg, bytes(bad))
        assert not ED.verify(pk[:-1] + bytes([pk[-1] ^ 1]), msg, sig)
    assert not ED.verify(b"", b"x", b"") and not ED.verify(b"\xff" * 32, b"x", b"\xff" * 64)


def test_a_list_is_taken_only_with_a_trusted_signature(tmp_path, st):
    reg = make_registry(tmp_path / "reg", [A1])
    data, sig = reg.manifest()
    assert [p.version for p in MP.verify_manifest(data, sig, KEYS)] == ["2026.11.1"]
    # nobody is trusted by default (no publishing key yet): no list is taken
    assert MP.TRUSTED == {}
    with pytest.raises(MP.PackageError) as e:
        MP.verify_manifest(data, sig)
    assert e.value.why == "signature"
    # signed by somebody else
    other = MP.sign_manifest(data, bytes(range(1, 33)), "test-1")
    with pytest.raises(MP.PackageError) as e:
        MP.verify_manifest(data, other, KEYS)
    assert e.value.why == "signature"
    # one byte of the list changed after it was signed (a bigger size for a file, another hash ...)
    with pytest.raises(MP.PackageError) as e:
        MP.verify_manifest(data.replace(b"5000", b"5001"), sig, KEYS)
    assert e.value.why == "signature"
    # a signature that is not one
    for junk in (b"", b"{}", b'{"key": "test-1", "sig": "@@"}', b'{"key": "test-1", "alg": "rsa", "sig": "AAAA"}'):
        with pytest.raises(MP.PackageError) as e:
            MP.verify_manifest(data, junk, KEYS)
        assert e.value.why == "signature"
    # and install refuses the same way, with nothing written
    (tmp_path / "reg" / "manifest.sig").write_bytes(other)
    with pytest.raises(MP.PackageError):
        _install(st, reg)
    assert st.versions("pose_x") == [] and st.active("pose_x") is None


def test_only_files_inside_the_package_can_be_named():
    ok = dict(key="k", version="1", sha256="0" * 64, files=[dict(path="a/b.onnx", size=1, sha256="0" * 64)])
    for bad in ("../x", "/abs", "a/../../x", "C:/x", "a\\b", "", "a//b", "./a", "package.json", ".stage-1/x", ".lock"):
        d = json.loads(json.dumps(ok))
        d["files"][0]["path"] = bad
        with pytest.raises(MP.PackageError):
            MP.package_from(d)
    for bad_key in ("../k", "K", "k/1", ""):
        with pytest.raises(MP.PackageError):
            MP.package_from(dict(ok, key=bad_key))
    for bad_ver in ("../1", "1/2", "", "a b"):
        with pytest.raises(MP.PackageError):
            MP.package_from(dict(ok, version=bad_ver))
    # the hash over all files must be the one of the files listed (a file added to or taken from a signed entry)
    with pytest.raises(MP.PackageError):
        MP.package_from(ok)


# ------------------------------------------------------------------------------------------------- installing
def test_install_checks_every_file_and_puts_the_package_in_use(tmp_path, st):
    reg = make_registry(tmp_path / "reg", [A1])
    seen = []
    pkg = _install(st, reg, progress=lambda f, m: seen.append(f))
    assert pkg.version == "2026.11.1" and st.active("pose_x") == "2026.11.1" and st.verify("pose_x")
    p = st.path("pose_x")
    assert p.name == "model.onnx" and p.read_bytes() == b"A" * 5000
    assert (p.parent / "model.onnx.data").stat().st_size == 300000, "a model with its weights file is one package"
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    assert st.package("pose_x").sha256 == pkg.sha256 and st.active_versions() == {"pose_x": "2026.11.1"}
    # nothing but the package and the record of what is in use
    assert sorted(x.name for x in (st.root / "pose_x").iterdir()) == ["2026.11.1", "active.json"]
    # installing it again reads the list, finds it there and downloads nothing
    class NoFiles(MP.DirRegistry):
        def open(self, pkg, f):
            raise AssertionError("a file was asked for again")
    _install(st, NoFiles(tmp_path / "reg"))


def test_the_newest_version_this_app_and_runtime_can_use(tmp_path, st):
    new_app = ("pose_x", "2027.1.1", {"model.onnx": b"C" * 10}, dict(main="model.onnx", app={"min": "1.30.0"}))
    new_rt = ("pose_x", "2027.2.1", {"model.onnx": b"D" * 10},
              dict(main="model.onnx", runtime={"onnxruntime": {"min": "1.25.0"}}))
    reg = make_registry(tmp_path / "reg", [A1, A2, new_app, new_rt])
    assert _install(st, reg).version == "2026.12.1"
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg, version="2027.1.1")
    assert e.value.why == "compat" and "1.30.0" in str(e.value)
    assert _install(st, reg, app_version="1.30.0").version == "2027.1.1"
    assert _install(st, reg, app_version="1.30.0", ort_version="1.25.1").version == "2027.2.1"
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg, key="nothing")
    assert e.value.why == "missing"


class Cut(MP.DirRegistry):
    """A registry whose connection ends after `after` bytes of the file named."""

    def __init__(self, root, name, after):
        super().__init__(root)
        self.name, self.after = name, after

    def open(self, pkg, f):
        import contextlib
        import io

        @contextlib.contextmanager
        def cm():
            data = (self.root / pkg.key / pkg.version / f.path).read_bytes()
            yield io.BytesIO(data[:self.after] if f.path == self.name else data)
        return cm()


def test_a_download_cut_short_leaves_what_was_in_use(tmp_path, st):
    reg = make_registry(tmp_path / "reg", [A1, A2])
    _install(st, reg, version="2026.11.1")
    with pytest.raises(MP.PackageError) as e:
        _install(st, Cut(tmp_path / "reg", "model.onnx.data", 100000), version="2026.12.1")
    assert e.value.why == "interrupted"
    assert st.active("pose_x") == "2026.11.1" and st.versions("pose_x") == ["2026.11.1"] and st.verify("pose_x")
    assert sorted(x.name for x in (st.root / "pose_x").iterdir()) == ["2026.11.1", "active.json"], "nothing half-made is left"
    # the first install of a package cut short: not installed at all
    st2 = MP.Store(tmp_path / "other")
    with pytest.raises(MP.PackageError):
        _install(st2, Cut(tmp_path / "reg", "model.onnx", 10))
    assert st2.active("pose_x") is None and st2.path("pose_x") is None
    # and the same install goes through when the connection holds
    assert _install(st, reg).version == "2026.12.1" and st.active("pose_x") == "2026.12.1"


def test_a_file_that_is_not_the_one_listed_is_refused(tmp_path, st):
    reg = make_registry(tmp_path / "reg", [A1, A2])
    _install(st, reg, version="2026.11.1")
    f = tmp_path / "reg" / "pose_x" / "2026.12.1" / "model.onnx.data"
    good = f.read_bytes()
    f.write_bytes(b"x" + good[1:])                      # same size, another content
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg)
    assert e.value.why == "hash" and st.active("pose_x") == "2026.11.1" and st.versions("pose_x") == ["2026.11.1"]
    f.write_bytes(good + b"more")                       # longer than listed: stopped at the listed size
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg)
    assert e.value.why == "size"
    f.unlink()
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg)
    assert e.value.why == "missing" and st.active("pose_x") == "2026.11.1"
    # an installed file changed afterwards is found by verify (and by nothing else: it is not read at every use)
    p = st.path("pose_x")
    p.write_bytes(b"Z" * 5000)
    assert not st.verify("pose_x")


def test_a_full_disk_leaves_what_was_in_use(tmp_path, st, monkeypatch):
    reg = make_registry(tmp_path / "reg", [A1, A2])
    _install(st, reg, version="2026.11.1")
    # known before it starts
    import shutil as SH
    real = SH.disk_usage
    monkeypatch.setattr(MP.shutil, "disk_usage", lambda p: type(real(p))(10 ** 12, 10 ** 12 - 1000, 1000))
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg)
    assert e.value.why == "space" and st.active("pose_x") == "2026.11.1"
    monkeypatch.setattr(MP.shutil, "disk_usage", real)
    # or found while writing
    real_open = open

    class Full:
        def __init__(self, fh):
            self.fh = fh

        def write(self, b):
            raise OSError(errno.ENOSPC, "No space left on device")

        def __getattr__(self, n):
            return getattr(self.fh, n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.fh.close()

    def opener(p, mode="r", *a, **k):
        fh = real_open(p, mode, *a, **k)
        return Full(fh) if "w" in mode and ".stage-" in str(p) and str(p).endswith(".data") else fh
    monkeypatch.setattr("builtins.open", opener)
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg)
    monkeypatch.undo()
    assert e.value.why == "space" and st.active("pose_x") == "2026.11.1" and st.verify("pose_x")
    assert sorted(x.name for x in (st.root / "pose_x").iterdir()) == ["2026.11.1", "active.json"]


def test_cancelled_install_leaves_nothing(tmp_path, st):
    reg = make_registry(tmp_path / "reg", [A1])
    n = [0]

    def cancel():
        n[0] += 1
        return n[0] > 1
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg, cancel=cancel)
    assert e.value.why == "cancelled" and st.active("pose_x") is None
    assert [x.name for x in (st.root / "pose_x").iterdir()] == []


def test_two_installs_of_one_package_do_not_run_together(tmp_path, st):
    reg = make_registry(tmp_path / "reg", [A1])
    inside = []
    gate = threading.Event()

    class Slow(MP.DirRegistry):
        def open(self, pkg, f):
            inside.append(threading.get_ident())
            assert len(set(inside)) == 1, "two installs were writing the same package"
            gate.wait(5)
            return super().open(pkg, f)
    res = {}

    def run(name):
        try:
            res[name] = _install(st, Slow(tmp_path / "reg"), wait=20.0).version
        except Exception as e:                           # pragma: no cover
            res[name] = e
    a = threading.Thread(target=run, args=("a",))
    b = threading.Thread(target=run, args=("b",))
    a.start()
    time.sleep(0.2)
    b.start()
    time.sleep(0.3)
    assert (st.root / "pose_x" / ".lock").exists()
    gate.set()
    a.join(20)
    b.join(20)
    assert res == {"a": "2026.11.1", "b": "2026.11.1"}, res
    assert len(set(inside)) == 1, "the second one found the package installed and downloaded nothing"
    assert st.verify("pose_x") and sorted(x.name for x in (st.root / "pose_x").iterdir()) == ["2026.11.1", "active.json"]
    # an install that will not wait says so
    hold = threading.Event()
    go = threading.Event()

    def holder():
        with st._held("pose_x", 1.0):
            go.set()
            hold.wait(5)
    t = threading.Thread(target=holder)
    t.start()
    go.wait(5)
    with pytest.raises(MP.PackageError) as e:
        _install(st, reg, wait=0.2)
    hold.set()
    t.join(5)
    assert e.value.why == "busy"


def test_what_a_dead_install_left_is_cleared_at_start(tmp_path, st, monkeypatch):
    reg = make_registry(tmp_path / "reg", [A1, A2])
    _install(st, reg, version="2026.11.1")
    kd = st.root / "pose_x"
    (kd / ".stage-2026.12.1-999-1").mkdir()
    (kd / ".stage-2026.12.1-999-1" / "model.onnx").write_bytes(b"half")
    (kd / "active.json.tmp999").write_text("{", encoding="utf-8")
    (kd / ".lock").write_text("999 0", encoding="utf-8")
    old = time.time() - MP.LOCK_STALE - 5
    os.utime(kd / ".lock", (old, old))
    assert st.active("pose_x") == "2026.11.1", "what is in use never depended on them"
    gone = st.recover()
    assert len(gone) == 3 and sorted(x.name for x in kd.iterdir()) == ["2026.11.1", "active.json"]
    # a lock that is alive is somebody installing right now: left alone
    (kd / ".stage-2026.12.1-7-7").mkdir()
    (kd / ".lock").write_text("7 0", encoding="utf-8")
    assert st.recover() == [] and (kd / ".stage-2026.12.1-7-7").exists()
    # a stale lock does not stop the next install
    os.utime(kd / ".lock", (old, old))
    assert _install(st, reg).version == "2026.12.1"


def test_the_version_in_use_falls_back_when_its_folder_is_gone(tmp_path, st):
    import shutil
    reg = make_registry(tmp_path / "reg", [A1, A2])
    _install(st, reg, version="2026.11.1")
    _install(st, reg, version="2026.12.1")
    assert st.active("pose_x") == "2026.12.1"
    shutil.rmtree(st.root / "pose_x" / "2026.12.1")
    assert st.active("pose_x") == "2026.11.1" and st.path("pose_x").read_bytes() == b"A" * 5000
    # going back by hand, and removing a version that is not in use
    _install(st, reg, version="2026.12.1")
    st.activate("pose_x", "2026.11.1")
    assert st.active("pose_x") == "2026.11.1"
    assert not st.remove("pose_x", "2026.11.1") and st.remove("pose_x", "2026.12.1")
    assert st.versions("pose_x") == ["2026.11.1"]


# ------------------------------------------------------------------------------------------------- pinned versions
@pytest.fixture
def app_store(tmp_path, monkeypatch):
    from contour.core.settings import settings
    monkeypatch.setattr(type(settings()), "models_path", property(lambda self: tmp_path / "models"))
    MP._store = None
    yield MP.store()
    MP._store = None


def test_work_keeps_the_versions_it_began_with(tmp_path, app_store):
    reg = make_registry(tmp_path / "reg", [A1, A2])
    st = app_store
    _install(st, reg, version="2026.11.1")
    assert MP.resolve("pose_x").read_bytes() == b"A" * 5000 and MP.version_in_use("pose_x") == "2026.11.1"
    seen = {}
    with MP.pinned() as pins:
        assert pins == {"pose_x": "2026.11.1"}
        _install(st, reg, version="2026.12.1")              # a newer one is put in use while the work goes on
        assert st.active("pose_x") == "2026.12.1"
        assert MP.resolve("pose_x").read_bytes() == b"A" * 5000 and MP.version_in_use("pose_x") == "2026.11.1"
        assert not st.remove("pose_x", "2026.11.1"), "a version work is pinned to is not removed"

        def worker():
            seen["its thread"] = MP.version_in_use("pose_x")         # a thread the work starts reads its versions
            with MP.adopt({"pose_x": "2026.12.1"}):
                seen["other work"] = MP.version_in_use("pose_x")
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen == {"its thread": "2026.11.1", "other work": "2026.12.1"}

    def after():
        seen["after"] = MP.version_in_use("pose_x")
    t = threading.Thread(target=after)
    t.start()
    t.join()
    assert seen["after"] == "2026.12.1", "once the work is done a thread reads what is in use"
    assert MP.resolve("pose_x").read_bytes() == b"B" * 5200 and MP.version_in_use("pose_x") == "2026.12.1"
    assert st.remove("pose_x", "2026.11.1")
    # pinned to a version that is not on this computer: never another one in its place
    with MP.pinned({"pose_x": "2026.11.1"}):
        assert MP.resolve("pose_x") is None
    # a model that is not a package: nothing here (models.model_path's own place applies)
    assert MP.resolve("det_yolox_m") is None and MP.version_in_use("det_yolox_m") == ""


def test_needs_names_the_versions_a_cut_was_analysed_with(tmp_path, app_store):
    reg = make_registry(tmp_path / "reg", [A1, A2])
    st = app_store
    _install(st, reg, version="2026.11.1")
    assert MP.needs({"pose_x": "2026.11.1"}) == [] and MP.needs({}) == []
    _install(st, reg, version="2026.12.1")
    assert MP.needs({"pose_x": "2026.11.1"}) == [dict(key="pose_x", want="2026.11.1", have="2026.12.1", installed=True)]
    st.remove("pose_x", "2026.11.1")
    assert MP.needs({"pose_x": "2026.11.1"}) == [dict(key="pose_x", want="2026.11.1", have="2026.12.1", installed=False)]
    assert MP.needs({"seg_x": "1"}) == [dict(key="seg_x", want="1", have="", installed=False)]


# ------------------------------------------------------------------------------------------------- offline, import, nothing sent
def test_installed_packages_need_no_network(tmp_path, app_store, monkeypatch):
    reg = make_registry(tmp_path / "reg", [A1])
    _install(app_store, reg)

    def no_network(*a, **k):
        raise AssertionError("a connection was opened")
    monkeypatch.setattr(MP.urllib.request, "urlopen", no_network)
    import socket
    monkeypatch.setattr(socket, "create_connection", no_network)
    st = MP.Store(app_store.root)                       # the app started again
    assert st.recover() == [] and st.active("pose_x") == "2026.11.1" and st.verify("pose_x")
    assert MP.resolve("pose_x").read_bytes() == b"A" * 5000
    with MP.pinned() as pins:
        assert pins == {"pose_x": "2026.11.1"} and MP.needs(pins) == []
    # a registry that cannot be reached says so, and what is installed stays
    with pytest.raises(MP.PackageError) as e:
        _install(st, MP.HttpRegistry("https://models.invalid/diap"))
    assert e.value.why == "io" and st.active("pose_x") == "2026.11.1"


def test_a_package_brought_in_from_a_folder_or_a_zip(tmp_path, st):
    reg_dir = tmp_path / "usb" / "diap_models"
    make_registry(reg_dir, [A1, A2])
    assert _install(st, MP.DirRegistry(reg_dir), version="2026.11.1").version == "2026.11.1"
    z = tmp_path / "usb" / "models.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in sorted(p for p in reg_dir.rglob("*") if p.is_file()):
            zf.write(f, f.relative_to(reg_dir).as_posix())
    with MP.zip_registry(z) as reg:
        assert _install(st, reg).version == "2026.12.1"
    assert st.active("pose_x") == "2026.12.1" and st.verify("pose_x")
    assert [p.name for p in (tmp_path / "usb").iterdir() if p.name.startswith(".modelpkg_")] == []
    # the same checks as a download: a zip whose list was changed, or whose members point outside
    bad = tmp_path / "usb" / "bad.zip"
    with zipfile.ZipFile(z) as zin, zipfile.ZipFile(bad, "w") as zout:
        for n in zin.namelist():
            data = zin.read(n)
            zout.writestr(n, data.replace(b"2026.12.1", b"2026.12.9") if n == "manifest.json" else data)
    st2 = MP.Store(tmp_path / "p2")
    with MP.zip_registry(bad) as reg, pytest.raises(MP.PackageError) as e:
        _install(st2, reg)
    assert e.value.why == "signature"
    evil = tmp_path / "usb" / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../outside.txt", b"x")
    with pytest.raises(MP.PackageError), MP.zip_registry(evil):
        pass
    assert not (tmp_path / "outside.txt").exists()
    with pytest.raises(MP.PackageError), MP.zip_registry(_files(tmp_path, {"not.zip": b"hello"}) / "not.zip"):
        pass


def test_a_registry_is_only_ever_read(tmp_path, monkeypatch):
    """§4-4: no path sends footage, frames, crops, masks or project files to the model service. Every request this
    module makes is a GET without a body for a path the list names."""
    reg_dir = tmp_path / "reg"
    make_registry(reg_dir, [A1])
    asked = []

    class Resp:
        def __init__(self, data):
            import io
            self.b = io.BytesIO(data)

        def read(self, n=-1):
            return self.b.read(n)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def urlopen(req, timeout=None):
        assert req.get_method() == "GET" and req.data is None, "something was sent"
        assert not [h for h in req.header_items() if h[0].lower() not in ("user-agent",)], req.header_items()
        url = req.full_url
        assert url.startswith("https://models.example/diap/") and "?" not in url
        asked.append(url[len("https://models.example/diap/"):])
        return Resp((reg_dir / asked[-1]).read_bytes())
    monkeypatch.setattr(MP.urllib.request, "urlopen", urlopen)
    st = MP.Store(tmp_path / "packages")
    _install(st, MP.HttpRegistry("https://models.example/diap"))
    assert asked == ["manifest.json", "manifest.sig", "pose_x/2026.11.1/model.onnx", "pose_x/2026.11.1/model.onnx.data"]
    assert st.verify("pose_x")
    # and the source has no way to send: no request with a body, no upload call
    import re
    src = Path(MP.__file__).read_text(encoding="utf-8")
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    code = "\n".join(line.split("#")[0] for line in code.splitlines())
    assert "urlopen" in code and len(code) > 5000, "the whole module is looked at"
    for word in ("POST", "PUT", "data=", "upload", "requests.", "socket", "http.client", "ftplib", "smtplib"):
        assert word not in code, word
    assert code.count("urllib.request.Request(") == 1 and 'method="GET"' in code


# ------------------------------------------------------------------------------------------------- in the app
RVM = ("matte_rvm", "2026.11.1", {"rvm.onnx": b"R" * 4000}, dict(main="rvm.onnx", label="RVM 인물 매트"))
RVM2 = ("matte_rvm", "2026.12.1", {"rvm.onnx": b"S" * 4100}, dict(main="rvm.onnx"))


def test_the_apps_models_are_opened_from_their_package(tmp_path, app_store, monkeypatch):
    from contour.track import models as M

    def no_network(*a, **k):
        raise AssertionError("a connection was opened")
    monkeypatch.setattr(M.urllib.request, "urlopen", no_network)
    legacy = M.legacy_path("matte_rvm")
    assert M.model_path("matte_rvm") == legacy and not M.is_downloaded("matte_rvm") and M.fingerprint("matte_rvm") == ""
    reg = make_registry(tmp_path / "reg", [RVM, RVM2])
    pkg = _install(app_store, reg, key="matte_rvm", version="2026.11.1")
    p = M.model_path("matte_rvm")
    assert p.read_bytes() == b"R" * 4000 and p != legacy
    assert M.is_downloaded("matte_rvm") and M.ensure_model("matte_rvm") == p, "nothing is downloaded for a package"
    fp1 = M.fingerprint("matte_rvm")
    assert len(fp1) == 16 and M.fingerprints(["matte_rvm"]) == {"matte_rvm": fp1}
    # pinned work reads its own version, and says so by its fingerprint
    with MP.pinned():
        _install(app_store, reg, key="matte_rvm", version="2026.12.1")
        assert M.ensure_model("matte_rvm").read_bytes() == b"R" * 4000 and M.fingerprint("matte_rvm") == fp1
    fp2 = M.fingerprint("matte_rvm")
    assert M.ensure_model("matte_rvm").read_bytes() == b"S" * 4100 and len(fp2) == 16 and fp2 != fp1
    # pinned to a version that is not here: an error that names it, not another version and not a download
    app_store.remove("matte_rvm", "2026.11.1")
    with MP.pinned({"matte_rvm": "2026.11.1"}):
        with pytest.raises(RuntimeError) as e:
            M.ensure_model("matte_rvm")
        assert "2026.11.1" in str(e.value) and "RVM" in str(e.value)
    # a model with no package is where it always was
    assert M.model_path("det_yolox_m") == M.legacy_path("det_yolox_m")


def test_the_same_files_have_the_same_fingerprint_as_a_package_as_downloaded(tmp_path, app_store):
    """A cut records the files its analysis ran with (models_used). The models this computer downloaded, published
    as packages and installed (tools/modelpkg.py pack-installed), are the same files: their fingerprint stays what
    it was, so nothing analysed before reads as made with another model."""
    import importlib.util
    from contour.track import models as M
    spec = importlib.util.spec_from_file_location("modelpkg_tool2", Path(__file__).resolve().parents[1] / "tools" / "modelpkg.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    one = M.legacy_path("matte_rvm")
    one.parent.mkdir(parents=True, exist_ok=True)
    one.write_bytes(b"\x08" + b"R" * 4000)
    many = M.legacy_path("seg_sam2_s")
    _files(many, {"vision_encoder.onnx": b"E" * 3000, "vision_encoder.onnx_data": b"e" * 90000,
                  "prompt_encoder_mask_decoder.onnx": b"D" * 2000, "config.json": b"{}", "left.part": b"x"})
    before = {k: M.fingerprint(k) for k in ("matte_rvm", "seg_sam2_s")}
    assert all(len(v) == 16 for v in before.values())
    reg, keys = tmp_path / "reg", tmp_path / "keys"
    assert T.main(["keygen", "--out", str(keys), "--id", "pub-2"]) == 0
    assert T.main(["pack-installed", "--registry", str(reg), "--version", "2026.12.1", "--keys", "matte_rvm,seg_sam2_s"]) == 0
    assert T.main(["sign", "--registry", str(reg), "--secret", str(keys / "pub-2.secret")]) == 0
    res = MP.bring_in(str(reg), trusted={"pub-2": (keys / "pub-2.pub").read_text().strip()})
    assert sorted(k for k, _ in res["installed"]) == ["matte_rvm", "seg_sam2_s"] and not res["failed"]
    assert app_store.owns(M.model_path("matte_rvm")) and app_store.owns(M.model_path("seg_sam2_s"))
    assert {k: M.fingerprint(k) for k in before} == before
    # and with the downloaded files gone (a computer that only ever had the packages)
    one.unlink()
    import shutil
    shutil.rmtree(many)
    assert {k: M.fingerprint(k) for k in before} == before
    # other content is another fingerprint
    _files(tmp_path / "src", {"matte_rvm.onnx": b"\x08" + b"S" * 4000})
    assert T.main(["pack", "--registry", str(reg), "--key", "matte_rvm", "--version", "2026.12.2", "--from",
                   str(tmp_path / "src" / "matte_rvm.onnx")]) == 0
    assert T.main(["sign", "--registry", str(reg), "--secret", str(keys / "pub-2.secret")]) == 0
    MP.bring_in(str(reg), ["matte_rvm"], trusted={"pub-2": (keys / "pub-2.pub").read_text().strip()})
    assert app_store.active("matte_rvm") == "2026.12.2" and M.fingerprint("matte_rvm") != before["matte_rvm"]


def test_a_cut_remembers_the_versions_only_when_there_are_any(tmp_path):
    from conftest import make_video
    from contour.core import media as MD
    from contour.core.project import Project
    prj = Project.create(str(tmp_path), "p")
    cut = prj.add_cut(MD.probe(str(make_video(tmp_path / "a.mp4"))), "c1")
    assert "model_versions" not in cut.to_dict(), "a project that uses no packages is written as before"
    cut.model_versions = {"matte_rvm": "2026.11.1"}
    prj.save()
    again = Project.load(prj.path).cuts[0]
    assert again.model_versions == {"matte_rvm": "2026.11.1"}


def test_work_on_an_analysed_cut_uses_the_versions_it_was_analysed_with(tmp_path, app_store):
    from types import SimpleNamespace as NS
    reg = make_registry(tmp_path / "reg", [RVM, RVM2, A1])
    _install(app_store, reg, key="matte_rvm", version="2026.11.1")
    _install(app_store, reg, key="matte_rvm", version="2026.12.1")
    _install(app_store, reg, key="pose_x", version="2026.11.1")
    cut = NS(model_versions={"matte_rvm": "2026.11.1"})
    with MP.for_cut(cut) as pins:
        assert pins == {"matte_rvm": "2026.11.1", "pose_x": "2026.11.1"}
        assert MP.resolve("matte_rvm").read_bytes() == b"R" * 4000
    # a cut analysed before packages (or with none): the versions in use, held for the length of the work
    with MP.for_cut(NS(model_versions={})) as pins:
        assert pins == {"matte_rvm": "2026.12.1", "pose_x": "2026.11.1"}
    # the version the cut needs is gone: the work does not start, and the error says what is needed
    app_store.remove("matte_rvm", "2026.11.1")
    with pytest.raises(MP.VersionNeeded) as e:
        with MP.for_cut(cut):
            raise AssertionError("the work began with another version")
    assert e.value.needs == [dict(key="matte_rvm", want="2026.11.1", have="2026.12.1", installed=False)]
    assert "2026.11.1" in str(e.value) and "2026.12.1" in str(e.value) and "다시 분석" in str(e.value)
    # inside the whole analysis (pinned to what is in use now) a part of it does not go back to the cut's old versions
    with MP.pinned() as outer:
        with MP.for_cut(cut) as pins:
            assert pins is outer and MP.version_in_use("matte_rvm") == "2026.12.1"
    # the functions that work on a part of a cut go through it
    calls = []

    @MP.cut_models
    def job(tracker, x):
        calls.append((x, MP.version_in_use("matte_rvm")))
        return x * 2
    assert job(NS(cut=NS(model_versions={"matte_rvm": "2026.12.1"})), 3) == 6 and calls == [(3, "2026.12.1")]
    with pytest.raises(MP.VersionNeeded):
        job(NS(cut=cut), 1)
    from contour.track import analyze, clothing, facelayer, matte_refresh, parts, rangefix
    for fn in (analyze.CutTracker.rebuild_matte, clothing.run, clothing.refresh_stale, facelayer.build, parts.run,
               matte_refresh.compute, rangefix.candidate):
        assert getattr(fn, "__wrapped__", None) is not None, fn


def test_an_analysis_runs_with_the_versions_it_began_with_and_writes_them_on_the_cut(tmp_path, app_store, monkeypatch):
    from types import SimpleNamespace as NS
    from contour.track import analyze as AN, matte_rev as MRV, models as M
    reg = make_registry(tmp_path / "reg", [RVM, RVM2, A1])
    _install(app_store, reg, key="matte_rvm", version="2026.11.1")
    _install(app_store, reg, key="pose_x", version="2026.11.1")
    monkeypatch.setattr(MRV, "begin", lambda *a, **k: None)
    monkeypatch.setattr(MRV, "staging", lambda *a, **k: False)
    seen = {}

    def fake_analyze(self, progress, cancel, accuracy):
        seen["start"] = M.ensure_model("matte_rvm").read_bytes()[:1]
        _install(app_store, reg, key="matte_rvm", version="2026.12.1")     # an update arrives half way
        seen["later"] = M.ensure_model("matte_rvm").read_bytes()[:1]
        self.cut.models_used = M.fingerprints(["matte_rvm"])
        self.cut.analyzed = True
        return "tracks"
    monkeypatch.setattr(AN.CutTracker, "_analyze", fake_analyze)
    cut = NS(persons=[], joint_keys={}, analyzed=False, analysis_version=0, cs_analysed="", models_used={},
             model_versions={}, work_range=(0, 4))
    tr = AN.CutTracker.__new__(AN.CutTracker)
    tr.cut, tr.store, tr.proxy = cut, None, NS(n=4)
    assert tr.analyze() == "tracks"
    assert seen == {"start": b"R", "later": b"R"}, "the analysis read one version from start to end"
    assert cut.model_versions == {"matte_rvm": "2026.11.1"}, "only the models it ran with, by the version it ran with"
    assert app_store.active("matte_rvm") == "2026.12.1"
    # a failed analysis leaves the cut's record as it was
    def boom(self, progress, cancel, accuracy):
        self.cut.models_used = {"matte_rvm": "x"}
        raise RuntimeError("no")
    monkeypatch.setattr(AN.CutTracker, "_analyze", boom)
    tr._label_cache, tr._geo_cache, tr._head_raw, tr._skin_models = {}, {}, {}, {}
    with pytest.raises(RuntimeError):
        tr.analyze()
    assert cut.model_versions == {"matte_rvm": "2026.11.1"} and MP.current_pins() is None


def test_results_kept_in_memory_carry_the_versions(tmp_path, app_store):
    from conftest import make_video
    from contour.core import media as MD, playcache as PC
    from contour.core.project import Project
    prj = Project.create(str(tmp_path / "prj"), "p")
    cut = prj.add_cut(MD.probe(str(make_video(tmp_path / "a.mp4"))), "c1")
    d0 = PC.state_digest(prj, cut)
    reg = make_registry(tmp_path / "reg", [RVM, RVM2])
    _install(app_store, reg, key="matte_rvm", version="2026.11.1")
    d1 = PC.state_digest(prj, cut)
    _install(app_store, reg, key="matte_rvm", version="2026.12.1")
    d2 = PC.state_digest(prj, cut)
    assert len({d0, d1, d2}) == 3, "another version in use: what was kept for the cut is not found again"
    assert PC.state_digest(prj, cut) == d2


# ------------------------------------------------------------------------------------------------- more of the store
def test_a_package_for_another_input_size_is_not_put_in_use_under_the_apps_key(tmp_path, st):
    # the app feeds pose_rtmw_x 288 x 384 pictures and reads 133 points: a list that says otherwise is not for it
    ok = ("pose_rtmw_x", "2026.11.1", {"m.onnx": b"x" * 3000}, dict(main="m.onnx", io=dict(input=[288, 384], kpts=133)))
    size = ("pose_rtmw_x", "2026.12.1", {"m.onnx": b"y" * 3000}, dict(main="m.onnx", io=dict(input=[192, 256], kpts=133)))
    pts = ("pose_rtmw_x", "2026.12.2", {"m.onnx": b"z" * 3000}, dict(main="m.onnx", io=dict(input=[288, 384], kpts=17)))
    reg = make_registry(tmp_path / "reg", [ok, size, pts])
    assert _install(st, reg, key="pose_rtmw_x").version == "2026.11.1", "the newest that fits, not the newest"
    for v, word in (("2026.12.1", "입력 크기"), ("2026.12.2", "133")):
        with pytest.raises(MP.PackageError) as e:
            _install(st, reg, key="pose_rtmw_x", version=v)
        assert e.value.why == "compat" and word in str(e.value)
    assert st.active("pose_rtmw_x") == "2026.11.1"


def test_an_installed_version_that_went_bad_is_taken_out_of_use(tmp_path, app_store, monkeypatch):
    from contour.track import models as M
    reg = make_registry(tmp_path / "reg", [RVM, RVM2])
    _install(app_store, reg, key="matte_rvm", version="2026.11.1")
    _install(app_store, reg, key="matte_rvm", version="2026.12.1")
    p = M.model_path("matte_rvm")
    assert app_store.owns(p) == ("matte_rvm", "2026.12.1") and app_store.owns(tmp_path / "elsewhere.onnx") is None
    assert M._package_of(M.legacy_path("matte_rvm")) is None

    def not_a_model(path):
        raise RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model failed: Protobuf parsing failed.")
    monkeypatch.setattr(M, "_session_raw", not_a_model)
    # the file is what the list said (this fake is no model at all): it is not renamed and not taken out of use
    with pytest.raises(M.ModelFileError) as e:
        M.session(p)
    assert p.is_file() and app_store.active("matte_rvm") == "2026.12.1" and "목록과 같습니다" in str(e.value)
    # the file changed on the disk: the version goes out of use, the one before it is in use, nothing is fetched
    p.write_bytes(b"T" * 4100)
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    with pytest.raises(M.ModelFileError) as e:
        M.session(p)
    assert "2026.12.1" in str(e.value) and "2026.11.1" in str(e.value) and "다시 설치" in str(e.value)
    assert app_store.active("matte_rvm") == "2026.11.1" and app_store.versions("matte_rvm") == ["2026.11.1"]
    assert M.model_path("matte_rvm").read_bytes() == b"R" * 4000
    # it can be installed again, and what was set aside is cleared at the next start
    _install(app_store, reg, key="matte_rvm", version="2026.12.1")
    assert app_store.active("matte_rvm") == "2026.12.1" and app_store.verify("matte_rvm")
    assert any(".bad-2026.12.1" in n for n in app_store.recover())


def test_what_work_needs_is_told_without_a_connection(tmp_path, app_store, monkeypatch):
    from contour.track import models as M
    import urllib.error

    def offline(*a, **k):
        raise urllib.error.URLError("[Errno 11001] getaddrinfo failed")
    monkeypatch.setattr(M.urllib.request, "urlopen", offline)
    av = {r["task"]: r for r in M.availability()}
    assert not av["analyze"]["ok"] and {m["key"] for m in av["analyze"]["missing"]} >= {"det_yolox_m", "seg_sam31"}
    assert av["analyze"]["limit"] and all(m["label"] and m["size_mb"] for m in av["analyze"]["missing"])
    assert av["face"]["ok"], "the face model that is part of the app needs nothing"
    with pytest.raises(M.ModelMissing) as e:
        M.ensure_model("det_yolox_m")
    t = str(e.value)
    assert "YOLOX-m" in t and "모델 관리" in t and "렌더" in t and e.value.key == "det_yolox_m"
    # a package for it brought in from a folder: the work has it, still without a connection
    reg = make_registry(tmp_path / "reg", [("det_yolox_m", "2026.11.1", {"det_yolox_m.onnx": b"D" * 4000},
                                            dict(main="det_yolox_m.onnx"))])
    _install(app_store, reg, key="det_yolox_m")
    assert M.ensure_model("det_yolox_m").read_bytes() == b"D" * 4000
    assert "det_yolox_m" not in {m["key"] for m in M.availability()[1]["missing"]}


def test_bringing_in_a_folder_or_a_zip_installs_what_it_holds_and_says_what_failed(tmp_path, app_store, monkeypatch):
    monkeypatch.setattr(MP, "TRUSTED", dict(KEYS))
    root = tmp_path / "reg"
    make_registry(root, [RVM, RVM2, A1])
    (root / "pose_x" / "2026.11.1" / "model.onnx.data").write_bytes(b"?" * 300000)      # changed after signing
    seen = []
    res = MP.bring_in(root, progress=lambda f, m: seen.append(f))
    assert res["installed"] == [("matte_rvm", "2026.12.1")] and list(res["failed"]) == ["pose_x"]
    assert "SHA-256" in res["failed"]["pose_x"] and app_store.active("pose_x") is None
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    assert MP.bring_in(root, keys=["matte_rvm"])["kept"] == [("matte_rvm", "2026.12.1")]
    # one level above the layout (a zip unpacked by hand), an exact version, and a zip
    outer = tmp_path / "outer"
    outer.mkdir()
    os.rename(root, outer / "models")
    res = MP.bring_in(outer, keys=["matte_rvm"], versions={"matte_rvm": "2026.11.1"}, activate=False)
    assert res["installed"] == [("matte_rvm", "2026.11.1")] and app_store.active("matte_rvm") == "2026.12.1"
    rows = {r["key"]: r for r in MP.listing()}
    assert rows["matte_rvm"]["source"] == "package" and rows["matte_rvm"]["versions"] == ["2026.11.1", "2026.12.1"]
    assert rows["matte_rvm"]["version"] == "2026.12.1" and rows["det_yolox_m"]["source"] == "none"
    off = {r["key"]: r for r in (lambda reg: MP.offered(reg))(MP.DirRegistry(outer / "models"))}
    assert off["matte_rvm"]["version"] == "2026.12.1" and off["matte_rvm"]["have"] == "2026.12.1" \
        and off["matte_rvm"]["installed"] and not off["pose_x"]["installed"]
    # a list signed by a key the app does not trust: nothing at all is installed
    make_registry(tmp_path / "other", [A2], secret=bytes(range(1, 33)), key_id="test-1")
    with pytest.raises(MP.PackageError) as e:
        MP.bring_in(tmp_path / "other")
    assert e.value.why == "signature" and app_store.active("pose_x") is None
    with pytest.raises(MP.PackageError) as e:
        MP.bring_in(tmp_path / "nowhere")
    assert e.value.why == "missing"


def test_a_folder_or_zip_with_some_of_the_lists_packages_gives_those(tmp_path, app_store, monkeypatch):
    """One model brought in by hand: the signed list names every package, the zip carries one (tools/modelpkg.py
    zip --keys). That one is installed; the others are not failures, they are simply not there. An older version
    that is there is taken when the newest is not."""
    monkeypatch.setattr(MP, "TRUSTED", dict(KEYS))
    root = tmp_path / "reg"
    make_registry(root, [RVM, RVM2, A1, A2])
    part = tmp_path / "part"
    (part / "matte_rvm").mkdir(parents=True)
    for f in ("manifest.json", "manifest.sig"):
        (part / f).write_bytes((root / f).read_bytes())
    os.rename(root / "matte_rvm" / "2026.11.1", part / "matte_rvm" / "2026.11.1")     # only the older version
    res = MP.bring_in(part)
    assert res["installed"] == [("matte_rvm", "2026.11.1")] and res["failed"] == {} and res["absent"] == ["pose_x"]
    assert app_store.active("matte_rvm") == "2026.11.1" and app_store.active("pose_x") is None
    # asked for by name: it has to be there
    res = MP.bring_in(part, keys=["pose_x"])
    assert list(res["failed"]) == ["pose_x"] and "파일이 없습니다" in res["failed"]["pose_x"] and res["absent"] == []
    # a package whose folder is there with a file gone is a failure, not an absence
    (part / "pose_x" / "2026.12.1").mkdir(parents=True)
    (part / "pose_x" / "2026.12.1" / "model.onnx").write_bytes(b"B" * 5200)
    res = MP.bring_in(part)
    assert res["kept"] == [("matte_rvm", "2026.11.1")] and list(res["failed"]) == ["pose_x"] and res["absent"] == []
    # the same as a zip
    z = tmp_path / "one.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in ("manifest.json", "manifest.sig"):
            zf.write(root / f, f)
        zf.write(root / "matte_rvm" / "2026.12.1" / "rvm.onnx", "matte_rvm/2026.12.1/rvm.onnx")
    res = MP.bring_in(z)
    assert res["installed"] == [("matte_rvm", "2026.12.1")] and res["failed"] == {} and res["absent"] == ["pose_x"]
    assert app_store.active("matte_rvm") == "2026.12.1"


def test_the_publishing_tool_makes_a_registry_the_app_takes(tmp_path, app_store, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("modelpkg_tool", Path(__file__).resolve().parents[1] / "tools" / "modelpkg.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    keys, reg, src = tmp_path / "keys", tmp_path / "reg", tmp_path / "src"
    assert T.main(["keygen", "--out", str(keys), "--id", "pub-1"]) == 0
    pub = (keys / "pub-1.pub").read_text().strip()
    assert f'"pub-1": "{pub}"' in capsys.readouterr().out and len((keys / "pub-1.secret").read_text().strip()) == 64
    with pytest.raises(SystemExit):
        T.main(["keygen", "--out", str(keys), "--id", "pub-1"])          # the key is never written over
    _files(src / "one", {"matte_rvm.onnx": b"R" * 4000})
    _files(src / "many", {"vision_encoder.onnx": b"E" * 3000, "vision_encoder.onnx_data": b"e" * 90000,
                          "left.part": b"x", "old.onnx.broken1700000000": b"y"})
    assert T.main(["pack", "--registry", str(reg), "--key", "matte_rvm", "--version", "2026.11.1", "--from",
                   str(src / "one" / "matte_rvm.onnx"), "--app-min", "1.20.0", "--backends", "cuda,cpu"]) == 0
    assert T.main(["pack", "--registry", str(reg), "--key", "seg_sam2_s", "--version", "2026.11.1", "--from",
                   str(src / "many")]) == 0
    pk = {p.key: p for p in MP.parse_manifest((reg / "manifest.json").read_bytes())}
    assert pk["matte_rvm"].main == "matte_rvm.onnx" and pk["matte_rvm"].app_min == "1.20.0" \
        and pk["matte_rvm"].backends == ("cuda", "cpu") and pk["matte_rvm"].label == "RVM 인물 매트"
    assert sorted(f.path for f in pk["seg_sam2_s"].files) == ["vision_encoder.onnx", "vision_encoder.onnx_data"]
    assert pk["seg_sam2_s"].io["input"] == [1024, 1024], "what the app feeds the model is in the list"
    # a version never changes: the same content again is fine, other content under the same version is refused
    assert T.main(["pack", "--registry", str(reg), "--key", "matte_rvm", "--version", "2026.11.1", "--from",
                   str(src / "one" / "matte_rvm.onnx"), "--app-min", "1.20.0", "--backends", "cuda,cpu"]) == 0
    (src / "one" / "matte_rvm.onnx").write_bytes(b"S" * 4000)
    with pytest.raises(SystemExit):
        T.main(["pack", "--registry", str(reg), "--key", "matte_rvm", "--version", "2026.11.1", "--from",
                str(src / "one" / "matte_rvm.onnx")])
    assert (reg / "matte_rvm" / "2026.11.1" / "matte_rvm.onnx").read_bytes() == b"R" * 4000
    # unsigned: the app takes nothing; signed: verify passes, a changed file is told
    assert T.main(["verify", "--registry", str(reg), "--pub", pub, "--id", "pub-1"]) == 1
    assert T.main(["sign", "--registry", str(reg), "--secret", str(keys / "pub-1.secret")]) == 0
    assert T.main(["verify", "--registry", str(reg), "--pub", pub, "--id", "pub-1"]) == 0
    assert T.main(["verify", "--registry", str(reg)]) == 1, "not a key this app trusts"
    z = tmp_path / "models.zip"
    assert T.main(["zip", "--registry", str(reg), "--out", str(z), "--keys", "matte_rvm"]) == 0
    with zipfile.ZipFile(z) as zf:
        assert sorted(zf.namelist()) == ["manifest.json", "manifest.sig", "matte_rvm/2026.11.1/matte_rvm.onnx"]
    assert T.main(["install", "--from", str(z), "--keys", "matte_rvm", "--pub", pub, "--id", "pub-1"]) == 0
    assert app_store.active("matte_rvm") == "2026.11.1"
    assert T.main(["install", "--from", str(z), "--pub", pub, "--id", "pub-1"]) == 0, "what the zip holds is enough"
    assert T.main(["install", "--from", str(z), "--keys", "seg_sam2_s", "--pub", pub, "--id", "pub-1"]) == 1, \
        "asked for by name and not in the zip"
    assert app_store.active("seg_sam2_s") is None
    assert T.main(["list"]) == 0 and "matte_rvm" in capsys.readouterr().out
    (reg / "seg_sam2_s" / "2026.11.1" / "vision_encoder.onnx").write_bytes(b"F" * 3000)
    assert T.main(["verify", "--registry", str(reg), "--pub", pub, "--id", "pub-1"]) == 1
    # packing after signing takes the signature away (it was of the list before)
    assert T.main(["pack", "--registry", str(reg), "--key", "matte_rvm", "--version", "2026.12.1", "--from",
                   str(src / "one" / "matte_rvm.onnx")]) == 0
    assert not (reg / "manifest.sig").exists()
