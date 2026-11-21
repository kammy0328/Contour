# -*- mode: python ; coding: utf-8 -*-
# DIAP.app for macOS on Apple Silicon (PyInstaller, one folder inside the bundle). Built by installer/macos/build.sh:
#   pyinstaller --noconfirm --distpath dist --workpath build/mac/work installer/macos/DIAP.spec
# FFMPEG_BIN: the folder with the ffmpeg and ffprobe that go into the app (build.sh prepares it).
# DIAP_CODESIGN_ID: a 'Developer ID Application: ...' identity for outside distribution; without it the app is
# signed ad hoc (runs on the Mac that built it and on others after 'Open Anyway', not for distribution).
import os
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]
VER = re.search(r'APP_VERSION = "(.+)"', (ROOT / "contour" / "core" / "settings.py").read_text(encoding="utf-8")).group(1)
FF = Path(os.environ.get("FFMPEG_BIN", ROOT / "build" / "mac" / "ffmpeg" / "bin"))

# what the code opens by its own path: the same places as in the source tree
datas = [(str(ROOT / "assets"), "assets"),
         (str(ROOT / "contour" / "track" / "data"), "contour/track/data"),
         (str(ROOT / "installer" / "RELEASE_NOTES.md"), "installer"),
         (str(ROOT / "README.md"), "."),
         (str(ROOT / "tools" / "ci_smoke.py"), "tools")]            # `DIAP --worker smoke`: the run every system makes
binaries = [(str(FF / n), "ffmpeg/bin") for n in ("ffmpeg", "ffprobe") if (FF / n).is_file()]
libs = FF.parent / "lib"                                            # an ffmpeg made self-contained keeps its libraries here
if libs.is_dir():
    binaries += [(str(p), "ffmpeg/lib") for p in sorted(libs.iterdir()) if p.is_file()]

a = Analysis([str(ROOT / "app.py")], pathex=[str(ROOT)], binaries=binaries, datas=datas,
             hiddenimports=collect_submodules("contour") + ["PyOpenColorIO"],
             excludes=["torch", "torchvision", "tkinter", "matplotlib", "IPython", "pytest"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DIAP", console=False, target_arch="arm64",
          codesign_identity=os.environ.get("DIAP_CODESIGN_ID") or None,
          entitlements_file=str(ROOT / "installer" / "macos" / "entitlements.plist"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="DIAP")
icon = ROOT / "build" / "mac" / "icon.icns"
app = BUNDLE(coll, name="DIAP.app", icon=str(icon) if icon.is_file() else None, bundle_identifier="kr.sewoo.diap",
             version=VER,
             info_plist={"CFBundleName": "DIAP", "CFBundleDisplayName": "DIAP", "CFBundleShortVersionString": VER,
                         "CFBundleVersion": VER, "NSHighResolutionCapable": True, "LSMinimumSystemVersion": "12.0",
                         "LSApplicationCategoryType": "public.app-category.video",
                         "NSRequiresAquaSystemAppearance": False})
