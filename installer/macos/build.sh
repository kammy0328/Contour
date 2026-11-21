#!/bin/bash
# DIAP.app and DIAP_<version>_macos_arm64.dmg, on an Apple Silicon Mac (macOS 12 or later) with Python 3.12.
#
#   installer/macos/build.sh
#
# Makes build/venv-mac (packages of requirements-mac.txt: the native arm64 Python, Qt, ONNX Runtime with Core ML),
# takes ffmpeg and ffprobe (FFMPEG_BIN=<folder with both>, else Homebrew's, made self-contained), builds the app
# (installer/macos/DIAP.spec), runs the app's own short run inside the built bundle, and makes the disk image with
# its SHA-256 beside it (what the update check reads).
# Signing: with DIAP_CODESIGN_ID='Developer ID Application: NAME (TEAM)' the app and the image are signed with it;
# with NOTARY_PROFILE=<name of a notarytool keychain profile> the image is also notarized and stapled (outside
# distribution needs both). Without them the app is signed ad hoc: for this Mac and for trying on others.
# (Written without a Mac at hand and not run yet: 미검증. The first run on a Mac decides what needs changing.)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
[ "$(uname -s)" = "Darwin" ] || { echo "macOS에서 실행하세요"; exit 1; }
[ "$(uname -m)" = "arm64" ] || { echo "Apple Silicon(arm64) Mac에서 실행하세요 (Rosetta 터미널이 아닌지 확인)"; exit 1; }
PY="${PYTHON:-python3.12}"
VER="$("$PY" -c "import re; print(re.search(r'APP_VERSION = \"(.+)\"', open('contour/core/settings.py', encoding='utf-8').read()).group(1))")"
echo "=== DIAP $VER · $(sw_vers -productVersion) · $(uname -m) · $("$PY" --version)"

# 1. packages (one runtime package only: onnxruntime, which has Core ML on macOS)
"$PY" -m venv build/venv-mac
source build/venv-mac/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mac.txt "pyinstaller>=6.6"
python - <<'PYEOF'
import platform, onnxruntime as ort
from importlib import metadata as m
rt = sorted({(d.metadata.get("Name") or "").lower() for d in m.distributions() if (d.metadata.get("Name") or "").lower().startswith("onnxruntime")})
assert platform.machine() == "arm64", "this Python is not arm64 (Rosetta?)"
assert rt == ["onnxruntime"], f"one runtime package only: {rt}"
assert "CoreMLExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
print("python", platform.python_version(), platform.machine(), "onnxruntime", ort.__version__, ort.get_available_providers())
PYEOF

# 2. ffmpeg and ffprobe for inside the app
mkdir -p build/mac/ffmpeg/bin build/mac/ffmpeg/lib
if [ -n "${FFMPEG_BIN:-}" ]; then
  cp "$FFMPEG_BIN/ffmpeg" "$FFMPEG_BIN/ffprobe" build/mac/ffmpeg/bin/
else
  command -v brew >/dev/null || { echo "FFMPEG_BIN=<ffmpeg와 ffprobe가 있는 폴더>를 주거나 Homebrew를 설치하세요"; exit 1; }
  brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
  brew list dylibbundler >/dev/null 2>&1 || brew install dylibbundler
  cp "$(brew --prefix)/bin/ffmpeg" "$(brew --prefix)/bin/ffprobe" build/mac/ffmpeg/bin/
  chmod u+w build/mac/ffmpeg/bin/*
  # the libraries they load are copied beside them and looked up there (nothing of Homebrew is needed to run)
  dylibbundler -od -b -x build/mac/ffmpeg/bin/ffmpeg -x build/mac/ffmpeg/bin/ffprobe \
               -d build/mac/ffmpeg/lib -p @executable_path/../lib
fi
export FFMPEG_BIN="$ROOT/build/mac/ffmpeg/bin"
"$FFMPEG_BIN/ffmpeg" -hide_banner -version | head -1
file "$FFMPEG_BIN/ffmpeg" | grep -q arm64 || { echo "ffmpeg가 arm64가 아닙니다"; exit 1; }
ENC="$("$FFMPEG_BIN/ffmpeg" -hide_banner -encoders)"
for e in prores_ks libx265 aac; do echo "$ENC" | grep -q " $e " || { echo "ffmpeg에 $e 인코더가 없습니다"; exit 1; }; done

# 3. icon
if [ -f assets/icon.png ]; then
  rm -rf build/mac/icon.iconset && mkdir -p build/mac/icon.iconset
  for s in 16 32 128 256 512; do
    sips -z $s $s assets/icon.png --out build/mac/icon.iconset/icon_${s}x${s}.png >/dev/null
    sips -z $((s*2)) $((s*2)) assets/icon.png --out build/mac/icon.iconset/icon_${s}x${s}@2x.png >/dev/null
  done
  iconutil -c icns build/mac/icon.iconset -o build/mac/icon.icns
fi

# 4. the app
rm -rf dist/DIAP.app dist/DIAP
pyinstaller --noconfirm --distpath dist --workpath build/mac/work installer/macos/DIAP.spec
APP="dist/DIAP.app"
[ -x "$APP/Contents/MacOS/DIAP" ] || { echo "앱이 만들어지지 않았습니다"; exit 1; }
if [ -n "${DIAP_CODESIGN_ID:-}" ]; then
  codesign --force --deep --options runtime --timestamp --entitlements installer/macos/entitlements.plist \
           --sign "$DIAP_CODESIGN_ID" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP"

# 5. the built app's own short run: project opened, proxy, the models that come with it, one cut exported, the
#    child job started without a second window
QT_QPA_PLATFORM=offscreen "$APP/Contents/MacOS/DIAP" --worker smoke --json "$ROOT/dist/smoke_macos.json"

# 6. the disk image and its SHA-256
DMG="dist/DIAP_${VER}_macos_arm64.dmg"
rm -f "$DMG"
rm -rf build/mac/dmg && mkdir -p build/mac/dmg
cp -R "$APP" build/mac/dmg/
ln -s /Applications build/mac/dmg/Applications
hdiutil create -volname "DIAP $VER" -srcfolder build/mac/dmg -ov -format UDZO "$DMG"
if [ -n "${DIAP_CODESIGN_ID:-}" ]; then
  codesign --force --timestamp --sign "$DIAP_CODESIGN_ID" "$DMG"
  if [ -n "${NOTARY_PROFILE:-}" ]; then
    xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$DMG"
    spctl --assess --type open --context context:primary-signature -v "$DMG"
  else
    echo "공증하지 않았습니다 (NOTARY_PROFILE 없음): 외부 배포 전에 공증이 필요합니다"
  fi
else
  echo "임시 서명(ad hoc)입니다: 다른 Mac에서는 '확인 없이 열기'가 필요하고 외부 배포용이 아닙니다"
fi
shasum -a 256 "$DMG" | awk '{print $1"  '"$(basename "$DMG")"'"}' > "$DMG.sha256"
ls -lh "$DMG" && cat "$DMG.sha256"
echo "=== 완료: $DMG"
