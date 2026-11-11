$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

function Say($t, $c = "Gray") { Write-Host $t -ForegroundColor $c }
function Stop-With($t) { Say $t Red; Read-Host "엔터를 누르면 닫힙니다"; exit 1 }

Say ""
Say "  Contour 설치" White
Say "  영상 인물 몸매 보정 · Windows · Python 3.12" DarkGray
Say ""

# ------------------------------------------------------------------ 1. Python 3.12
function Find-Py {
    try {
        $v = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return $v.Trim() }
    } catch {}
    foreach ($p in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "C:\Program Files\Python312\python.exe", "C:\Python312\python.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}
$py = Find-Py
if (-not $py) {
    Say "[1/6] Python 3.12이 없어 설치합니다 (winget)" Yellow
    try { winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements --silent } catch {}
    $py = Find-Py
}
if (-not $py) { Stop-With "Python 3.12을 설치하지 못했습니다. https://www.python.org/downloads/ 에서 Python 3.12(64-bit)을 설치한 뒤 install.bat을 다시 실행하세요." }
Say "[1/6] Python: $py" Green

# ------------------------------------------------------------------ 2. venv + packages
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $py -m venv .venv
    if ($LASTEXITCODE -ne 0) { Stop-With "가상환경(.venv)을 만들지 못했습니다." }
}
$vpy = (Resolve-Path ".venv\Scripts\python.exe").Path
& $vpy -m pip install --upgrade pip --quiet
Say "[2/6] 필수 패키지 설치 중 (처음엔 5~10분 걸립니다)" Cyan
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Stop-With "패키지 설치에 실패했습니다. 인터넷 연결을 확인하고 다시 실행하세요." }

# ------------------------------------------------------------------ 3. GPU
Say "[3/6] GPU 확인 중" Cyan
$gpu = (& $vpy tools\check_gpu.py 2>$null | Select-Object -Last 1)
if ($gpu -ne "CUDA") {
    Say "      CUDA를 쓸 수 없어 DirectML로 바꿉니다 (NVIDIA 드라이버가 오래되면 그렇습니다. 드라이버를 업데이트한 뒤 install.bat을 다시 실행하면 CUDA로 돌아갑니다)." Yellow
    & $vpy -m pip uninstall -y onnxruntime-gpu | Out-Null
    & $vpy -m pip install onnxruntime-directml==1.24.4
    $gpu = (& $vpy tools\check_gpu.py 2>$null | Select-Object -Last 1)
}
Say "      추론 장치: $gpu" Green

# ------------------------------------------------------------------ 4. learning (optional)
Say "[4/6] 누적 학습 기능" Cyan
$ans = Read-Host "      PyTorch(약 2.5GB)를 설치해 누적 학습을 켤까요? [Y/n]"
if ($ans -eq "" -or $ans -match "^[yY]") {
    & $vpy -m pip install -r requirements-learn.txt
    if ($LASTEXITCODE -ne 0) { Say "      PyTorch 설치에 실패했습니다. 나중에 install.bat을 다시 실행하면 됩니다. (분석·보정·렌더는 그대로 쓸 수 있습니다)" Yellow }
} else {
    Say "      건너뜀 (나중에 install.bat을 다시 실행하면 설치할 수 있습니다)" DarkGray
}

# ------------------------------------------------------------------ 5. models
Say "[5/6] 모델 받는 중 (약 1.1GB, 한 번만)" Cyan
& $vpy tools\download_models.py

# ------------------------------------------------------------------ 6. ffmpeg + shortcut
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ff -or (Test-Path "C:\ffmpeg\bin\ffmpeg.exe")) { Say "[6/6] ffmpeg 확인됨" Green }
else { Say "[6/6] ffmpeg를 찾지 못했습니다. C:\ffmpeg\bin 에 두거나 앱의 설정에서 폴더를 지정하세요." Yellow }
try {
    $ws = New-Object -ComObject WScript.Shell
    # versioned name: installing a new version never repoints the shortcut of an older one
    $ver = (Select-String -Path "contour\core\settings.py" -Pattern 'APP_VERSION = "(.+)"').Matches[0].Groups[1].Value
    $lnk = $ws.CreateShortcut([Environment]::GetFolderPath("Desktop") + "\Contour $ver.lnk")
    $lnk.TargetPath = (Resolve-Path ".venv\Scripts\pythonw.exe").Path
    $lnk.Arguments = '"' + (Resolve-Path "app.py").Path + '"'
    $lnk.WorkingDirectory = $PSScriptRoot
    $lnk.IconLocation = (Resolve-Path "assets\icon.ico").Path
    $lnk.Save()
    Say "      바탕화면에 Contour 바로가기를 만들었습니다." Green
} catch {}

Say ""
Say "  설치 완료. 바탕화면의 Contour 아이콘이나 run.bat으로 실행하세요." White
Say ""
Read-Host "엔터를 누르면 닫힙니다"
