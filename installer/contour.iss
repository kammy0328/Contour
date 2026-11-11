; Contour installer (Inno Setup 6). Built by .github/workflows/release.yml:
;   ISCC /DVer=1.8.1 /DSrc=<repo> /DPy=<embeddable python folder> installer\contour.iss
; Per-user install (no admin): {localappdata}\Programs\Contour
;   {app}\python   embeddable Python 3.12 + packages (pip, installed by tools\setup_env.py)
;   {app}\app      Contour itself (replaced on every update)
;   {app}\ffmpeg   ffmpeg, fetched once when none is found on the PC
; Projects, learned data, presets and models stay in Documents\Contour_Data and are never touched.

#ifndef Ver
  #define Ver "0.0.0"
#endif
#ifndef Src
  #define Src ".."
#endif
#ifndef Py
  #define Py "..\build\python"
#endif

[Setup]
AppId={{0432DE7E-AD09-4B89-8929-0B95959C384F}
AppName=Contour
AppVersion={#Ver}
AppVerName=Contour {#Ver}
AppPublisher=SEWOO
DefaultDirName={localappdata}\Programs\Contour
DisableDirPage=auto
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#Src}\dist
OutputBaseFilename=Contour_Setup_{#Ver}
SetupIconFile={#Src}\assets\icon.ico
UninstallDisplayIcon={app}\app\assets\icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "ko"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "learn"; Description: "누적 학습 기능 (PyTorch 약 2.5GB 추가 설치)"; Flags: unchecked
Name: "desktop"; Description: "바탕화면 바로가기"

[InstallDelete]
; old code out first, so a module removed in this version does not linger
Type: filesandordirs; Name: "{app}\app\contour"
Type: filesandordirs; Name: "{app}\app\tools"

[Files]
Source: "{#Py}\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Src}\app.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#Src}\requirements.txt"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#Src}\requirements-learn.txt"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#Src}\README.md"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#Src}\contour\*"; DestDir: "{app}\app\contour"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Src}\tools\*"; DestDir: "{app}\app\tools"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Src}\assets\*"; DestDir: "{app}\app\assets"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Contour"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\app.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\assets\icon.ico"; AppUserModelID: "SEWOO.Contour"
Name: "{autodesktop}\Contour"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\app.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\assets\icon.ico"; AppUserModelID: "SEWOO.Contour"; Tasks: desktop

[Run]
; first install: visible console (package download takes a while)
Filename: "{app}\python\python.exe"; Parameters: """{app}\app\tools\setup_env.py"" --learn {code:LearnFlag}"; WorkingDir: "{app}\app"; StatusMsg: "패키지와 모델을 준비하는 중 (처음엔 5~10분)"; Check: not IsUpdate
; update: hidden, packages only when they changed
Filename: "{app}\python\python.exe"; Parameters: """{app}\app\tools\setup_env.py"" --learn auto"; WorkingDir: "{app}\app"; StatusMsg: "업데이트 마무리 중"; Flags: runhidden; Check: IsUpdate
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\app.py"""; WorkingDir: "{app}\app"; Description: "Contour 실행"; Flags: nowait postinstall skipifsilent
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\app.py"""; WorkingDir: "{app}\app"; Flags: nowait runasoriginaluser; Check: IsUpdate

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\ffmpeg"
Type: files; Name: "{app}\setup_log.txt"

[Code]
function IsUpdate: Boolean;
begin
  Result := ExpandConstant('{param:UPDATE|0}') = '1';
end;

function LearnFlag(Param: String): String;
begin
  if WizardIsTaskSelected('learn') then
    Result := '1'
  else
    Result := 'auto';
end;
