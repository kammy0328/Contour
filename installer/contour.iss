; DIAP installer (Inno Setup 6). Built by .github/workflows/release.yml:
; DIAP installer.
;   ISCC /DAppVer=1.8.1 /DSrc=<repo> /DPy=<embeddable python folder> installer\contour.iss
; Installs for all users in Program Files\Diap (asks for admin rights once per install or update)
;   {app}\python   embeddable Python 3.12 + packages (pip, installed by tools\setup_env.py)
; {app}\app      DIAP itself (code package: contour) (replaced on every update)
;   {app}\ffmpeg   ffmpeg and ffprobe, packed into the installer (never looked up elsewhere on the PC)
; Projects, learned data, presets and models stay in Documents\Contour_Data and are never touched.

#ifndef AppVer
  #define AppVer "0.0.0"
#endif
#ifndef Src
  #define Src ".."
#endif
#ifndef Py
  #define Py "..\build\python"
#endif
#ifndef Ffm
  #define Ffm "..\build\ffmpeg"
#endif

[Setup]
AppId={{0432DE7E-AD09-4B89-8929-0B95959C384F}
AppName=DIAP
AppVersion={#AppVer}
AppVerName=DIAP {#AppVer}
AppPublisher=SEWOO
DefaultDirName={autopf}\Diap
DefaultGroupName=DIAP
DisableDirPage=auto
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir={#Src}\dist
OutputBaseFilename=DIAP_Setup_{#AppVer}
SetupIconFile={#Src}\assets\icon.ico
UninstallDisplayIcon={app}\python\DIAP.exe
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
Name: "desktop"; Description: "바탕화면 바로가기"

[InstallDelete]
; old code out first, so a module removed in this version does not linger
Type: filesandordirs; Name: "{app}\app\contour"
Type: filesandordirs; Name: "{app}\app\tools"

[Dirs]
; originals gathered for training and the developer's step training live here; the app (not elevated) writes to it
Name: "{app}\Cloud"; Permissions: users-modify
Name: "{app}\Cloud\data"; Permissions: users-modify

[Files]
Source: "{#Src}\cloud\*"; DestDir: "{app}\Cloud"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion
Source: "{#Py}\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Ffm}\bin\*"; DestDir: "{app}\ffmpeg\bin"; Flags: ignoreversion
Source: "{#Src}\app.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#Src}\requirements.txt"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#Src}\requirements-learn.txt"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#Src}\README.md"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#Src}\contour\*"; DestDir: "{app}\app\contour"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Src}\tools\*"; DestDir: "{app}\app\tools"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Src}\assets\*"; DestDir: "{app}\app\assets"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\DIAP"; Filename: "{app}\python\DIAP.exe"; Parameters: """{app}\app\app.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\python\DIAP.exe"; AppUserModelID: "SEWOO.DIAP"
Name: "{autodesktop}\DIAP"; Filename: "{app}\python\DIAP.exe"; Parameters: """{app}\app\app.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\python\DIAP.exe"; AppUserModelID: "SEWOO.DIAP"; Tasks: desktop

[Run]
; first install into this folder (also the update from 1.8.1, which moves to Program Files): visible console,
; the package download takes a while
Filename: "{app}\python\python.exe"; Parameters: """{app}\app\tools\setup_env.py"" --learn 1"; WorkingDir: "{app}\app"; StatusMsg: "패키지, 학습 기능, 모델을 준비하는 중 (처음엔 10~20분, 약 3GB)"; Check: FirstSetup and not IsUpdate
; the same from the app's update button: hidden (no console window over the progress)
Filename: "{app}\python\python.exe"; Parameters: """{app}\app\tools\setup_env.py"" --learn 1"; WorkingDir: "{app}\app"; StatusMsg: "패키지, 학습 기능, 모델을 준비하는 중 (처음엔 10~20분, 약 3GB)"; Flags: runhidden; Check: FirstSetup and IsUpdate
; update in place: hidden, packages only when they changed
Filename: "{app}\python\python.exe"; Parameters: """{app}\app\tools\setup_env.py"" --learn 1"; WorkingDir: "{app}\app"; StatusMsg: "업데이트 마무리 중"; Flags: runhidden; Check: not FirstSetup
Filename: "{app}\python\DIAP.exe"; Parameters: """{app}\app\app.py"""; WorkingDir: "{app}\app"; Description: "DIAP 실행"; Flags: nowait postinstall skipifsilent
Filename: "{app}\python\DIAP.exe"; Parameters: """{app}\app\app.py"""; WorkingDir: "{app}\app"; Flags: nowait runasoriginaluser; Check: IsUpdate

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\ffmpeg"
Type: files; Name: "{app}\setup_log.txt"

[Code]
// 1.8.1 was installed per user under %LOCALAPPDATA%\Programs\Contour: removed once the new copy is in place
// (projects, presets and learned data live in Documents and are not touched)
const
  OldKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{0432DE7E-AD09-4B89-8929-0B95959C384F}_is1';

function NoSlash(S: String): String;
begin
  Result := Lowercase(RemoveBackslashUnlessRoot(S));
end;

procedure RemoveOldPerUserInstall;
var
  Un, Loc: String;
  Code: Integer;
begin
  if not RegQueryStringValue(HKCU, OldKey, 'UninstallString', Un) then
    exit;
  if not RegQueryStringValue(HKCU, OldKey, 'InstallLocation', Loc) then
    Loc := ExtractFileDir(RemoveQuotes(Un));
  // never run the old uninstaller over this very install: it would delete python, app and ffmpeg in it
  if NoSlash(Loc) = NoSlash(ExpandConstant('{app}')) then
  begin
    RegDeleteKeyIncludingSubkeys(HKCU, OldKey);
    exit;
  end;
  Un := RemoveQuotes(Un);
  if FileExists(Un) then
    Exec(Un, '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

// 1.8.1's updater passes its own folder (/DIR=...\AppData\Local\Programs\Contour): DIAP goes to Program Files
procedure InitializeWizard;
var
  D: String;
begin
  D := Lowercase(WizardForm.DirEdit.Text);
  if (Pos(Lowercase(ExpandConstant('{localappdata}')), D) = 1) or (Pos('\appdata\local\programs\', D) > 0) then
    WizardForm.DirEdit.Text := ExpandConstant('{autopf}\Diap');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    RemoveOldPerUserInstall;
end;

// packages not installed in this folder yet (setup_env.py writes the stamp when they are)
function FirstSetup: Boolean;
begin
  Result := not FileExists(ExpandConstant('{app}\python\.contour_packages'));
end;

function IsUpdate: Boolean;
begin
  Result := ExpandConstant('{param:UPDATE|0}') = '1';
end;

