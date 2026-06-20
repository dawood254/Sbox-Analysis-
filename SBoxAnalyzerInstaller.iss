[Setup]
AppName=S-Box Cryptographic Analyzer
AppVersion=1.0
DefaultDirName={autopf}\SBoxAnalyzer
DefaultGroupName=S-Box Cryptographic Analyzer
OutputBaseFilename=SBoxAnalyzerSetup
OutputDir=dist
Compression=lzma
SolidCompression=yes
SetupIconFile=assets\app_icon.ico
UninstallDisplayIcon={app}\SBoxAnalyzer.exe

[Files]
Source: "dist\SBoxAnalyzer.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\S-Box Cryptographic Analyzer"; Filename: "{app}\SBoxAnalyzer.exe"
Name: "{commondesktop}\S-Box Cryptographic Analyzer"; Filename: "{app}\SBoxAnalyzer.exe"

[Run]
Filename: "{app}\SBoxAnalyzer.exe"; Description: "Launch S-Box Cryptographic Analyzer"; Flags: nowait postinstall skipifsilent
