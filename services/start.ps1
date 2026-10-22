# Start the mywebtools model services (one PowerShell window each, or -Here for one in this window).
#
#   powershell -ExecutionPolicy Bypass -File services\start.ps1
#   powershell -ExecutionPolicy Bypass -File services\start.ps1 -Only whisper -Here
#
# Models default to %USERPROFILE%\Models\<folder>; mywebtools sends the folders from
# Admin > Model services with every request, and they must be inside -ModelsRoot.
param(
    [ValidateSet("whisper", "aligner", "indextts")][string[]]$Only = @("whisper", "aligner", "indextts"),
    [string]$IndexTTSRepo = "C:\AI\index-tts",
    [string]$ModelsRoot = (Join-Path $env:USERPROFILE "Models"),
    [string]$BindHost = "0.0.0.0",          # 127.0.0.1 = only this PC
    [string]$Token = $env:MWT_SERVICE_TOKEN, # same value as NODE_TOKEN in mywebtools (optional)
    [int]$IdleUnload = 900,                  # free GPU memory after this many idle seconds
    [switch]$Here
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$common = @("--host", $BindHost, "--models-root", "`"$ModelsRoot`"", "--idle-unload", $IdleUnload)
if ($Token) { $common += @("--token", $Token) }

$commands = @{
    whisper  = "& `"$here\.venv-whisper\Scripts\python.exe`" `"$here\whisper\server.py`" $common"
    aligner  = "& `"$here\.venv-aligner\Scripts\python.exe`" `"$here\aligner\server.py`" $common"
    indextts = "uv run --project `"$IndexTTSRepo`" --with fastapi --with uvicorn --with python-multipart " +
               "python `"$here\indextts\server.py`" --repo `"$IndexTTSRepo`" $common"
}

foreach ($name in $Only) {
    $cmd = $commands[$name]
    if ($Here) {
        Write-Host "==> $name"
        Invoke-Expression $cmd
        return
    }
    Write-Host "==> starting $name in a new window"
    Start-Process powershell -ArgumentList @("-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='mywebtools $name'; $cmd")
}
