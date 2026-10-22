# Install the mywebtools model services on the Windows PC with the graphics card.
# Each service gets its own Python environment (their dependencies conflict):
#   whisper  -> services\.venv-whisper        (faster-whisper + CUDA 12 / cuDNN 9 wheels)
#   aligner  -> services\.venv-aligner        (torch CUDA + transformers + uroman)
#   indextts -> the IndexTTS checkout's own environment (uv sync in $IndexTTSRepo)
#
#   powershell -ExecutionPolicy Bypass -File services\install.ps1                 # all three
#   powershell -ExecutionPolicy Bypass -File services\install.ps1 -Only whisper   # just one
#
# Needs: uv (winget install astral-sh.uv), git, an NVIDIA driver. Run elevated once to add the firewall rule.
param(
    [ValidateSet("whisper", "aligner", "indextts")][string[]]$Only = @("whisper", "aligner", "indextts"),
    [string]$IndexTTSRepo = "C:\AI\index-tts",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [string]$Python = "3.11"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install it with:  winget install astral-sh.uv   (then open a new PowerShell)"
}

function New-ServiceVenv([string]$name) {
    $venv = Join-Path $here ".venv-$name"
    if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) { uv venv $venv --python $Python }
    return (Join-Path $venv "Scripts\python.exe")
}

if ($Only -contains "whisper") {
    Write-Host "==> whisper: faster-whisper service"
    $py = New-ServiceVenv "whisper"
    uv pip install --python $py -r (Join-Path $here "whisper\requirements.txt")
}

if ($Only -contains "aligner") {
    Write-Host "==> aligner: MMS forced-alignment service"
    $py = New-ServiceVenv "aligner"
    uv pip install --python $py torch --index-url $TorchIndex
    uv pip install --python $py -r (Join-Path $here "aligner\requirements.txt")
}

if ($Only -contains "indextts") {
    Write-Host "==> indextts: IndexTTS 2.5 service (environment of $IndexTTSRepo)"
    if (-not (Test-Path (Join-Path $IndexTTSRepo "indextts\infer_v2_5.py"))) {
        if (Test-Path $IndexTTSRepo) { git -C $IndexTTSRepo pull } else {
            New-Item -ItemType Directory -Force -Path (Split-Path $IndexTTSRepo) | Out-Null
            git clone https://github.com/index-tts/index-tts.git $IndexTTSRepo
        }
    }
    Push-Location $IndexTTSRepo
    try { uv sync } finally { Pop-Location }  # the model folder is passed by path: no checkpoints\ download needed
}

Write-Host "==> Firewall: allow the board to reach TCP 8391-8393 on private networks"
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if ($admin) {
    if (-not (Get-NetFirewallRule -DisplayName "mywebtools model services" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName "mywebtools model services" -Direction Inbound -Protocol TCP `
            -LocalPort 8391-8393 -Action Allow -Profile Private | Out-Null
    }
} else {
    Write-Warning "Not elevated: skipped the firewall rule (only needed when the board calls these services)."
}

Write-Host ""
Write-Host "Done. Start the services with:  powershell -ExecutionPolicy Bypass -File services\start.ps1"
Write-Host "Check them in mywebtools: Admin > Model services."
