# Run mywebtools directly on Windows (same PC as Duix: no SMB share needed).
# Needs Python 3.10+ (py launcher), ffmpeg on PATH (winget install Gyan.FFmpeg) and, for OCR,
# Tesseract (https://github.com/UB-Mannheim/tesseract/wiki; add it to PATH or set TESSERACT_CMD in .env).
param([switch]$WithWhisper)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

if (-not (Test-Path ".venv")) { py -3 -m venv .venv }
.\.venv\Scripts\python -m pip install --upgrade pip | Out-Null
.\.venv\Scripts\python -m pip install -r requirements.txt
if ($WithWhisper) { .\.venv\Scripts\python -m pip install -r requirements-optional.txt }
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

# On Windows the Duix folder (C:\AI\Duix\data\face2face\temp) is used directly.
.\.venv\Scripts\python -m mywebtools
