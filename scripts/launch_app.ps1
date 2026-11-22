$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot '.cache/ultralytics'
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache/matplotlib'
$env:TEMP = Join-Path $projectRoot '.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& (Join-Path $projectRoot '.venv/Scripts/python.exe') -m streamlit run app/main.py --server.address 127.0.0.1 --browser.gatherUsageStats false
