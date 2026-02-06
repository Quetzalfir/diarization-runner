param(
  [ValidateSet("cuda", "cpu")]
  [string]$Mode = "cuda",

  # For CUDA, what works today with torch 2.8 is usually cu126.
  # If torch ends up CPU or fails, try: .\setup.ps1 -Mode cuda -CudaIndex cu124
  [string]$CudaIndex = "cu126"
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function Run-Cmd {
  param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [Parameter(Mandatory=$true)][string[]]$Args
  )
  Write-Host ">> $Exe $($Args -join ' ')" -ForegroundColor Cyan
  & $Exe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed ($LASTEXITCODE): $Exe $($Args -join ' ')"
  }
}

function Get-PythonExeForVenv {
  # Prefer the "py" launcher to force 3.12
  $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
  if ($pyLauncher) {
    return @("py", @("-3.12"))
  }

  $python = Get-Command python -ErrorAction SilentlyContinue
  if (-not $python) {
    throw "Couldn't find 'py' or 'python'. Install Python 3.12 and try again."
  }

  # Validate 3.12
  $ver = & python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
  if ($ver -ne "3.12") {
    throw "Your 'python' is $ver. To avoid issues with WhisperX/torch, use Python 3.12. Install it and use 'py -3.12'."
  }

  return @("python", @())
}

# 1) Create venv if it doesn't exist
if (!(Test-Path ".venv")) {
  $pyInfo = Get-PythonExeForVenv
  $baseExe = $pyInfo[0]
  $baseArgs = $pyInfo[1]

  $args = @()
  $args += $baseArgs
  $args += @("-m", "venv", ".venv")
  Run-Cmd $baseExe $args
}

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# 2) Upgrade base tools
Run-Cmd $Py @("-m", "pip", "install", "-U", "pip", "setuptools", "wheel")

# 3) Preventive cleanup (avoid the famous torch/torchvision mismatch bug)
#    torchvision is NOT required for whisperx (audio), and sometimes breaks due to incompatibilities.
Run-Cmd $Py @("-m", "pip", "uninstall", "-y", "torchvision")

# 4) Install torch/torchaudio (CUDA or CPU)
if ($Mode -eq "cuda") {
  $torchIndex = "https://download.pytorch.org/whl/$CudaIndex"
  Write-Host "Installing CUDA torch/torchaudio from: $torchIndex" -ForegroundColor Yellow
  Run-Cmd $Py @("-m", "pip", "install", "--index-url", $torchIndex, "torch==2.8.0", "torchaudio==2.8.0")
} else {
  Write-Host "Installing CPU torch/torchaudio (PyPI)..." -ForegroundColor Yellow
  Run-Cmd $Py @("-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0")
}

# 5) Install requirements (whisperx + deps)
Run-Cmd $Py @("-m", "pip", "install", "-r", "requirements.txt")

# 6) Quick check (CUDA)
Write-Host "`n=== Torch/CUDA Check ===" -ForegroundColor Green
Run-Cmd $Py @("-c", "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)")

# 7) ffmpeg check
Write-Host "`n=== ffmpeg Check ===" -ForegroundColor Green
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
  Write-Host "WARNING: Couldn't find ffmpeg in PATH." -ForegroundColor Yellow
  Write-Host "Install it with:" -ForegroundColor Yellow
  Write-Host "  winget install -e --id Gyan.FFmpeg" -ForegroundColor Yellow
  Write-Host "Then close/reopen the terminal and try again." -ForegroundColor Yellow
} else {
  Run-Cmd "ffmpeg" @("-version")
}

# 8) Expected folders
if (!(Test-Path "videos")) {
  New-Item -ItemType Directory -Force "videos" | Out-Null
}

Write-Host "`nDone ✅" -ForegroundColor Green
Write-Host "Next:" -ForegroundColor Green
Write-Host "1) Make sure you have videos in .\videos\" -ForegroundColor Green
Write-Host "2) Set the HF token in the terminal (see README)" -ForegroundColor Green
Write-Host "3) Run:" -ForegroundColor Green
Write-Host "   .\.venv\Scripts\python.exe .\src\procesar_video_diarizado.py" -ForegroundColor Green
