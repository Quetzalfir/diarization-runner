# Diarize + Transcribe (WhisperX) Locally (Windows)

This project processes all videos inside `./videos/` and generates diarized `.txt` files in `./procesed/` with the same base name.

---

## Requirements

- Windows 10/11
- Python **3.12** (recommended)
- `ffmpeg` in `PATH`
- (Optional) NVIDIA GPU (e.g., RTX 4090) to speed up

---

## Setup (one-time per machine / per environment)

In PowerShell, from the **repo root**.

### 1) Allow running scripts (this session only)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 2) Run the installer

```powershell
.\setup.ps1
```

If you want CPU mode for any reason:

```powershell
.\setup.ps1 -Mode cpu
```

> Note: setup creates `.venv/` and installs dependencies.

---

## HuggingFace Token (for diarization)

Diarization uses `pyannote` models, so it requires a token and accepting terms.

### 1) Accept terms on HuggingFace

You must accept terms for these models:

- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-3.1`

### 2) Export the token in the SAME terminal where you will run the script

```powershell
$env:HUGGINGFACE_HUB_TOKEN="hf_TU_TOKEN_AQUI"
```

Optional (persistent on Windows):

```powershell
setx HUGGINGFACE_HUB_TOKEN "hf_TU_TOKEN_AQUI"
```

> Important: if you use `setx`, **close and reopen** the terminal for it to apply.

---

## Usage

1) Put your videos in:

```
./videos/
```

2) Run:

```powershell
.\.venv\Scripts\python.exe .\src\procesar_video_diarizado.py
```

---

## Output

- `./procesed/` is created
- For each video `videos\something.mp4` it generates:
    - `procesed\something.txt`

---

## Common Options (if your script supports them)

Change language/model/device:

```powershell
.\.venv\Scripts\python.exe .\src\procesar_video_diarizado.py --language es --model large-v2 --device cuda --compute_type float16
```

If you ever want to run **without diarization** (transcription only):

```powershell
.\.venv\Scripts\python.exe .\src\procesar_video_diarizado.py --no_diarize
```

---

## Notes / Troubleshooting

### "Red output in the console"
In Windows Terminal/PowerShell, many libraries print **warnings to stderr** and they appear in red.
As long as it ends with `exit code 0` and the summary says `OK`, it's usually fine.

### Torch / torchvision
This project **does not need `torchvision`**. If you have it installed and see weird errors, remove it:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torchvision
```

### Do I need to install everything every time I clone?
No. Typically:
- `setup.ps1` runs once per machine (or if you delete `.venv/`)
- after that you just run the script

Recommendation: **DO NOT** commit `.venv/` to the repo.

---

## Suggestion: .gitignore (optional)

Add this to your `.gitignore`:

```txt
.venv/
procesed/
__pycache__/
*.pyc
```
