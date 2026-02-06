import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MEDIA_EXTS = {
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv",
    ".mp3", ".m4a", ".wav", ".flac", ".ogg"
}


def die(msg: str, code: int = 1) -> None:
    print(msg)
    raise SystemExit(code)


def mask_cmd(cmd: List[str]) -> List[str]:
    safe: List[str] = []
    skip_next = False
    for c in cmd:
        if skip_next:
            safe.append("hf_***REDACTED***")
            skip_next = False
            continue
        if c == "--hf_token":
            safe.append(c)
            skip_next = True
            continue
        safe.append(c)
    return safe


def run_cmd(cmd: List[str]) -> None:
    safe_cmd = mask_cmd(cmd)
    print("\n>> " + " ".join(safe_cmd))

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.stdout:
        print(p.stdout)

    if p.returncode != 0:
        die(f"ERROR: command failed with exit code {p.returncode}\nCommand: {' '.join(safe_cmd)}")


def check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        die(
            "ERROR: Couldn't find ffmpeg in PATH.\n"
            "Install it with:\n"
            "  winget install -e --id Gyan.FFmpeg\n"
            "Then open a new terminal and try again."
        )
    except subprocess.CalledProcessError:
        pass


def sec_to_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def extract_audio_to_wav(video_path: Path, wav_path: Path) -> None:
    # WAV 16kHz mono (ideal for ASR/diarization)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    run_cmd(cmd)


def find_output_files(work_dir: Path, base_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    json_candidate = work_dir / f"{base_name}.json"
    srt_candidate = work_dir / f"{base_name}.srt"

    json_path = json_candidate if json_candidate.exists() else None
    srt_path = srt_candidate if srt_candidate.exists() else None

    # Fallback: if the name doesn't match for some reason, take the most recent
    if json_path is None:
        json_files = list(work_dir.glob("*.json"))
        if json_files:
            json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            json_path = json_files[0]

    if srt_path is None:
        srt_files = list(work_dir.glob("*.srt"))
        if srt_files:
            srt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            srt_path = srt_files[0]

    return json_path, srt_path


def get_segments_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = data.get("segments")
    if isinstance(segs, list):
        return segs
    segs = data.get("chunks")
    if isinstance(segs, list):
        return segs
    return []


def get_speaker(seg: Dict[str, Any]) -> str:
    for k in ("speaker", "speaker_id", "speaker_label"):
        if k in seg and seg[k] is not None:
            return str(seg[k])
    return "SPEAKER_UNKNOWN"


def merge_segments(segments: List[Dict[str, Any]], merge_gap: float) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        speaker = get_speaker(seg)

        if merged:
            prev = merged[-1]
            same_speaker = prev["speaker"] == speaker
            close_enough = (start - prev["end"]) <= merge_gap
            if same_speaker and close_enough:
                prev["end"] = max(prev["end"], end)
                prev["text"] = (prev["text"] + " " + text).strip()
                continue

        merged.append({"start": start, "end": end, "speaker": speaker, "text": text})

    return merged


def parse_srt_time_to_seconds(t: str) -> float:
    # "HH:MM:SS,mmm" -> seconds
    # e.g. "00:01:02,345"
    hh, mm, rest = t.split(":")
    ss, mmm = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(mmm) / 1000.0


def parse_srt(srt_path: Path) -> List[Dict[str, Any]]:
    # Simple SRT parser
    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out: List[Dict[str, Any]] = []

    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        # lines[0] = index
        # lines[1] = time
        time_line = lines[1]
        if "-->" not in time_line:
            continue

        left, right = [x.strip() for x in time_line.split("-->")]
        start = parse_srt_time_to_seconds(left)
        end = parse_srt_time_to_seconds(right)

        body = " ".join(lines[2:]).strip()

        # WhisperX usually puts "SPEAKER_00: text"
        speaker = "SPEAKER_UNKNOWN"
        txt = body
        if ":" in body:
            possible = body.split(":", 1)[0].strip()
            if possible.upper().startswith("SPEAKER_"):
                speaker = possible
                txt = body.split(":", 1)[1].strip()

        out.append({"start": start, "end": end, "speaker": speaker, "text": txt})

    return out

def run_whisperx(wav_path: Path, work_dir: Path, args: argparse.Namespace) -> None:
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        die(
            "ERROR: Couldn't find HUGGINGFACE_HUB_TOKEN.\n"
            "In PowerShell set it like this:\n"
            "  $env:HUGGINGFACE_HUB_TOKEN=\"hf_...\"\n"
            "You also must accept terms for:\n"
            "  pyannote/segmentation-3.0\n"
            "  pyannote/speaker-diarization-3.1"
        )

    # Bootstrap: force torch.load(weights_only=False) to avoid PyTorch 2.6+ blocking
    bootstrap = """
import torch

_real_load = torch.load

def _patched_load(*args, **kwargs):
    # Force compatibility with older checkpoints (pyannote/omegaconf)
    kwargs["weights_only"] = False
    return _real_load(*args, **kwargs)

torch.load = _patched_load

import whisperx.__main__ as m
m.cli()
""".strip()

    cmd = [
        sys.executable, "-c", bootstrap,
        str(wav_path),
        "--language", args.language,
        "--model", args.model,
        "--device", args.device,
        "--compute_type", args.compute_type,
        "--output_dir", str(work_dir),
    ]

    if args.diarize:
        cmd += [
            "--diarize",
            "--min_speakers", str(args.min_speakers),
            "--max_speakers", str(args.max_speakers),
            "--hf_token", token,
        ]

    run_cmd(cmd)


def write_diarized_txt(
        out_txt: Path,
        video_rel: Path,
        segments: List[Dict[str, Any]],
        meta: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append(f"File: {video_rel.as_posix()}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"WhisperX: model={meta['model']} language={meta['language']} "
        f"device={meta['device']} compute_type={meta['compute_type']} "
        f"diarize={meta['diarize']} min_speakers={meta['min_speakers']} max_speakers={meta['max_speakers']}"
    )
    lines.append("")
    lines.append("TRANSCRIPT")
    lines.append("")

    for s in segments:
        start_ts = sec_to_ts(s["start"])
        end_ts = sec_to_ts(s["end"])
        speaker = s.get("speaker", "SPEAKER_UNKNOWN")
        lines.append(f"[{start_ts} --> {end_ts}] {speaker}")
        lines.append(str(s.get("text", "")).strip())
        lines.append("")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def iter_media_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            files.append(p)
    files.sort()
    return files


def process_one(video_path: Path, input_root: Path, output_root: Path, args: argparse.Namespace) -> Tuple[bool, str]:
    rel = video_path.relative_to(input_root)
    out_txt = (output_root / rel).with_suffix(".txt")

    # Temp folder per file (inside procesed/_tmp/...)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = output_root / "_tmp" / rel.parent / f"{rel.stem}_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    wav_path = work_dir / f"{rel.stem}.wav"

    try:
        extract_audio_to_wav(video_path, wav_path)
        run_whisperx(wav_path, work_dir, args)

        json_path, srt_path = find_output_files(work_dir, rel.stem)

        segments: List[Dict[str, Any]] = []
        if json_path and json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
            raw_segments = get_segments_from_json(data)
            if raw_segments:
                segments = merge_segments(raw_segments, merge_gap=args.merge_gap)

        # Fallback to SRT if JSON doesn't have usable segments
        if not segments and srt_path and srt_path.exists():
            segments = merge_segments(parse_srt(srt_path), merge_gap=args.merge_gap)

        if not segments:
            return False, f"{rel.as_posix()} -> ERROR: could not get segments (JSON/SRT)."

        meta = {
            "model": args.model,
            "language": args.language,
            "device": args.device,
            "compute_type": args.compute_type,
            "diarize": args.diarize,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
        }
        write_diarized_txt(out_txt, rel, segments, meta)
        return True, f"{rel.as_posix()} -> OK: {out_txt.relative_to(output_root).as_posix()}"

    except Exception as e:
        return False, f"{rel.as_posix()} -> ERROR: {e}"

    finally:
        if args.keep_temp:
            print(f"Temp files kept at: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Processes ALL videos inside ./videos and saves diarized .txt files in ./procesed"
    )
    parser.add_argument("--language", default="es")
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--min_speakers", type=int, default=2)
    parser.add_argument("--max_speakers", type=int, default=2)
    parser.add_argument("--merge_gap", type=float, default=0.8)
    parser.add_argument("--keep_temp", action="store_true")

    # Diarize by default (as requested).
    # If you ever want transcription only without a token: --no_diarize
    parser.add_argument("--no_diarize", action="store_true", help="Disable diarization (transcribe only)")
    args = parser.parse_args()
    args.diarize = not args.no_diarize

    check_ffmpeg()

    project_root = Path.cwd().resolve()
    input_root = project_root / "videos"
    output_root = project_root / "procesed"

    if not input_root.exists() or not input_root.is_dir():
        die(f"ERROR: Folder does not exist: {input_root}\nCreate 'videos' in the project and put your videos there.")

    output_root.mkdir(parents=True, exist_ok=True)

    media_files = iter_media_files(input_root)
    if not media_files:
        die(f"Couldn't find videos/audio in: {input_root}\nSupported extensions: {sorted(MEDIA_EXTS)}")

    print(f"Project:   {project_root}")
    print(f"Input:     {input_root}")
    print(f"Output:    {output_root}")
    print(f"Files:     {len(media_files)}")
    print("")

    ok = 0
    fail = 0
    for f in media_files:
        print(f"\n=== Processing: {f.relative_to(input_root).as_posix()} ===")
        success, msg = process_one(f, input_root, output_root, args)
        print(msg)
        if success:
            ok += 1
        else:
            fail += 1

    print("\n==== SUMMARY ====")
    print(f"OK:   {ok}")
    print(f"FAIL: {fail}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
