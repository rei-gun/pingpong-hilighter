import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT = (PROJECT_ROOT / "prompt.txt").read_text(encoding="utf-8")

MAX_UPLOAD_BYTES = 1_900_000_000  # keep under Gemini's 2GB File API limit
REACTION_PAD_SEC = 2.0  # safety pad beyond the model's end (prompt already asks for reaction)

# Manual cue: shout "HIGHLIGHT" + a loud clap (usually together).
CUE_LOOKBACK_SEC = 18.0  # cover the rally that just happened before the cue
CUE_REACTION_SEC = 4.0
CUE_MERGE_SEC = 2.5  # clap + word within this window = one cue
CLAP_MIN_SEP_SEC = 1.0
WHISPER_MODEL_SIZE = "base"

_whisper_model = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def ensure_uploadable(source: Path, output_dir: Path) -> Path:
    """Return a path Gemini can accept (<= ~2GB). Compress if the source is too large."""
    if source.stat().st_size <= MAX_UPLOAD_BYTES:
        return source

    compressed = output_dir / f"{source.stem}_gemini_upload.mp4"
    if compressed.exists() and compressed.stat().st_size <= MAX_UPLOAD_BYTES:
        print(f"Reusing compressed upload file: {compressed}")
        return compressed

    print(
        f"Source is {source.stat().st_size / 1e9:.1f}GB (>2GB). "
        f"Compressing for Gemini upload -> {compressed.name}"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        "scale=-2:720",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(compressed),
    ]
    subprocess.run(cmd, check=True)
    print(f"Compressed size: {compressed.stat().st_size / 1e9:.2f}GB")
    return compressed


def extract_wav(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> Path:
    if wav_path.exists():
        return wav_path
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path


def load_wav_mono(wav_path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sample_rate


def detect_claps(audio: np.ndarray, sample_rate: int) -> List[float]:
    """Find short, loud transient peaks that look like a deliberate clap."""
    win = max(1, int(0.02 * sample_rate))  # 20ms
    hop = max(1, int(0.01 * sample_rate))  # 10ms
    if len(audio) < win * 4:
        return []

    rms = []
    for i in range(0, len(audio) - win, hop):
        chunk = audio[i : i + win]
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    rms = np.asarray(rms, dtype=np.float32)
    if rms.size == 0:
        return []

    baseline = float(np.median(rms))
    threshold = max(baseline * 8.0, float(np.percentile(rms, 95)) * 1.4, 0.12)

    # Claps are brief: require a sharp rise and quick drop.
    candidates = []
    min_sep_frames = int(CLAP_MIN_SEP_SEC / (hop / sample_rate))
    last_idx = -10**9
    for i in range(2, len(rms) - 2):
        if rms[i] < threshold:
            continue
        if rms[i] < rms[i - 1] or rms[i] < rms[i + 1]:
            continue
        rise = rms[i] / (rms[i - 2] + 1e-6)
        fall = rms[i] / (rms[i + 2] + 1e-6)
        if rise < 2.5 or fall < 1.8:
            continue
        if i - last_idx < min_sep_frames:
            # Keep the louder peak in a cluster.
            if candidates and rms[i] > rms[last_idx]:
                candidates[-1] = i * hop / sample_rate
                last_idx = i
            continue
        candidates.append(i * hop / sample_rate)
        last_idx = i

    return candidates


def detect_highlight_words(wav_path: Path) -> List[float]:
    """Return timestamps where Whisper hears the word HIGHLIGHT."""
    model = get_whisper_model()
    segments, _info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        language="en",
        vad_filter=True,
    )
    hits = []
    pattern = re.compile(r"high[\s\-]*light", re.IGNORECASE)
    for segment in segments:
        words = segment.words or []
        # Check individual words and adjacent pairs ("high" + "light").
        for i, word in enumerate(words):
            text = (word.word or "").strip()
            if pattern.search(text):
                hits.append(float(word.start))
                continue
            if i + 1 < len(words):
                pair = f"{text} {words[i + 1].word or ''}".strip()
                if pattern.search(pair):
                    hits.append(float(word.start))
    return hits


def merge_cue_times(word_times: List[float], clap_times: List[float]) -> List[dict]:
    """
    Build cue events. Prefer clap+word pairs within CUE_MERGE_SEC.
    Solo word or solo clap still counts (cue may be incomplete in noisy audio).
    """
    used_claps = set()
    used_words = set()
    cues = []

    for wi, wt in enumerate(word_times):
        best_cj = None
        best_dist = CUE_MERGE_SEC
        for cj, ct in enumerate(clap_times):
            if cj in used_claps:
                continue
            dist = abs(ct - wt)
            if dist <= best_dist:
                best_dist = dist
                best_cj = cj
        if best_cj is not None:
            ct = clap_times[best_cj]
            used_claps.add(best_cj)
            used_words.add(wi)
            cues.append(
                {
                    "time": min(wt, ct),
                    "kind": "highlight+clap",
                    "word": wt,
                    "clap": ct,
                }
            )

    for wi, wt in enumerate(word_times):
        if wi in used_words:
            continue
        cues.append({"time": wt, "kind": "highlight", "word": wt, "clap": None})

    for cj, ct in enumerate(clap_times):
        if cj in used_claps:
            continue
        cues.append({"time": ct, "kind": "clap", "word": None, "clap": ct})

    cues.sort(key=lambda c: c["time"])

    # Collapse cues that are still very close (e.g. double detection).
    merged = []
    for cue in cues:
        if merged and cue["time"] - merged[-1]["time"] <= CUE_MERGE_SEC:
            prev = merged[-1]
            kinds = {prev["kind"], cue["kind"]}
            if "highlight+clap" in kinds or kinds == {"highlight", "clap"}:
                prev["kind"] = "highlight+clap"
            prev["time"] = min(prev["time"], cue["time"])
            prev["word"] = prev["word"] if prev["word"] is not None else cue["word"]
            prev["clap"] = prev["clap"] if prev["clap"] is not None else cue["clap"]
        else:
            merged.append(dict(cue))
    return merged


def cues_to_clips(cues: List[dict], duration: Optional[float] = None) -> List[dict]:
    clips = []
    for cue in cues:
        start = max(0.0, cue["time"] - CUE_LOOKBACK_SEC)
        end = cue["time"] + CUE_REACTION_SEC
        if duration is not None:
            end = min(end, duration)
        clips.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "description": f"Manual cue ({cue['kind']}) at {cue['time']:.1f}s",
                "source": "cue",
                "cue": cue,
            }
        )
    return clips


def detect_audio_cues(video_path: Path, output_dir: Path) -> List[dict]:
    wav_path = output_dir / f"{video_path.stem}_audio_16k.wav"
    print("Extracting audio for cue detection...")
    extract_wav(video_path, wav_path)
    audio, sample_rate = load_wav_mono(wav_path)
    duration = len(audio) / float(sample_rate)

    print("Detecting loud claps...")
    clap_times = detect_claps(audio, sample_rate)
    print(f"  claps: {len(clap_times)} -> {[round(t, 1) for t in clap_times]}")

    print("Transcribing for HIGHLIGHT keyword...")
    word_times = detect_highlight_words(wav_path)
    print(f"  HIGHLIGHT: {len(word_times)} -> {[round(t, 1) for t in word_times]}")

    cues = merge_cue_times(word_times, clap_times)
    print(f"  merged cues: {len(cues)}")
    (output_dir / "cues.json").write_text(json.dumps(cues, indent=2))
    return cues_to_clips(cues, duration=duration)


def intervals_overlap(a: dict, b: dict, min_overlap_sec: float = 2.0) -> bool:
    start = max(float(a["start"]), float(b["start"]))
    end = min(float(a["end"]), float(b["end"]))
    return end - start >= min_overlap_sec


def merge_highlights(gemini_clips: List[dict], cue_clips: List[dict]) -> List[dict]:
    """Prefer cue clips; keep Gemini clips that don't substantially overlap a cue."""
    merged = []
    for clip in cue_clips:
        merged.append(
            {
                "start": float(clip["start"]),
                "end": float(clip["end"]),
                "description": clip["description"],
                "source": "cue",
            }
        )

    for clip in gemini_clips:
        candidate = {
            "start": float(clip["start"]),
            "end": float(clip["end"]),
            "description": clip.get("description", "Gemini highlight"),
            "source": "gemini",
        }
        if any(intervals_overlap(candidate, existing) for existing in merged):
            # Expand an overlapping cue/gemini window to cover both.
            for existing in merged:
                if intervals_overlap(candidate, existing):
                    existing["start"] = min(existing["start"], candidate["start"])
                    existing["end"] = max(existing["end"], candidate["end"])
                    if existing["source"] == "cue":
                        existing["description"] = (
                            f"{existing['description']} | also: {candidate['description']}"
                        )
                    else:
                        existing["description"] = candidate["description"]
                    break
            continue
        merged.append(candidate)

    merged.sort(key=lambda c: c["start"])
    return merged


def cut_clips(video_path: Path, output_dir: Path, highlights: List[dict]) -> None:
    for idx, clip in enumerate(highlights, 1):
        start_sec = float(clip["start"])
        end_sec = float(clip["end"]) + REACTION_PAD_SEC
        output_filename = output_dir / f"highlight_{idx:02d}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-to",
            str(end_sec),
            "-i",
            str(video_path),
            "-c",
            "copy",
            str(output_filename),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        source = clip.get("source", "?")
        print(
            f"Saved {output_filename.name} [{source}] ({clip['description']}) "
            f"[{start_sec}s -> {end_sec}s]"
        )


def parse_gemini_json(text: str) -> List[dict]:
    """Parse Gemini JSON output, tolerating fences / trailing junk."""
    if not text or not text.strip():
        return []
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if not match:
            print(f"WARNING: could not parse Gemini JSON, raw preview: {raw[:300]!r}")
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            print(f"WARNING: JSON repair failed ({exc}), raw preview: {raw[:300]!r}")
            return []
    if not isinstance(data, list):
        return []
    clips = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        clips.append(item)
    return clips


def generate_highlights(client: genai.Client, video_file, retries: int = 3) -> List[dict]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Analyzing video with Gemini Flash (attempt {attempt}/{retries})...")
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=[video_file, PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return parse_gemini_json(response.text or "")
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            retryable = any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if not retryable or attempt == retries:
                raise
            wait = 15 * attempt
            print(f"  transient error: {exc}; retrying in {wait}s...")
            time.sleep(wait)
    raise last_err  # pragma: no cover


def process_video(client: genai.Client, video_path: Path, project_root: Path) -> None:
    output_dir = project_root / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    highlights_path = output_dir / "highlights.json"
    if highlights_path.exists():
        print(f"\n=== {video_path.name} ===")
        print(f"Skipping (already has {highlights_path.name})")
        return

    print(f"\n=== {video_path.name} ===")
    cue_clips = detect_audio_cues(video_path, output_dir)

    upload_path = ensure_uploadable(video_path, output_dir)
    print(f"Uploading {upload_path.name} to Gemini File API...")
    video_file = client.files.upload(file=str(upload_path))

    while video_file.state.name == "PROCESSING":
        print("Processing video frames on Gemini server...")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise Exception(f"Video processing failed: {video_file.error.message}")

    print(f"Video active! URI: {video_file.uri}")

    gemini_clips = generate_highlights(client, video_file)
    print(f"Gemini found {len(gemini_clips)} highlights; cues found {len(cue_clips)}.")

    highlights = merge_highlights(gemini_clips, cue_clips)
    print(f"Merged total: {len(highlights)} highlights.")
    highlights_path.write_text(json.dumps(highlights, indent=2))

    cut_clips(video_path, output_dir, highlights)

    client.files.delete(name=video_file.name)
    print(f"Done! Clips in {output_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4> [more videos...]")
        sys.exit(1)

    project_root = PROJECT_ROOT
    client = genai.Client()

    ok, failed = [], []
    for arg in sys.argv[1:]:
        video_path = Path(arg).expanduser().resolve()
        if not video_path.exists():
            print(f"MISSING: {video_path}")
            failed.append(str(video_path))
            continue
        try:
            process_video(client, video_path, project_root)
            ok.append(video_path.name)
        except Exception as exc:
            print(f"FAILED {video_path.name}: {exc}")
            failed.append(video_path.name)

    print("\n========== BATCH SUMMARY ==========")
    print(f"Succeeded: {len(ok)}")
    print(f"Failed:    {len(failed)}")
    if failed:
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
