import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")

MAX_UPLOAD_BYTES = 1_900_000_000  # keep under Gemini's 2GB File API limit
REACTION_PAD_SEC = 2.0  # safety pad beyond the model's end (prompt already asks for reaction)

PROMPT = """
You are an extremely picky table tennis highlight editor.
You are building a highlight reel — not a point catalog.

NO MINIMUM. NO MAXIMUM.
- There is no quota, target, or expected range of clips per video.
- If nothing notable happens, return an empty list: []
- Returning zero clips for a boring video is correct and preferred.
- Do not invent, stretch, or include mediocre points just to have output.
- Do not stop selecting good moments because you already have "enough."
- Clip count must only reflect what is actually in the footage.

ALWAYS INCLUDE (mandatory):
- Any rally with 8 or more ball exchanges (count each hit by either player).
  Clip the FULL rally from serve/first exchange through the point end + reaction.
  Long rallies are automatic highlights even without a flashy finish.

SHORT RALLIES (< 8 exchanges) — INCLUDE ONLY if there is an amazing moment:
- Exceptionally skillful winner (sharp angle, heavy spin finish, smash off a tough ball)
- Spectacular defensive save or counter under real pressure
- Clear lucky/net/edge moment that decides or nearly decides the point
- A single "wow" shot so good it stands alone

If a short rally is merely decent, fast, or "pretty good club play" — SKIP IT.

EXCLUDE:
- Routine serves + weak returns that die immediately
- Soft push exchanges with no attack or drama
- Unforced errors on easy balls with no build-up
- Dead time: pauses, ball pickup, towel, coaching, between-point idle
- Average points that look like every other club match point
- Anything included only so the video "has some clips"

Clip timing:
- start: a beat before the interesting action (serve that starts the highlight is fine)
- end: NEVER cut on the final bounce — keep 3–5 seconds of reaction after the point
- Length follows content: crazy single shot = short clip; 8+ rally = longer clip
- Do NOT homogenize clip lengths

In each description, briefly say WHY (e.g. "10-shot rally", "net cord winner",
"diving save then counter"). For long rallies, state the approximate exchange count.

Return strictly a JSON list. Empty [] is a valid, successful answer when nothing qualifies:
[
  {
    "start": 14.5,
    "end": 28.0,
    "description": "Why this is highlight-worthy"
  }
]
"""


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


def process_video(client: genai.Client, video_path: Path, project_root: Path) -> None:
    output_dir = project_root / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    highlights_path = output_dir / "highlights.json"
    if highlights_path.exists():
        print(f"\n=== {video_path.name} ===")
        print(f"Skipping (already has {highlights_path.name})")
        return

    upload_path = ensure_uploadable(video_path, output_dir)

    print(f"\n=== {video_path.name} ===")
    print(f"Uploading {upload_path.name} to Gemini File API...")
    video_file = client.files.upload(file=str(upload_path))

    while video_file.state.name == "PROCESSING":
        print("Processing video frames on Gemini server...")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise Exception(f"Video processing failed: {video_file.error.message}")

    print(f"Video active! URI: {video_file.uri}")

    print("Analyzing video with Gemini Flash...")
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[video_file, PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    highlights = json.loads(response.text)
    print(f"Found {len(highlights)} highlights.")
    (output_dir / "highlights.json").write_text(json.dumps(highlights, indent=2))

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
        print(
            f"Saved {output_filename.name} ({clip['description']}) "
            f"[{start_sec}s -> {end_sec}s]"
        )

    client.files.delete(name=video_file.name)
    print(f"Done! Clips in {output_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4> [more videos...]")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent
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
