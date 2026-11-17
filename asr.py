import argparse
import sys
import time
from pathlib import Path

from asr_engine import OpenThaiASR


def main():
    parser = argparse.ArgumentParser(
        description="OpenThai 2.0 ASR CLI - Fast Speech-to-Text on CPU"
    )
    parser.add_argument("audio", help="Path to input audio file (.wav, .mp3, .flac, .ogg, etc.)")
    parser.add_argument(
        "--lang",
        default="th-TH",
        choices=["th-TH", "th-mix"],
        help="Language mode: 'th-TH' for Thai, 'th-mix' for mixed Thai-English (default: th-TH)",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=-1.0,
        help="Blank logit bias (default: -1.0, use -2.5 for shorter clips)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads to use for inference (default: 4)",
    )
    parser.add_argument(
        "-o", "--out",
        help="Output text file path to save transcription",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading OpenThai 2.0 ASR model (threads={args.threads})...")
    t0 = time.time()
    engine = OpenThaiASR(threads=args.threads)
    load_time = time.time() - t0
    print(f"Model ready in {load_time:.2f}s")

    print(f"Transcribing {audio_path.name}...")
    t1 = time.time()
    result = engine.transcribe(audio_path, lang=args.lang, blank_bias=args.bias)
    proc_time = time.time() - t1

    text = result["text"]
    dur = result["duration"]
    rtf = proc_time / dur if dur > 0 else 0.0

    print("\n" + "=" * 50)
    print("Transcription Result:")
    print("=" * 50)
    print(text if text else "[No speech detected]")
    print("=" * 50)
    print(
        f"Audio Duration: {dur:.2f}s | Processed in: {proc_time:.2f}s | RTF: {rtf:.3f}x"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved result to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
