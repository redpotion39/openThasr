from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mcp.server.mcpserver import MCPServer
from asr_engine import OpenThaiASR

server = MCPServer(
    name="openthai-asr",
    title="OpenThai 2.0 ASR Server",
    description="High-speed, high-accuracy Thai and English Speech-to-Text (ASR) server powered by OpenThai 2.0 ONNX.",
    version="1.0.0",
)

_asr_instance: Optional[OpenThaiASR] = None


def get_asr() -> OpenThaiASR:
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = OpenThaiASR(threads=4)
    return _asr_instance


@server.tool(
    name="thai_asr_file",
    title="Thai Speech-to-Text for Audio File",
    description="Transcribes Thai and English speech from an audio file (.wav, .flac, .ogg, .mp3, etc.) using OpenThai 2.0 ASR Streaming ONNX. Returns transcribed text and audio duration.",
)
def thai_asr_file(
    audio_path: str, lang: str = "th-TH", blank_bias: float = -1.0
) -> str:
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return f"Error: Audio file not found at {path}"

    try:
        asr = get_asr()
        result = asr.transcribe(path, lang=lang, blank_bias=blank_bias)
        text = result.get("text", "")
        duration = result.get("duration", 0.0)
        tokens_count = result.get("tokens_count", 0)

        if not text:
            return f"[No speech detected or output was silent] (Duration: {duration}s)"

        return (
            f"Transcription ({lang}):\n"
            f"{text}\n\n"
            f"[Duration: {duration}s, Tokens: {tokens_count}]"
        )
    except Exception as e:
        return f"Error processing audio file: {str(e)}"


@server.tool(
    name="thai_asr_base64",
    title="Thai Speech-to-Text for Base64 Audio",
    description="Transcribes Thai and English speech directly from a base64-encoded audio string (e.g. PCM / WAV / FLAC / MP3 / OGG).",
)
def thai_asr_base64(
    audio_base64: str, lang: str = "th-TH", blank_bias: float = -1.0
) -> str:
    try:
        # Strip data URL prefix if present (e.g., data:audio/wav;base64,...)
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",", 1)[1]

        data = base64.b64decode(audio_base64)
        asr = get_asr()
        result = asr.transcribe(data, lang=lang, blank_bias=blank_bias)
        text = result.get("text", "")
        duration = result.get("duration", 0.0)

        if not text:
            return f"[No speech detected] (Duration: {duration}s)"

        return f"Transcription ({lang}):\n{text}"
    except Exception as e:
        return f"Error processing base64 audio: {str(e)}"


@server.tool(
    name="thai_asr_info",
    title="Get OpenThai ASR Model Information",
    description="Returns metadata, supported languages, model architecture, and runtime settings of the OpenThai 2.0 ASR server.",
)
def thai_asr_info() -> str:
    return (
        "OpenThai 2.0 ASR Streaming Engine (ONNX INT8 / FP32)\n"
        "- Parameters: 0.64B (FastConformer 24 layers + RNN-T 2-layer LSTM)\n"
        "- Base Model: NVIDIA Nemotron 3.5 ASR Streaming 0.6B (Fine-tuned on 12,000+ hrs Thai speech by iApp & OpenThai)\n"
        "- Supported Languages: th-TH (Thai), th-mix (Thai mixed with English)\n"
        "- Input Format: 16 kHz mono (auto-resampled from WAV, MP3, FLAC, OGG)\n"
        "- Execution Provider: CPUExecutionProvider (Optimized for AVX2 / Intel DL Boost VNNI)\n"
        "- Latency Presets Supported: 80 ms, 320 ms, 560 ms (default), 1120 ms\n"
        "- License: OpenMDW 1.1"
    )


if __name__ == "__main__":
    server.run("stdio")
