from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa

SAMPLE_RATE = 16000
BLANK_ID = 14988

class OpenThaiASR:
    """High-speed Thai & English ASR Engine using OpenThai 2.0 Streaming ONNX."""

    def __init__(self, model_dir: Optional[Union[str, Path]] = None, threads: int = 4):
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parent / "models"
        else:
            self.model_dir = Path(model_dir).resolve()

        self.encoder_path = self.model_dir / "encoder.int8.onnx"
        if not self.encoder_path.exists():
            # Fallback to fp32 if int8 is not found
            self.encoder_path = self.model_dir / "encoder.onnx"

        self.decoder_path = self.model_dir / "decoder_joint.onnx"
        self.tokens_path = self.model_dir / "tokens.txt"

        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found in {self.model_dir}")
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Decoder model not found at {self.decoder_path}")
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens vocabulary not found at {self.tokens_path}")

        # Setup ONNX Runtime sessions
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.encoder_session = ort.InferenceSession(
            str(self.encoder_path), so, providers=["CPUExecutionProvider"]
        )
        self.decoder_session = ort.InferenceSession(
            str(self.decoder_path), so, providers=["CPUExecutionProvider"]
        )

        # Load tokens
        self.tokens = []
        with open(self.tokens_path, "r", encoding="utf-8") as f:
            for line in f:
                self.tokens.append(line.rstrip("\r\n"))

        # Precompute Mel Filterbank
        self.mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=512,
            n_mels=128,
            fmin=0.0,
            fmax=8000.0,
            htk=False,
            norm="slaney",
        )

    def load_audio(self, source: Any) -> np.ndarray:
        """Load audio from path, bytes, or array into float32 16 kHz mono."""
        if isinstance(source, np.ndarray):
            data = source.astype(np.float32)
            sr = SAMPLE_RATE
        elif isinstance(source, (bytes, bytearray)):
            data, sr = sf.read(io.BytesIO(source), dtype="float32", always_2d=True)
        else:
            path = Path(source).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)

        if data.ndim == 2:
            data = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)

        if sr != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

        return np.ascontiguousarray(data, dtype=np.float32)

    def compute_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract 128 log-mel filterbank features (25ms Hann, 10ms hop, pre-emphasis 0.97)."""
        if len(audio) == 0:
            return np.zeros((1, 128, 0), dtype=np.float32)

        # Pre-emphasis filter: y[t] = x[t] - 0.97 * x[t-1]
        emphasized = np.empty_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - 0.97 * audio[:-1]

        # STFT with Hann window, 25ms (400 samples), 10ms hop (160 samples), n_fft=512
        stft = librosa.stft(
            emphasized,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
        mag = np.abs(stft) ** 2
        mel = np.dot(self.mel_basis, mag)
        log_mel = np.log(mel + 1e-5).astype(np.float32)

        # Output shape: [1, 128, T]
        return np.expand_dims(log_mel, axis=0)

    def decode_tokens_to_text(self, token_ids: list[int]) -> str:
        """Convert token IDs to Thai / English text, normalizing space and vowels."""
        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self.tokens):
                t = self.tokens[tid]
                if t in ("<unk>", "<bg-BG>"):
                    continue
                pieces.append(t)

        raw_text = "".join(pieces).replace("▁", " ")
        # SARA AM decomposition normalizer: U+0E4D + U+0E32 -> U+0E33
        normalized = raw_text.replace("ํา", "ำ").strip()
        return normalized

    def _greedy_decode(
        self,
        enc_out: np.ndarray,
        blank_bias: float = -1.0,
        max_symbols_per_step: int = 2,
    ) -> list[int]:
        """Greedy RNN-T decode over acoustic frames."""
        t_enc = enc_out.shape[1]
        targets = np.array([[BLANK_ID]], dtype=np.int32)
        target_len = np.array([1], dtype=np.int32)
        s1 = np.zeros((2, 1, 640), dtype=np.float32)
        s2 = np.zeros((2, 1, 640), dtype=np.float32)

        emitted_tokens: list[int] = []

        for t in range(t_enc):
            enc_step = enc_out[:, t : t + 1, :].transpose(0, 2, 1)  # [1, 1024, 1]
            for _ in range(max_symbols_per_step):
                outputs = self.decoder_session.run(
                    None,
                    {
                        "encoder_outputs": enc_step,
                        "targets": targets,
                        "target_length": target_len,
                        "input_states_1": s1,
                        "input_states_2": s2,
                    },
                )
                logits = outputs[0]  # [1, 1, 1, 14989]
                logits[0, 0, 0, BLANK_ID] += blank_bias
                pred_token = int(np.argmax(logits[0, 0, 0, :]))

                if pred_token == BLANK_ID:
                    break

                emitted_tokens.append(pred_token)
                targets = np.array([[pred_token]], dtype=np.int32)
                s1 = outputs[2]
                s2 = outputs[3]

        return emitted_tokens

    def transcribe(
        self,
        audio_source: Any,
        lang: str = "th-TH",
        blank_bias: float = -1.0,
        max_symbols_per_step: int = 2,
    ) -> dict[str, Any]:
        """Transcribe an audio input into text.
        
        Args:
            audio_source: Path to file, bytes, or float32 16kHz numpy array.
            lang: 'th-TH' (prompt_index 32) or 'th-mix' (prompt_index 63).
            blank_bias: Logit bias added to the blank symbol (-1.0 standard, -2.5 short clip).
            max_symbols_per_step: Max symbols emitted per acoustic frame.
        """
        audio = self.load_audio(audio_source)
        duration = len(audio) / SAMPLE_RATE

        if len(audio) == 0:
            return {"text": "", "duration": 0.0, "tokens_count": 0}

        features = self.compute_features(audio)
        t_frames = features.shape[2]

        prompt_idx = 32 if lang == "th-TH" else 63

        # Run Encoder
        enc_out, _ = self.encoder_session.run(
            None,
            {
                "audio_signal": features,
                "length": np.array([t_frames], dtype=np.int64),
                "prompt_index": np.array([prompt_idx], dtype=np.int64),
            },
        )

        # Run Decoder
        tokens = self._greedy_decode(
            enc_out, blank_bias=blank_bias, max_symbols_per_step=max_symbols_per_step
        )
        text = self.decode_tokens_to_text(tokens)

        # If clip is short and yielded empty output, retry with higher sensitivity (more negative bias)
        if not text and duration < 3.0 and blank_bias > -3.0:
            tokens = self._greedy_decode(
                enc_out, blank_bias=-3.0, max_symbols_per_step=max_symbols_per_step
            )
            text = self.decode_tokens_to_text(tokens)

        return {
            "text": text,
            "duration": round(duration, 3),
            "tokens_count": len(tokens),
            "lang": lang,
        }
