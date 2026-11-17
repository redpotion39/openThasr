import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

def download():
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    repo_id = "iapp/OpenThai2.0-ASR-Streaming-ONNX"

    files = [
        "encoder.int8.onnx",
        "encoder.int8.onnx.data",
        "decoder_joint.onnx",
        "tokens.txt",
    ]

    print(f"Downloading OpenThai 2.0 ASR models from {repo_id}...")
    for f in files:
        target = models_dir / f
        if target.exists():
            print(f" - {f} already exists.")
            continue
        print(f" - Downloading {f}...")
        hf_hub_download(repo_id=repo_id, filename=f, local_dir=str(models_dir))

    print("\nAll models ready in models/")

if __name__ == "__main__":
    download()
