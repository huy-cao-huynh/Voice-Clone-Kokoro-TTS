# Environment setup (training and inference)

This project trains **adapter layers** on top of frozen [Kokoro](https://github.com/hexgrad/kokoro) and frozen WavLM-SV. You need a recent Python (3.10–3.13 per `kokoro/pyproject.toml`), PyTorch, torchaudio, Hugging Face libraries, and the patched `kokoro` package in this repo.

## AMD GPU (ROCm) — e.g. RX 7800 XT on WSL2

PyTorch’s ROCm build still exposes the GPU through the **`cuda`** device name (`torch.device("cuda")`, `autocast("cuda")`, `GradScaler("cuda")`). That matches this codebase: **no AMD-specific code paths are required** as long as ROCm PyTorch is installed and `torch.cuda.is_available()` is true.

1. **Use WSL2** (or native Linux) with ROCm installed per AMD’s current docs; training on native Windows with ROCm is not the typical path.
2. **Create a virtual environment** (recommended; many distros use PEP 668 and block `pip install --user` on the system interpreter):

   ```bash
   cd /path/to/Voice-Clone-Kokoro-TTS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install PyTorch + torchaudio** for your ROCm version from the [official PyTorch “Get Started”](https://pytorch.org/get-started/locally/) page (choose *Linux*, *Pip*, *ROCm*, and your ROCm release). Example shape (replace the index URL with the one PyTorch shows for your stack):

   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
   ```

   Keep `torch` and `torchaudio` on **matching** builds when possible.

4. **Install project requirements**:

   ```bash
   pip install -r requirements-training.txt
   ```

5. **Verify**:

   ```bash
   python scripts/verify_training_env.py
   ```

   You should see `Torch CUDA available: True` and your GPU name when the ROCm stack is visible inside WSL.

### Optional: Conda

If you prefer Conda/Mamba, create an env with Python 3.12, then still install **ROCm PyTorch via pip** inside that env using the same `--index-url` as above (Conda’s `pytorch` channel layout changes often; pip + PyTorch’s ROCm wheels is the most predictable for this repo).

## NVIDIA GPU (CUDA)

Follow [PyTorch “Get Started”](https://pytorch.org/get-started/locally/) for *CUDA*, then:

```bash
pip install -r requirements-training.txt
python scripts/verify_training_env.py
```

## CPU only (slow / debugging)

Install CPU wheels from PyTorch, then `pip install -r requirements-training.txt`.

## What gets installed

| Component | Role |
|-----------|------|
| `kokoro/` (editable `-e ./kokoro`) | Patched inference/training stack; pulls `misaki`, `loguru`, `numpy`, and a generic `torch` dependency — you should already have the correct `torch` from the step above. |
| `transformers`, `huggingface_hub` | WavLM-SV and checkpoint download. |
| `torchaudio` | I/O and resampling in `voice_clone` (installed with PyTorch). |

Audio loading uses **torchaudio** (no separate `soundfile` requirement).

## Running training

From the repo root, with `kokoro` importable (editable install handles this):

```bash
export PYTHONPATH="$(pwd)/kokoro:$(pwd)"
python -m voice_clone.train_adapters --help
```

Use `--amp` only if mixed precision is stable on your ROCm/CUDA stack; start without it when debugging.

## Pinning exact versions for reproducibility

After a working setup you can save `pip freeze` output locally or commit a freeze file if you need to match versions across machines (e.g. for a release tag).
