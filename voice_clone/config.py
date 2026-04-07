"""Training hyperparameters and Kokoro config loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from huggingface_hub import hf_hub_download


@dataclass
class LossWeights:
    lambda_mel: float = 1.0
    lambda_spk: float = 1.0
    lambda_slm: float = 0.05


@dataclass
class MelLossConfig:
    """Log-mel reconstruction at Kokoro output rate (24 kHz); `n_mels` comes from Kokoro `config.json`."""

    sample_rate: int = 24_000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 0.0
    f_max: Optional[float] = None


@dataclass
class SLMDiscriminatorConfig:
    hidden_channels: int = 256
    num_layers: int = 4
    kernel_size: int = 7
    use_spectral_norm: bool = True


@dataclass
class TrainConfig:
    kokoro_repo_id: str = "hexgrad/Kokoro-82M"
    wavlm_model_id: str = "microsoft/wavlm-base-plus-sv"
    adapter_bottleneck: int = 64
    loss_weights: LossWeights = field(default_factory=LossWeights)
    mel: MelLossConfig = field(default_factory=MelLossConfig)
    slm_disc: SLMDiscriminatorConfig = field(default_factory=SLMDiscriminatorConfig)
    lr_g: float = 1e-4
    lr_d: float = 5e-5
    weight_decay_g: float = 0.0
    weight_decay_d: float = 0.0
    grad_clip_g: Optional[float] = 1.0
    grad_clip_d: Optional[float] = 1.0
    use_amp: bool = False
    # How often to log to W&B and refresh training metrics in the terminal (tqdm postfix or one-line \r).
    log_interval: int = 1
    checkpoint_interval: int = 1000
    warmup_steps: int = 50
    grad_accum_steps: int = 8
    disc_start_step: int = 500
    slm_d_steps_per_g_step: int = 1
    speed: float = 1.0
    wavlm_grad_max_norm: Optional[float] = 1.0
    lr_min_g: float = 1e-6
    lr_min_d: float = 1e-7


def load_kokoro_config(repo_id: str) -> Dict[str, Any]:
    path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def kokoro_vocab_and_context_length(repo_id: str) -> tuple[Dict[str, int], int]:
    """Vocabulary and max token positions for a Hugging Face Kokoro checkpoint (``config.json``)."""
    cfg = load_kokoro_config(repo_id)
    vocab = cfg["vocab"]
    n_ctx = int(cfg["plbert"]["max_position_embeddings"])
    return vocab, n_ctx
