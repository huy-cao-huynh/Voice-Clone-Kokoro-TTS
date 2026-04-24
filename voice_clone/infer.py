"""Inference: reference wav + text + language -> audio (Kokoro + SegmentGST)."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchaudio

from kokoro.model import KModel
from kokoro.pipeline import KPipeline

from .config import LossWeights, MelLossConfig, TrainConfig, kokoro_vocab_and_context_length
from .dataset import load_audio_mono, normalize_lang_code, phonemes_to_input_ids, text_to_phonemes
from .mhubert_encoder import MHuBERTEncoder
from .segment_gst import SegmentGST
from .train_adapters import build_models

KOKORO_OUTPUT_SR = 24_000

_LEGACY_CHECKPOINT_KEYS = {
    "kokoro_lora",
    "duration_adapters",
    "decoder_adapters",
    "generator_adapters",
}


def train_config_from_checkpoint_dict(d: Dict[str, Any]) -> TrainConfig:
    d = dict(d)
    if "loss_weights" in d and isinstance(d["loss_weights"], dict):
        d["loss_weights"] = LossWeights(**d["loss_weights"])
    if "mel" in d and isinstance(d["mel"], dict):
        d["mel"] = MelLossConfig(**d["mel"])
    valid = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in d.items() if k in valid})


def _assert_checkpoint_schema(ckpt: Dict[str, Any]) -> None:
    legacy = sorted(k for k in _LEGACY_CHECKPOINT_KEYS if k in ckpt)
    if legacy:
        raise ValueError(
            "Legacy adapter/LoRA checkpoint detected; this inference path only supports the new SegmentGST-only schema. "
            f"Found keys: {legacy}"
        )


def apply_voice_clone_checkpoint(
    ckpt: Dict[str, Any],
    *,
    gst: SegmentGST,
    kmodel: KModel,
) -> None:
    _assert_checkpoint_schema(ckpt)
    gst.load_state_dict(ckpt["segment_gst"])
    _ = kmodel


def build_stack_for_inference(
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[KModel, SegmentGST, MHuBERTEncoder]:
    kmodel, gst, mhubert, _sv_model, _disc, _mel, _kokoro_cfg = build_models(cfg, device)
    return kmodel, gst, mhubert


def infer_waveform(
    *,
    ckpt_path: Union[str, Path],
    ref_wav_path: Union[str, Path],
    text: str,
    lang_code: str,
    device: Optional[torch.device] = None,
    speed: Optional[float] = None,
    kokoro_repo_id: Optional[str] = None,
    strict_single_chunk: bool = False,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=True)
    raw_tc = ckpt.get("train_config")
    if raw_tc is not None:
        cfg = train_config_from_checkpoint_dict(raw_tc)
    else:
        if kokoro_repo_id is None:
            raise ValueError("Checkpoint has no train_config; pass kokoro_repo_id.")
        cfg = TrainConfig(kokoro_repo_id=kokoro_repo_id)
    if kokoro_repo_id is not None:
        cfg.kokoro_repo_id = kokoro_repo_id

    kmodel, gst, mhubert = build_stack_for_inference(cfg, device)
    apply_voice_clone_checkpoint(ckpt, gst=gst, kmodel=kmodel)

    lang = normalize_lang_code(lang_code)
    pipeline = KPipeline(lang_code=lang, repo_id=cfg.kokoro_repo_id, model=False)
    phonemes = text_to_phonemes(pipeline, text, strict_single_chunk=strict_single_chunk)
    vocab, context_length = kokoro_vocab_and_context_length(cfg.kokoro_repo_id)
    input_ids = phonemes_to_input_ids(vocab, phonemes, context_length=context_length).unsqueeze(0).to(device)
    ref_16 = load_audio_mono(ref_wav_path, target_sr=16_000).unsqueeze(0).to(device)

    gst.eval()
    mhubert.eval()
    kmodel.eval()
    with torch.inference_mode():
        mhubert_out = mhubert(ref_16)
        gst_out, _ = gst(mhubert_out.hidden_states, mhubert_out.frame_mask)
        audio, _ = kmodel.forward_with_tokens(input_ids, gst_out.ref_s, speed=speed or cfg.speed)
        audio = audio.clamp(-1.0, 1.0)
    return audio.squeeze(0).detach().float().cpu()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voice-clone inference: ref wav + text + lang -> WAV (24 kHz).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ref-wav", type=Path, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--lang", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--kokoro-repo", type=str, default=None)
    p.add_argument("--strict-single-chunk", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    wav = infer_waveform(
        ckpt_path=args.checkpoint,
        ref_wav_path=args.ref_wav,
        text=args.text,
        lang_code=args.lang,
        device=device,
        speed=args.speed,
        kokoro_repo_id=args.kokoro_repo,
        strict_single_chunk=args.strict_single_chunk,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), wav.unsqueeze(0), KOKORO_OUTPUT_SR)
    print(f"Wrote {args.out} ({wav.numel()} samples @ {KOKORO_OUTPUT_SR} Hz)")


if __name__ == "__main__":
    main()
