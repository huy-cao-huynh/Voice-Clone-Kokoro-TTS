"""Inference: reference wav + text + language -> audio (Kokoro + trained SegmentGST / L-adapters, frozen WeSpeaker-SV)."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchaudio

from kokoro.model import KModel
from kokoro.pipeline import KPipeline

from .config import LossWeights, MelLossConfig, TrainConfig, kokoro_vocab_and_context_length
from .dataset import load_audio_mono, normalize_lang_code, phonemes_to_input_ids, text_to_phonemes
from .segment_gst import SegmentGST
from .train_adapters import build_models
from .wespeaker_sv import WeSpeakerSV

KOKORO_OUTPUT_SR = 24_000


def train_config_from_checkpoint_dict(d: Dict[str, Any]) -> TrainConfig:
    """Rebuild :class:`TrainConfig` from ``torch.save(..., {"train_config": asdict(cfg)})`` payload."""
    d = dict(d)
    if "loss_weights" in d and isinstance(d["loss_weights"], dict):
        d["loss_weights"] = LossWeights(**d["loss_weights"])
    if "mel" in d and isinstance(d["mel"], dict):
        d["mel"] = MelLossConfig(**d["mel"])
    valid = {f.name for f in fields(TrainConfig)}
    filtered = {k: v for k, v in d.items() if k in valid}
    return TrainConfig(**filtered)


def apply_voice_clone_checkpoint(
    ckpt: Dict[str, Any],
    *,
    gst: SegmentGST,
    kmodel: KModel,
) -> None:
    """Load SegmentGST and L-adapter weights from an in-memory checkpoint dict."""
    gst.load_state_dict(ckpt["segment_gst"])
    if ckpt.get("duration_adapters") and kmodel.predictor.text_encoder.adapters is not None:
        kmodel.predictor.text_encoder.adapters.load_state_dict(ckpt["duration_adapters"])
    if ckpt.get("decoder_adapters") and kmodel.decoder.decoder_adapters is not None:
        kmodel.decoder.decoder_adapters.load_state_dict(ckpt["decoder_adapters"])
    if ckpt.get("generator_adapters") and kmodel.decoder.generator.adapters is not None:
        kmodel.decoder.generator.adapters.load_state_dict(ckpt["generator_adapters"])


def build_stack_for_inference(
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[KModel, SegmentGST, WeSpeakerSV]:
    """Construct Kokoro (with adapter slots), SegmentGST, and frozen WeSpeaker-SV."""
    kmodel, gst, sv_model, _disc, _mel, _kokoro_cfg = build_models(cfg, device)
    return kmodel, gst, sv_model


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
    """Load checkpoint, run G2P + style from reference audio, return 1-D float waveform at 24 kHz."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_tc = ckpt.get("train_config")
    if raw_tc is not None:
        cfg = train_config_from_checkpoint_dict(raw_tc)
    else:
        if kokoro_repo_id is None:
            raise ValueError(
                "Checkpoint has no train_config; pass kokoro_repo_id (e.g. hexgrad/Kokoro-82M) "
                "so Kokoro layout matches training."
            )
        cfg = TrainConfig(kokoro_repo_id=kokoro_repo_id)
    if kokoro_repo_id is not None:
        cfg = replace(cfg, kokoro_repo_id=kokoro_repo_id)

    kmodel, gst, sv_model = build_stack_for_inference(cfg, device)
    apply_voice_clone_checkpoint(ckpt, gst=gst, kmodel=kmodel)

    lang = normalize_lang_code(lang_code)
    pipeline = KPipeline(lang_code=lang, repo_id=cfg.kokoro_repo_id, model=False)
    phonemes = text_to_phonemes(pipeline, text, strict_single_chunk=strict_single_chunk)
    vocab, context_length = kokoro_vocab_and_context_length(cfg.kokoro_repo_id)
    input_ids = phonemes_to_input_ids(vocab, phonemes, context_length=context_length).unsqueeze(0).to(device)

    ref_16 = load_audio_mono(ref_wav_path, target_sr=16_000).unsqueeze(0).to(device)
    sp = speed if speed is not None else cfg.speed

    gst.eval()
    sv_model.eval()
    kmodel.eval()
    with torch.inference_mode():
        wv = sv_model(ref_16, sampling_rate=cfg.wespeaker_sample_rate, grad_through_input=False)
        if wv.frame_features is None:
            raise RuntimeError("WeSpeakerSV must return frame_features for SegmentGST conditioning.")
        frame_mask = torch.ones(
            wv.frame_features.size(0),
            wv.frame_features.size(1),
            device=wv.frame_features.device,
            dtype=wv.frame_features.dtype,
        )
        gst_out, _ = gst(wv.frame_features, frame_mask)
        ref_s = gst_out.ref_s
        audio, _ = kmodel.forward_with_tokens(input_ids, ref_s, speed=sp)
        audio = audio.clamp(-1.0, 1.0)
    return audio.squeeze(0).detach().float().cpu()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voice-clone inference: ref wav + text + lang → WAV (24 kHz).")
    p.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint (.pt) with GST + adapters.")
    p.add_argument("--ref-wav", type=Path, required=True, help="Reference speaker audio (any rate; resampled to 16 kHz for WeSpeaker).")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--lang", type=str, required=True, help="Kokoro lang code or alias (e.g. a, en-us, z).")
    p.add_argument("--out", type=Path, required=True, help="Output WAV path.")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu | mps (default: auto).")
    p.add_argument("--speed", type=float, default=None, help="Kokoro speed factor (default: from checkpoint TrainConfig).")
    p.add_argument(
        "--kokoro-repo",
        type=str,
        default=None,
        help="Kokoro HF repo id: required if checkpoint omits train_config; optional override otherwise (must match training).",
    )
    p.add_argument(
        "--strict-single-chunk",
        action="store_true",
        help="Fail if G2P splits text into multiple chunks (default: concatenate with warning).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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
