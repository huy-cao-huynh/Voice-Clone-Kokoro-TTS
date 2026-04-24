"""Voice-clone training and inference (Kokoro + frozen frontends)."""

from .config import (
    LossWeights,
    MelLossConfig,
    TrainConfig,
    kokoro_vocab_and_context_length,
    load_kokoro_config,
)
from .dataset import (
    VoiceCloneManifestDataset,
    collate_voice_clone_batch,
    load_audio_mono,
    normalize_lang_code,
    phonemes_to_input_ids,
    text_to_phonemes,
)
from .discriminators import HiFiGANMPDMSDDiscriminator
from .losses import (
    MelReconstructionLoss,
    discriminator_loss_lsgan,
    duration_loss_log_space,
    feature_matching_loss,
    generator_loss_lsgan,
    masked_l1_loss,
    speaker_contrastive_loss,
)
from .wespeaker_sv import WeSpeakerSV, WeSpeakerSVOutput, WeSpeakerToolkitEncoder

__all__ = [
    "LossWeights",
    "MelLossConfig",
    "TrainConfig",
    "kokoro_vocab_and_context_length",
    "load_kokoro_config",
    "VoiceCloneManifestDataset",
    "collate_voice_clone_batch",
    "load_audio_mono",
    "normalize_lang_code",
    "phonemes_to_input_ids",
    "text_to_phonemes",
    "MelReconstructionLoss",
    "HiFiGANMPDMSDDiscriminator",
    "discriminator_loss_lsgan",
    "generator_loss_lsgan",
    "duration_loss_log_space",
    "feature_matching_loss",
    "masked_l1_loss",
    "speaker_contrastive_loss",
    "WeSpeakerSV",
    "WeSpeakerSVOutput",
    "WeSpeakerToolkitEncoder",
]
