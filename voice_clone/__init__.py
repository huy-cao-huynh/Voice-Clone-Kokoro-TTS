"""Voice-clone training and inference (Kokoro + frozen WeSpeaker frontend)."""

from .adapters import (
    AdapterRegistry,
    ResidualAdapter,
    build_decoder_adapters,
    build_duration_encoder_adapters,
)
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
    feature_matching_loss,
    speaker_cosine_loss,
    generator_loss_lsgan,
)
from .wespeaker_sv import WeSpeakerSV, WeSpeakerSVOutput, WeSpeakerToolkitEncoder

__all__ = [
    "AdapterRegistry",
    "ResidualAdapter",
    "build_decoder_adapters",
    "build_duration_encoder_adapters",
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
    "feature_matching_loss",
    "speaker_cosine_loss",
    "WeSpeakerSV",
    "WeSpeakerSVOutput",
    "WeSpeakerToolkitEncoder",
]
