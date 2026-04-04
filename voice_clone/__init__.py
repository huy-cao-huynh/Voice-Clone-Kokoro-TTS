"""Voice-clone training and inference (Kokoro + frozen WavLM-SV)."""

from .adapters import (
    AdapterRegistry,
    ResidualAdapter,
    build_decoder_adapters,
    build_duration_encoder_adapters,
)
from .config import (
    LossWeights,
    MelLossConfig,
    SLMDiscriminatorConfig,
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
from .losses import (
    MelReconstructionLoss,
    SLMFeatureDiscriminator,
    slm_discriminator_loss_hinge,
    slm_generator_loss_hinge,
    speaker_cosine_loss,
)

__all__ = [
    "AdapterRegistry",
    "ResidualAdapter",
    "build_decoder_adapters",
    "build_duration_encoder_adapters",
    "LossWeights",
    "MelLossConfig",
    "SLMDiscriminatorConfig",
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
    "SLMFeatureDiscriminator",
    "slm_discriminator_loss_hinge",
    "slm_generator_loss_hinge",
    "speaker_cosine_loss",
]
