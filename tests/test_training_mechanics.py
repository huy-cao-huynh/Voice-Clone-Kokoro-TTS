"""Focused training-mechanics regressions for the SegmentGST-only stack."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def train_mod(monkeypatch):
    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(_ROOT / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)

    kokoro_pkg = types.ModuleType("kokoro")
    kokoro_pkg.__path__ = [str(_ROOT / "kokoro")]
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_pkg)

    disc_pkg = types.ModuleType("voice_clone.discriminators")
    disc_pkg.__path__ = [str(_ROOT / "voice_clone" / "discriminators")]
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators", disc_pkg)

    config_mod = types.ModuleType("voice_clone.config")

    @dataclass
    class LossWeights:
        lambda_mel: float = 20.0
        lambda_spk_contrastive: float = 1.0
        lambda_adv: float = 1.0
        lambda_fm: float = 2.0
        lambda_dur: float = 1.0
        lambda_f0: float = 1.0

    @dataclass
    class MelLossConfig:
        sample_rate: int = 24_000
        n_fft: int = 1024
        hop_length: int = 256
        win_length: int = 1024
        f_min: float = 0.0
        f_max: float | None = None

    @dataclass
    class TrainConfig:
        kokoro_repo_id: str = "dummy/kokoro"
        mhubert_repo_id: str = "dummy/mhubert"
        mhubert_extract_layer: int = 6
        wespeaker_checkpoint_path: str = "dummy/wespeaker/avg_model.pt"
        wespeaker_embedding_dim: int = 256
        wespeaker_sample_rate: int = 16_000
        universal_style_vector_path: str = "voice_clone/universal_style_vector.pt"
        feature_cache_root: str = "cache"
        mfa_min_coverage_ratio: float = 0.90
        mfa_max_coverage_ratio: float = 1.10
        disable_amp_for_stft: bool = True
        gst_embed_dim: int = 1024
        loss_weights: LossWeights = field(default_factory=LossWeights)
        mel: MelLossConfig = field(default_factory=MelLossConfig)
        contrastive_temperature: float = 0.07
        validate_cache_freshness: bool = True
        min_language_speakers: int = 2
        lr_g: float = 1e-4
        lr_d: float = 5e-5
        adam_b1: float = 0.8
        adam_b2: float = 0.99
        weight_decay_g: float = 0.0
        weight_decay_d: float = 0.0
        use_amp: bool = False
        log_interval: int = 1
        checkpoint_interval: int = 100
        save_final_checkpoint: bool = False
        warmup_steps: int = 0
        batch_size: int = 2
        grad_accum_steps: int = 1
        disc_start_step: int = 99999999
        speed: float = 1.0
        style_decoder_only_steps: int = 0
        gst_dropout: float = 0.0
        gst_conv_kernel_size: int = 5
        gst_conv_stride: int = 2
        gst_conv_padding: int = 2
        grad_clip_norm_g: float = 5.0
        grad_clip_norm_d: float = 1.0
        lr_min_g: float = 1e-4
        lr_min_d: float = 5e-5

    config_mod.LossWeights = LossWeights
    config_mod.MelLossConfig = MelLossConfig
    config_mod.TrainConfig = TrainConfig
    config_mod.kokoro_vocab_and_context_length = lambda repo_id: ({"a": 1, "b": 2}, 8)
    config_mod.load_kokoro_config = lambda repo_id: {"hidden_dim": 4, "n_layer": 2, "n_mels": 80, "vocab": {}}
    monkeypatch.setitem(sys.modules, "voice_clone.config", config_mod)

    dataset_mod = types.ModuleType("voice_clone.dataset")
    dataset_mod.VoiceCloneManifestDataset = object
    dataset_mod.collate_voice_clone_batch = lambda samples: samples
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", dataset_mod)

    losses_mod = types.ModuleType("voice_clone.losses")
    losses_mod.MelReconstructionLoss = nn.Identity
    losses_mod.discriminator_loss_lsgan = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.duration_loss_log_space = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.feature_matching_loss = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.generator_loss_lsgan = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.masked_l1_loss = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.speaker_contrastive_loss = lambda *args, **kwargs: torch.tensor(0.0)
    monkeypatch.setitem(sys.modules, "voice_clone.losses", losses_mod)

    mhubert_mod = types.ModuleType("voice_clone.mhubert_encoder")

    class DummyMHuBERT(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.hidden_size = 768

    mhubert_mod.MHuBERTEncoder = DummyMHuBERT
    monkeypatch.setitem(sys.modules, "voice_clone.mhubert_encoder", mhubert_mod)

    wespeaker_mod = types.ModuleType("voice_clone.wespeaker_sv")

    class DummyWeSpeaker(nn.Module):
        @classmethod
        def from_checkpoint(cls, *args, device=None, **kwargs):
            model = cls()
            return model.to(device=device) if device is not None else model

    wespeaker_mod.WeSpeakerSV = DummyWeSpeaker
    monkeypatch.setitem(sys.modules, "voice_clone.wespeaker_sv", wespeaker_mod)

    disc_mod = types.ModuleType("voice_clone.discriminators.hifigan")

    class DummyDisc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    disc_mod.HiFiGANMPDMSDDiscriminator = DummyDisc
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators.hifigan", disc_mod)

    kokoro_model_mod = types.ModuleType("kokoro.model")

    class DummyKModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.predictor = types.SimpleNamespace(text_encoder=types.SimpleNamespace(adapters=None))
            self.decoder = types.SimpleNamespace(decoder_adapters=None, generator=types.SimpleNamespace(adapters=None))

    kokoro_model_mod.KModel = DummyKModel
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_mod)

    segment_mod = _load_module(monkeypatch, "voice_clone.segment_gst", _ROOT / "voice_clone" / "segment_gst.py")
    train_mod = _load_module(monkeypatch, "voice_clone.train_adapters", _ROOT / "voice_clone" / "train_adapters.py")
    return train_mod, config_mod, segment_mod


def test_build_models_freezes_kokoro_and_loads_universal_style(train_mod, monkeypatch, tmp_path):
    train_adapters, config_mod, _segment_mod = train_mod
    monkeypatch.setattr(train_adapters, "build_mel_loss", lambda *args, **kwargs: nn.Identity())
    style_path = tmp_path / "u.pt"
    torch.save(torch.arange(256, dtype=torch.float32), style_path)
    cfg = config_mod.TrainConfig(universal_style_vector_path=str(style_path))
    kmodel, gst, _mhubert, _sv_model, _disc, _mel, _cfg = train_adapters.build_models(cfg, torch.device("cpu"))
    assert all(not p.requires_grad for p in kmodel.parameters())
    torch.testing.assert_close(gst.universal_style_vector, torch.arange(256, dtype=torch.float32))


def test_sampler_keeps_language_homogeneous_and_unique_speakers(train_mod):
    train_adapters, _config_mod, _segment_mod = train_mod

    class DummyDataset:
        rows = [
            {"lang_code": "a", "speaker_id": "s1"},
            {"lang_code": "a", "speaker_id": "s2"},
            {"lang_code": "a", "speaker_id": "s3"},
            {"lang_code": "b", "speaker_id": "t1"},
            {"lang_code": "b", "speaker_id": "t1"},
        ]

    sampler = train_adapters.LanguageHomogeneousUniqueSpeakerBatchSampler(
        DummyDataset(),
        batch_size=2,
        generator=torch.Generator().manual_seed(0),
        min_language_speakers=2,
    )
    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        langs = {DummyDataset.rows[i]["lang_code"] for i in batch}
        speakers = [DummyDataset.rows[i]["speaker_id"] for i in batch]
        assert len(langs) == 1
        assert len(speakers) == len(set(speakers))
    assert sampler.skipped_languages == {"b": 1}


def test_train_loop_validates_batch_and_logging_constraints(train_mod):
    train_adapters, config_mod, _segment_mod = train_mod
    with pytest.raises(ValueError, match="batch_size"):
        train_adapters.train_loop(object(), config_mod.TrainConfig(batch_size=1), torch.device("cpu"))
    with pytest.raises(ValueError, match="grad_accum_steps"):
        train_adapters.train_loop(object(), config_mod.TrainConfig(grad_accum_steps=0), torch.device("cpu"))
    with pytest.raises(ValueError, match="log_interval"):
        train_adapters.train_loop(object(), config_mod.TrainConfig(log_interval=0), torch.device("cpu"))


def test_forward_batch_outputs_forces_target_total_frames(train_mod):
    train_adapters, _config_mod, _segment_mod = train_mod

    class DummyGST(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, ref_hidden_states, ref_frame_mask, use_universal_style_pred=False):
            del ref_hidden_states, ref_frame_mask
            self.calls.append(use_universal_style_pred)
            return types.SimpleNamespace(
                ref_s=torch.zeros(2, 256),
                style_dec=torch.zeros(2, 128),
                style_pred=torch.zeros(2, 128),
                pooled_style=torch.zeros(2, 1024),
            ), None

    class DummyKModel:
        def __init__(self):
            self.calls = []

        def forward_with_tokens(self, input_ids, ref_s, speed, gt_dur_frames=None, force_total_frames=None, return_training_outputs=False):
            del ref_s
            self.calls.append(
                {
                    "tokens": tuple(input_ids.squeeze(0).tolist()),
                    "speed": speed,
                    "gt_dur_frames": None if gt_dur_frames is None else tuple(gt_dur_frames.squeeze(0).tolist()),
                    "force_total_frames": force_total_frames,
                    "return_training_outputs": return_training_outputs,
                }
            )
            frames = max(int(force_total_frames or (gt_dur_frames.sum().item() if gt_dur_frames is not None else 1)), 1)
            return types.SimpleNamespace(
                audio=torch.zeros(1, frames * 600),
                duration_logits=torch.ones(1, input_ids.size(1)),
                rounded_durations=torch.ones(1, input_ids.size(1), dtype=torch.long),
                f0_pred=torch.zeros(1, frames),
                n_pred=torch.zeros(1, frames),
            )

    batch = {
        "ref_hidden_states": torch.zeros(2, 3, 4),
        "ref_frame_mask": torch.ones(2, 3, dtype=torch.bool),
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "input_ids_lengths": torch.tensor([3, 2]),
        "target_lengths": torch.tensor([24000, 12600]),
        "gt_dur_frames": torch.tensor([[0, 5, 6], [0, 0, 0]], dtype=torch.long),
        "prosody_enabled": torch.tensor([True, False], dtype=torch.bool),
    }

    kmodel = DummyKModel()
    gst = DummyGST()
    outputs, gst_out = train_adapters._forward_batch_outputs(
        kmodel,
        gst,
        batch,
        torch.device("cpu"),
        speed=1.0,
        force_target_total_frames=True,
        style_decoder_only=True,
    )

    assert len(outputs) == 2
    assert gst_out.ref_s.shape == (2, 256)
    assert kmodel.calls[0]["gt_dur_frames"] == (0, 5, 6)
    assert kmodel.calls[0]["force_total_frames"] is None
    assert kmodel.calls[1]["force_total_frames"] == 21
    assert all(call["return_training_outputs"] is True for call in kmodel.calls)
    assert gst.calls == [True]


def test_effective_target_audio_lengths_and_speaker_targets_follow_tf_span(train_mod):
    train_adapters, _config_mod, _segment_mod = train_mod
    batch = {
        "target_lengths": torch.tensor([24000, 18000], dtype=torch.long),
        "prosody_enabled": torch.tensor([True, False], dtype=torch.bool),
        "gt_total_duration_samples": torch.tensor([12000, 9000], dtype=torch.long),
        "target_wespeaker_embedding": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        "target_wespeaker_embedding_tf_span": torch.tensor([[3.0, 3.0], [4.0, 4.0]]),
    }
    tf_lengths = train_adapters._effective_target_audio_lengths(
        batch,
        torch.device("cpu"),
        match_teacher_forced_span=True,
    )
    free_lengths = train_adapters._effective_target_audio_lengths(
        batch,
        torch.device("cpu"),
        match_teacher_forced_span=False,
    )
    spk_targets = train_adapters._speaker_target_embeddings(
        batch,
        torch.device("cpu"),
        match_teacher_forced_span=True,
    )
    assert tf_lengths.tolist() == [12000, 18000]
    assert free_lengths.tolist() == [24000, 18000]
    assert spk_targets.tolist() == [[3.0, 3.0], [2.0, 2.0]]


def test_forward_batch_outputs_leaves_validation_free_running(train_mod):
    train_adapters, _config_mod, _segment_mod = train_mod

    class DummyGST(nn.Module):
        def forward(self, ref_hidden_states, ref_frame_mask, use_universal_style_pred=False):
            del ref_hidden_states, ref_frame_mask
            assert use_universal_style_pred is False
            return types.SimpleNamespace(
                ref_s=torch.zeros(1, 256),
                style_dec=torch.zeros(1, 128),
                style_pred=torch.zeros(1, 128),
                pooled_style=torch.zeros(1, 1024),
            ), None

    class DummyKModel:
        def __init__(self):
            self.force_total_frames = []

        def forward_with_tokens(self, input_ids, ref_s, speed, gt_dur_frames=None, force_total_frames=None, return_training_outputs=False):
            del input_ids, ref_s, speed, return_training_outputs
            self.force_total_frames.append(force_total_frames)
            assert gt_dur_frames is None
            return types.SimpleNamespace(
                audio=torch.zeros(1, 600),
                duration_logits=torch.ones(1, 3),
                rounded_durations=torch.ones(1, 3, dtype=torch.long),
                f0_pred=torch.zeros(1, 1),
                n_pred=torch.zeros(1, 1),
            )

    batch = {
        "ref_hidden_states": torch.zeros(1, 3, 4),
        "ref_frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "input_ids": torch.tensor([[1, 2, 3]]),
        "input_ids_lengths": torch.tensor([3]),
        "target_lengths": torch.tensor([24000]),
    }

    kmodel = DummyKModel()
    train_adapters._forward_batch_outputs(
        kmodel,
        DummyGST(),
        batch,
        torch.device("cpu"),
        speed=1.0,
        force_target_total_frames=False,
    )

    assert kmodel.force_total_frames == [None]


def test_collapse_diagnostics_detects_collapsed_batch(train_mod):
    train_adapters, _config_mod, _segment_mod = train_mod
    ref_s = torch.ones(3, 256)
    metrics = train_adapters._collapse_diagnostics(ref_s, torch.zeros(256))
    assert metrics["collapse/ref_s_std_mean"] == pytest.approx(0.0)
    assert metrics["collapse/ref_s_pairwise_cos_mean"] == pytest.approx(1.0)


def test_gst_projection_diagnostics_report_projection_and_prior_norms(train_mod):
    train_adapters, _config_mod, _segment_mod = train_mod
    gst_out = types.SimpleNamespace(
        style_dec=torch.ones(2, 128) * 3.0,
        style_pred=torch.ones(2, 128) * 5.0,
    )
    metrics = train_adapters._gst_projection_diagnostics(gst_out, torch.ones(256))
    assert metrics["gst/proj_dec_norm_mean"] > 0.0
    assert metrics["gst/proj_pred_norm_mean"] > metrics["gst/proj_dec_norm_mean"]
    assert set(metrics) == {"gst/proj_dec_norm_mean", "gst/proj_pred_norm_mean"}
