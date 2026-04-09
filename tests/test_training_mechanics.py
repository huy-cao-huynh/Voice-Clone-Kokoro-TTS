"""Focused training-mechanics regressions for ``voice_clone.train_adapters``."""

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
        lambda_mel: float = 1.0
        lambda_spk: float = 15.0
        lambda_adv: float = 2.0
        lambda_fm: float = 15.0

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
        wespeaker_checkpoint_path: str = "dummy/wespeaker/avg_model.pt"
        wespeaker_embedding_dim: int = 256
        wespeaker_sample_rate: int = 16_000
        universal_style_vector_path: str = "voice_clone/universal_style_vector.pt"
        disable_amp_for_stft: bool = True
        adapter_bottleneck: int = 64
        loss_weights: LossWeights = field(default_factory=LossWeights)
        mel: MelLossConfig = field(default_factory=MelLossConfig)
        lr_g: float = 1e-4
        lr_d: float = 5e-5

    config_mod.LossWeights = LossWeights
    config_mod.MelLossConfig = MelLossConfig
    config_mod.TrainConfig = TrainConfig
    config_mod.kokoro_vocab_and_context_length = lambda repo_id: ({}, 0)
    config_mod.load_kokoro_config = lambda repo_id: {
        "hidden_dim": 4,
        "n_layer": 2,
        "n_mels": 80,
        "vocab": {},
        "istftnet": {"upsample_initial_channel": 16, "upsample_rates": [2, 2]},
    }
    monkeypatch.setitem(sys.modules, "voice_clone.config", config_mod)

    adapters_mod = types.ModuleType("voice_clone.adapters")

    class DummyRegistry:
        @staticmethod
        def from_dims(**kwargs):
            return types.SimpleNamespace(
                duration_encoder=nn.ModuleList([nn.Linear(1, 1)]),
                decoder=nn.ModuleList([nn.Linear(1, 1)]),
                generator=nn.ModuleList([nn.Linear(1, 1)]),
            )

    adapters_mod.AdapterRegistry = DummyRegistry
    monkeypatch.setitem(sys.modules, "voice_clone.adapters", adapters_mod)

    dataset_mod = types.ModuleType("voice_clone.dataset")
    dataset_mod.VoiceCloneManifestDataset = object
    dataset_mod.collate_voice_clone_batch = lambda samples: samples
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", dataset_mod)

    losses_mod = types.ModuleType("voice_clone.losses")

    class DummyMelLoss(nn.Module):
        def forward(self, pred_wav, target_wav):
            loss = (pred_wav - target_wav).pow(2).mean()
            return types.SimpleNamespace(
                loss=loss,
                mel_pred=pred_wav.unsqueeze(1),
                mel_target=target_wav.unsqueeze(1),
            )

    losses_mod.MelReconstructionLoss = DummyMelLoss
    losses_mod.discriminator_loss_lsgan = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.feature_matching_loss = lambda real_feats, fake_feats: sum(
        f.mean() for per_disc in fake_feats for f in per_disc
    )
    losses_mod.generator_loss_lsgan = lambda fake_logits: fake_logits.mean()
    losses_mod.speaker_cosine_loss = lambda ref, gen: (ref - gen).pow(2).mean()
    losses_mod.speaker_input_mel_from_waveform = lambda *args, **kwargs: torch.zeros(1)
    monkeypatch.setitem(sys.modules, "voice_clone.losses", losses_mod)

    profiling_mod = types.ModuleType("voice_clone.train_profiling")
    profiling_mod.BreakdownAggregator = object
    profiling_mod.StepStopwatch = object
    profiling_mod.build_torch_profiler = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "voice_clone.train_profiling", profiling_mod)

    wespeaker_mod = types.ModuleType("voice_clone.wespeaker_sv")

    @dataclass
    class DummySVOutput:
        pooled_embedding: torch.Tensor
        frame_features: torch.Tensor | None
        frame_mask: torch.Tensor | None = None

    class DummyWeSpeaker(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(1, 1, bias=False)
            self.encoder.requires_grad_(False)

        @property
        def dtype(self) -> torch.dtype:
            return next(self.encoder.parameters()).dtype

        @classmethod
        def from_checkpoint(cls, *args, device=None, **kwargs):
            model = cls()
            if device is not None:
                model = model.to(device=device)
            return model

        def forward_from_mel(self, mel, **kwargs):
            batch = mel.size(0)
            return DummySVOutput(
                pooled_embedding=torch.zeros(batch, 256, device=mel.device, dtype=mel.dtype),
                frame_features=torch.zeros(batch, 8, 1024, device=mel.device, dtype=mel.dtype),
            )

    wespeaker_mod.WeSpeakerSV = DummyWeSpeaker
    wespeaker_mod.WeSpeakerSVOutput = DummySVOutput
    monkeypatch.setitem(sys.modules, "voice_clone.wespeaker_sv", wespeaker_mod)

    disc_mod = types.ModuleType("voice_clone.discriminators.hifigan")

    class DummyDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    disc_mod.HiFiGANMPDMSDDiscriminator = DummyDiscriminator
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators.hifigan", disc_mod)

    kokoro_model_mod = types.ModuleType("kokoro.model")

    class DummyTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapters = nn.ModuleList([nn.Linear(1, 1)])

    class DummyPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = DummyTextEncoder()

    class DummyGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapters = nn.ModuleList([nn.Linear(1, 1)])

    class DummyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_adapters = nn.ModuleList([nn.Linear(1, 1)])
            self.generator = DummyGenerator()

    class DummyKModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.backbone = nn.Linear(1, 1)
            self.predictor = DummyPredictor()
            self.decoder = DummyDecoder()

    kokoro_model_mod.KModel = DummyKModel
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_mod)

    segment_mod = _load_module(monkeypatch, "voice_clone.segment_gst", _ROOT / "voice_clone" / "segment_gst.py")
    train_mod = _load_module(monkeypatch, "voice_clone.train_adapters", _ROOT / "voice_clone" / "train_adapters.py")
    return train_mod, config_mod, segment_mod


class _ToyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Linear(2, 2)])
        self.trunk = nn.Linear(2, 2)


class _ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = _ToyTextEncoder()
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.LSTM(4, 4, num_layers=2, dropout=0.25, batch_first=True)


class _ToyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Linear(2, 2)])
        self.base = nn.Linear(2, 2)


class _ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_adapters = nn.ModuleList([nn.Linear(2, 2)])
        self.generator = _ToyGenerator()
        self.dropout = nn.Dropout(0.4)


class _ToyKModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.predictor = _ToyPredictor()
        self.decoder = _ToyDecoder()


class _ToyGST(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3)


class _ToyDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.forward_calls = 0

    def forward(self, wav: torch.Tensor):
        self.forward_calls += 1
        logits = wav.mean(dim=-1, keepdim=True)
        feats = [[wav.unsqueeze(1)]]
        return logits, feats


def _adapter_param_ids(kmodel: _ToyKModel) -> set[int]:
    adapter_ids = set()
    adapter_ids.update(id(p) for p in kmodel.predictor.text_encoder.adapters.parameters())
    adapter_ids.update(id(p) for p in kmodel.decoder.decoder_adapters.parameters())
    adapter_ids.update(id(p) for p in kmodel.decoder.generator.adapters.parameters())
    return adapter_ids


class TestFreezeKokoroExceptAdapters:
    def test_freezes_only_non_adapter_params(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        kmodel = _ToyKModel()

        train_adapters.freeze_kokoro_except_adapters(kmodel)

        adapter_ids = _adapter_param_ids(kmodel)
        for name, param in kmodel.named_parameters():
            if id(param) in adapter_ids:
                assert param.requires_grad, f"adapter param {name} should stay trainable"
            else:
                assert not param.requires_grad, f"non-adapter param {name} should be frozen"

    def test_keeps_model_in_train_mode_but_disables_dropout(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        kmodel = _ToyKModel()

        train_adapters.freeze_kokoro_except_adapters(kmodel)

        assert kmodel.training is True
        for module in kmodel.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.0
            if isinstance(module, nn.RNNBase):
                assert module.dropout == 0.0


class TestUniversalStyleVectorLoading:
    def test_build_models_loads_universal_style_vector_into_initial_ref_s(self, train_mod, tmp_path, monkeypatch):
        train_adapters, config_mod, _segment_mod = train_mod
        monkeypatch.setattr(train_adapters, "build_mel_loss", lambda *args, **kwargs: nn.Identity())

        device = torch.device("cpu")
        base_voice = torch.linspace(-1.0, 1.0, 256)
        vec_path = tmp_path / "universal_style_vector.pt"
        torch.save(base_voice, vec_path)

        cfg = config_mod.TrainConfig(universal_style_vector_path=str(vec_path))
        _kmodel, gst, _sv_model, _disc, _mel_loss, _kokoro_cfg = train_adapters.build_models(cfg, device)

        torch.testing.assert_close(gst.universal_style_vector.cpu(), base_voice)
        frames = torch.randn(2, 12, gst.frame_dim)
        mask = torch.ones(2, 12)
        with torch.no_grad():
            out, _ = gst(frames, mask)
        expected = base_voice.unsqueeze(0).expand(2, -1)
        torch.testing.assert_close(out.ref_s.cpu(), expected)


class TestGeneratorTrainableParameters:
    def test_returns_gst_and_trainable_kokoro_params_only(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        kmodel = _ToyKModel()
        gst = _ToyGST()

        train_adapters.freeze_kokoro_except_adapters(kmodel)
        params = train_adapters.generator_trainable_parameters(kmodel, gst)

        returned_ids = {id(p) for p in params}
        expected_ids = {id(p) for p in gst.parameters()} | {
            id(p) for p in kmodel.parameters() if p.requires_grad
        }
        assert returned_ids == expected_ids
        assert all(p.requires_grad for p in params)


class TestSpeakerFrameMask:
    def test_uses_explicit_mask_when_present(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        frame_features = torch.randn(2, 5, 7)
        explicit = torch.tensor(
            [[True, True, False, False, False], [True, True, True, False, False]]
        )
        ref_out = types.SimpleNamespace(frame_features=frame_features, frame_mask=explicit)

        mask = train_adapters._speaker_frame_mask(ref_out)

        assert torch.equal(mask, explicit)

    def test_defaults_to_all_true_when_wespeaker_omits_mask(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        frame_features = torch.randn(2, 5, 7)
        ref_out = types.SimpleNamespace(frame_features=frame_features, frame_mask=None)

        mask = train_adapters._speaker_frame_mask(ref_out)

        assert mask.dtype == torch.bool
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_rejects_missing_frame_features(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        ref_out = types.SimpleNamespace(frame_features=None, frame_mask=None)

        with pytest.raises(RuntimeError, match="frame_features"):
            train_adapters._speaker_frame_mask(ref_out)


class TestPhasedTraining:
    def test_no_gan_terms_during_warmup(self, train_mod):
        train_adapters, config_mod, _segment_mod = train_mod
        opt_g = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=1e-4)
        disc = _ToyDisc()
        mel_loss = train_adapters.MelReconstructionLoss()
        pred_wav = torch.randn(1, 8, requires_grad=True)
        pred_wav_seg = pred_wav[:, :4]
        ref_pooled = torch.randn(1, 4)
        gen_pooled = torch.randn(1, 4, requires_grad=True)
        tgt_24 = torch.randn(1, 8)

        metrics = train_adapters.generator_loss_backward(
            _ToyKModel(),
            _ToyGST(),
            mel_loss,
            opt_g,
            pred_wav,
            pred_wav_seg,
            ref_pooled,
            gen_pooled,
            tgt_24,
            disc_waveform=disc,
            real_disc_features=None,
            weights=config_mod.LossWeights(),
            scaler=None,
            use_amp=False,
            scale_factor=1.0,
        )

        assert metrics["loss_adv_g"] == 0.0
        assert metrics["loss_fm"] == 0.0
        assert disc.forward_calls == 0
        assert pred_wav.grad is not None
        assert gen_pooled.grad is not None

    def test_disc_requires_grad_unchanged_and_real_features_detached(self, train_mod, monkeypatch):
        train_adapters, config_mod, _segment_mod = train_mod
        disc = _ToyDisc()
        opt_g = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=1e-4)
        mel_loss = train_adapters.MelReconstructionLoss()
        pred_wav = torch.randn(1, 8, requires_grad=True)
        pred_wav_seg = pred_wav[:, :4]
        ref_pooled = torch.randn(1, 4)
        gen_pooled = torch.randn(1, 4, requires_grad=True)
        tgt_24 = torch.randn(1, 8)
        real_disc_features = [[torch.randn(1, 1, 4, requires_grad=True)]]
        seen_requires_grad = []

        def fake_feature_matching(real_feats, fake_feats):
            seen_requires_grad.extend(feat.requires_grad for per_disc in real_feats for feat in per_disc)
            return sum(feat.mean() for per_disc in fake_feats for feat in per_disc)

        monkeypatch.setattr(train_adapters, "feature_matching_loss", fake_feature_matching)
        before = [param.requires_grad for param in disc.parameters()]

        train_adapters.generator_loss_backward(
            _ToyKModel(),
            _ToyGST(),
            mel_loss,
            opt_g,
            pred_wav,
            pred_wav_seg,
            ref_pooled,
            gen_pooled,
            tgt_24,
            disc_waveform=disc,
            real_disc_features=real_disc_features,
            weights=config_mod.LossWeights(),
            scaler=None,
            use_amp=False,
            scale_factor=1.0,
        )

        after = [param.requires_grad for param in disc.parameters()]
        assert before == after
        assert all(after)
        assert seen_requires_grad == [False]


class TestWarmupLR:
    def test_lr_reaches_full_after_warmup(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        lr = 1e-4
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = train_adapters._build_scheduler(opt, warmup_steps=50, total_steps=500, lr_min=1e-6)

        for _ in range(50):
            opt.step()
            sched.step()

        assert opt.param_groups[0]["lr"] == pytest.approx(lr, rel=1e-4)

    def test_lr_starts_low(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        lr = 1e-4
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        train_adapters._build_scheduler(opt, warmup_steps=50, total_steps=500, lr_min=1e-6)
        assert opt.param_groups[0]["lr"] == pytest.approx(lr * 1e-3, rel=1e-5)

    def test_lr_decays_after_warmup(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        lr = 1e-4
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = train_adapters._build_scheduler(opt, warmup_steps=50, total_steps=500, lr_min=1e-6)

        for _ in range(150):
            opt.step()
            sched.step()

        actual_lr = opt.param_groups[0]["lr"]
        assert actual_lr < lr
        assert actual_lr > 1e-6

    def test_lr_reaches_minimum_at_end(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        lr = 1e-4
        lr_min = 1e-6
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = train_adapters._build_scheduler(opt, warmup_steps=50, total_steps=500, lr_min=lr_min)

        for _ in range(500):
            opt.step()
            sched.step()

        assert opt.param_groups[0]["lr"] == pytest.approx(lr_min, rel=1e-3)

    def test_warmup_only_when_no_total_steps(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        lr = 1e-4
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = train_adapters._build_scheduler(opt, warmup_steps=50, total_steps=None, lr_min=1e-6)

        for _ in range(150):
            opt.step()
            sched.step()

        assert opt.param_groups[0]["lr"] == pytest.approx(lr, rel=1e-5)
