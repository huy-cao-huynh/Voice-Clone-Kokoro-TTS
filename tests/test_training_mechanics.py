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
        mhubert_repo_id: str = "dummy/mhubert"
        mhubert_extract_layer: int = 9
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
        gst_dropout: float = 0.1
        speed: float = 1.0
        use_amp: bool = False

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
        def forward(self, pred_wav, target_wav, *, pred_lengths=None, target_lengths=None):
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
            self.last_waveform_lengths = None

        @property
        def dtype(self) -> torch.dtype:
            return next(self.encoder.parameters()).dtype

        @classmethod
        def from_checkpoint(cls, *args, device=None, **kwargs):
            model = cls()
            if device is not None:
                model = model.to(device=device)
            return model

        def forward(self, waveforms, **kwargs):
            self.last_waveform_lengths = kwargs.get("waveform_lengths")
            batch = waveforms.size(0)
            return DummySVOutput(
                pooled_embedding=torch.zeros(batch, 256, device=waveforms.device, dtype=waveforms.dtype),
                frame_features=None,
            )

        def _waveforms_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
            return torch.zeros(wav.size(0), 80, 32, device=wav.device, dtype=wav.dtype)

        def forward_from_mel(self, mel, **kwargs):
            batch = mel.size(0)
            return DummySVOutput(
                pooled_embedding=torch.zeros(batch, 256, device=mel.device, dtype=mel.dtype),
                frame_features=torch.zeros(batch, 8, 1024, device=mel.device, dtype=mel.dtype),
            )

    wespeaker_mod.WeSpeakerSV = DummyWeSpeaker
    wespeaker_mod.WeSpeakerSVOutput = DummySVOutput
    monkeypatch.setitem(sys.modules, "voice_clone.wespeaker_sv", wespeaker_mod)

    mhubert_mod = types.ModuleType("voice_clone.mhubert_encoder")

    @dataclass
    class DummyMHuBERTOutput:
        hidden_states: torch.Tensor
        frame_mask: torch.Tensor

    class DummyMHuBERTEncoder(nn.Module):
        def __init__(self, repo_id="dummy", extract_layer=9):
            super().__init__()
            self.hidden_size = 768
            self._dummy = nn.Linear(1, 1, bias=False)
            self.requires_grad_(False)
            self.last_attention_mask = None

        def forward(self, waveforms_16k, attention_mask=None):
            self.last_attention_mask = attention_mask
            B = waveforms_16k.size(0)
            return DummyMHuBERTOutput(
                hidden_states=torch.zeros(B, 8, self.hidden_size, device=waveforms_16k.device),
                frame_mask=torch.ones(B, 8, device=waveforms_16k.device, dtype=torch.bool),
            )

    mhubert_mod.MHuBERTEncoder = DummyMHuBERTEncoder
    mhubert_mod.MHuBERTOutput = DummyMHuBERTOutput
    monkeypatch.setitem(sys.modules, "voice_clone.mhubert_encoder", mhubert_mod)

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
            self.last_forward_token_len: int | None = None

        def forward_with_tokens(self, input_ids, ref_s, speed=1.0):
            self.last_forward_token_len = int(input_ids.shape[1])
            return torch.zeros(input_ids.shape[0], 32, device=input_ids.device, dtype=input_ids.dtype), None

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


def _trainable_adapter_param_ids(kmodel: _ToyKModel) -> set[int]:
    """Params that freeze_kokoro_except_adapters should leave trainable.

    All adapters are trainable: duration encoder, decoder, and generator.
    """
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

        trainable_ids = _trainable_adapter_param_ids(kmodel)
        for name, param in kmodel.named_parameters():
            if id(param) in trainable_ids:
                assert param.requires_grad, f"adapter param {name} should stay trainable"
            else:
                assert not param.requires_grad, f"non-adapter param {name} should be frozen"

    def test_duration_encoder_adapters_trainable(self, train_mod):
        """Duration encoder adapters are trainable along with decoder/generator adapters."""
        train_adapters, _config_mod, _segment_mod = train_mod
        kmodel = _ToyKModel()

        train_adapters.freeze_kokoro_except_adapters(kmodel)

        for p in kmodel.predictor.text_encoder.adapters.parameters():
            assert p.requires_grad, "duration encoder adapter params should be trainable"

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
        _kmodel, gst, _mhubert, _sv_model, _disc, _mel_loss, _kokoro_cfg = train_adapters.build_models(cfg, device)

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


def test_effective_optimizer_steps_per_epoch_ceil(train_mod):
    train_adapters, _config_mod, _segment_mod = train_mod
    f = train_adapters._effective_optimizer_steps_per_epoch
    assert f(1, 8) == 1
    assert f(8, 8) == 1
    assert f(9, 8) == 2
    assert f(0, 8) == 0


class TestWarmupLR:
    def test_no_warmup_is_pure_cosine_when_total_known(self, train_mod):
        train_adapters, _config_mod, _segment_mod = train_mod
        lr = 1e-4
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = train_adapters._build_scheduler(opt, warmup_steps=0, total_steps=100, lr_min=1e-6)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert opt.param_groups[0]["lr"] == pytest.approx(lr, rel=1e-5)
        for _ in range(100):
            opt.step()
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-6, rel=1e-2)

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


class TestWandbValidationForward:
    """Regression: validation inference must mirror training (trim tokens, mask reference)."""

    def test_log_wandb_validation_trims_tokens_and_masks_reference(self, train_mod, monkeypatch, tmp_path):
        train_adapters, config_mod, _segment_mod = train_mod
        true_tokens = 5
        ref_len = 100

        def evil_collate(samples):
            _ = samples[0]
            pad = torch.zeros(1, true_tokens + 4, dtype=torch.long)
            pad[0, :true_tokens] = torch.arange(1, true_tokens + 1, dtype=torch.long)
            ref = torch.zeros(1, 200)
            ref[0, :ref_len] = torch.randn(ref_len)
            return {
                "ref_wav_16k": ref,
                "ref_lengths": torch.tensor([ref_len]),
                "target_wav_24k": torch.zeros(1, 32),
                "target_lengths": torch.tensor([32]),
                "input_ids": pad,
                "input_ids_lengths": torch.tensor([true_tokens]),
                "text": "caption",
            }

        monkeypatch.setattr(train_adapters, "collate_voice_clone_batch", evil_collate)
        wb_mod = types.SimpleNamespace(
            Audio=lambda *a, **k: None,
            Table=lambda *a, **k: None,
        )
        monkeypatch.setattr(train_adapters, "_import_wandb", lambda: wb_mod)

        vec_path = tmp_path / "universal_style_vector.pt"
        torch.save(torch.zeros(256), vec_path)
        cfg = config_mod.TrainConfig(universal_style_vector_path=str(vec_path))
        device = torch.device("cpu")
        monkeypatch.setattr(train_adapters, "build_mel_loss", lambda *args, **kwargs: train_adapters.MelReconstructionLoss())
        kmodel, gst, mhubert, sv_model, _disc, mel_loss_mod, kokoro_cfg = train_adapters.build_models(cfg, device)

        class _OneEval:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {}

        wandb_run = types.SimpleNamespace(log=lambda *a, **k: None)
        train_adapters._log_wandb_validation(
            wandb_run,
            dataset=_OneEval(),
            kmodel=kmodel,
            gst=gst,
            mhubert=mhubert,
            sv_model=sv_model,
            mel_loss_mod=mel_loss_mod,
            cfg=cfg,
            device=device,
            global_step=1,
            num_samples=1,
            vocab=kokoro_cfg["vocab"],
        )

        assert kmodel.last_forward_token_len == true_tokens
        assert mhubert.last_attention_mask is not None
        assert tuple(mhubert.last_attention_mask.shape) == (1, 200)
        assert int(mhubert.last_attention_mask[0, :ref_len].sum()) == ref_len
        assert int(mhubert.last_attention_mask[0, ref_len:].sum()) == 0
        assert sv_model.last_waveform_lengths is not None
        assert torch.equal(sv_model.last_waveform_lengths.cpu(), torch.tensor([ref_len]))
