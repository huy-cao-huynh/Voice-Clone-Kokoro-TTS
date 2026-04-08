"""Integration tests for training mechanics: freezing, gradient flow, phased training, warmup.

Marked @pytest.mark.slow because they build/download the full model stack.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def full_stack():
    """Build the full model stack once for the module (expensive)."""
    from voice_clone.config import TrainConfig
    from voice_clone.train_adapters import (
        build_models,
        freeze_kokoro_except_adapters,
        generator_trainable_parameters,
    )
    from voice_clone.segment_gst import SegmentGST

    device = torch.device("cpu")
    cfg = TrainConfig()
    kmodel, gst, xlsr, disc, mel_loss, kokoro_cfg = build_models(cfg, device)
    return {
        "kmodel": kmodel,
        "gst": gst,
        "xlsr": xlsr,
        "disc": disc,
        "mel_loss": mel_loss,
        "kokoro_cfg": kokoro_cfg,
        "cfg": cfg,
        "device": device,
    }


class TestFreezeKokoroExceptAdapters:
    def test_base_params_frozen(self, full_stack):
        kmodel = full_stack["kmodel"]
        adapter_ids = set()
        enc = kmodel.predictor.text_encoder
        if enc.adapters is not None:
            adapter_ids.update(id(p) for p in enc.adapters.parameters())
        if kmodel.decoder.decoder_adapters is not None:
            adapter_ids.update(id(p) for p in kmodel.decoder.decoder_adapters.parameters())
        if kmodel.decoder.generator.adapters is not None:
            adapter_ids.update(id(p) for p in kmodel.decoder.generator.adapters.parameters())

        for name, p in kmodel.named_parameters():
            if id(p) in adapter_ids:
                assert p.requires_grad, f"Adapter param {name} should be trainable"
            else:
                assert not p.requires_grad, f"Non-adapter param {name} should be frozen"

    def test_gst_params_all_trainable(self, full_stack):
        gst = full_stack["gst"]
        for name, p in gst.named_parameters():
            assert p.requires_grad, f"GST param {name} should be trainable"

    def test_xlsr_all_frozen(self, full_stack):
        xlsr = full_stack["xlsr"]
        for name, p in xlsr.named_parameters():
            assert not p.requires_grad, f"XLS-R param {name} should be frozen"


class TestGeneratorTrainableParameters:
    def test_returns_gst_plus_adapter_params(self, full_stack):
        from voice_clone.train_adapters import generator_trainable_parameters
        kmodel = full_stack["kmodel"]
        gst = full_stack["gst"]
        params = generator_trainable_parameters(kmodel, gst)

        gst_ids = {id(p) for p in gst.parameters()}
        kmodel_trainable_ids = {id(p) for p in kmodel.parameters() if p.requires_grad}
        returned_ids = {id(p) for p in params}

        assert returned_ids == gst_ids | kmodel_trainable_ids

    def test_no_frozen_params_included(self, full_stack):
        from voice_clone.train_adapters import generator_trainable_parameters
        kmodel = full_stack["kmodel"]
        gst = full_stack["gst"]
        params = generator_trainable_parameters(kmodel, gst)
        for p in params:
            assert p.requires_grad


class TestPhasedTraining:
    def test_no_gan_terms_during_warmup(self, full_stack):
        """When `real_disc_features=None`, generator_loss_backward returns zero adv/fm."""
        from voice_clone.train_adapters import generator_loss_backward

        device = full_stack["device"]
        kmodel = full_stack["kmodel"]
        gst = full_stack["gst"]
        disc = full_stack["disc"]
        mel_loss = full_stack["mel_loss"]
        cfg = full_stack["cfg"]

        opt_g = torch.optim.AdamW(
            [p for p in list(gst.parameters()) + list(kmodel.parameters()) if p.requires_grad],
            lr=1e-4,
        )
        opt_g.zero_grad(set_to_none=True)

        pred_wav = torch.randn(1, 24_000, requires_grad=True)
        ref_pooled = torch.randn(1, 1024)
        gen_pooled = torch.randn(1, 1024, requires_grad=True)
        tgt_24 = torch.randn(1, 24_000)

        metrics = generator_loss_backward(
            kmodel,
            gst,
            mel_loss,
            opt_g,
            pred_wav,
            ref_pooled,
            gen_pooled,
            tgt_24,
            disc_waveform=disc,
            real_disc_features=None,  # warmup: waveform D disabled
            weights=cfg.loss_weights,
            scaler=None,
            use_amp=False,
            scale_factor=1.0,
        )
        assert metrics["loss_adv_g"] == 0.0
        assert metrics["loss_fm"] == 0.0

    def test_disc_requires_grad_unchanged_during_g_step(self, full_stack):
        """generator_loss_backward should not toggle discriminator requires_grad flags."""
        from voice_clone.train_adapters import generator_loss_backward

        device = full_stack["device"]
        kmodel = full_stack["kmodel"]
        gst = full_stack["gst"]
        disc = full_stack["disc"]
        mel_loss = full_stack["mel_loss"]
        cfg = full_stack["cfg"]

        for p in disc.parameters():
            assert p.requires_grad, "D should start trainable"

        opt_g = torch.optim.AdamW(
            [p for p in list(gst.parameters()) + list(kmodel.parameters()) if p.requires_grad],
            lr=1e-4,
        )
        opt_g.zero_grad(set_to_none=True)

        pred_wav = torch.randn(1, 24_000, requires_grad=True)
        ref_pooled = torch.randn(1, 1024)
        gen_pooled = torch.randn(1, 1024, requires_grad=True)
        tgt_24 = torch.randn(1, 24_000)

        with torch.no_grad():
            _real_logits, real_feats = disc(tgt_24)

        before = [p.requires_grad for p in disc.parameters()]

        generator_loss_backward(
            kmodel,
            gst,
            mel_loss,
            opt_g,
            pred_wav,
            ref_pooled,
            gen_pooled,
            tgt_24,
            disc_waveform=disc,
            real_disc_features=real_feats,
            weights=cfg.loss_weights,
            scaler=None,
            use_amp=False,
            scale_factor=1.0,
        )

        after = [p.requires_grad for p in disc.parameters()]
        assert before == after
        for p in disc.parameters():
            assert p.requires_grad, "D requires_grad should remain enabled after G step"


class TestWarmupLR:
    def test_lr_reaches_full_after_warmup(self):
        """_build_scheduler warmup ramps from start_factor*lr to lr over warmup_steps."""
        from voice_clone.train_adapters import _build_scheduler

        lr = 1e-4
        warmup_steps = 50
        total_steps = 500
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = _build_scheduler(opt, warmup_steps, total_steps, lr_min=1e-6)
        for _ in range(warmup_steps):
            sched.step()
        actual_lr = opt.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(lr, rel=1e-4)

    def test_lr_starts_low(self):
        from voice_clone.train_adapters import _build_scheduler

        lr = 1e-4
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = _build_scheduler(opt, warmup_steps=50, total_steps=500, lr_min=1e-6)
        actual_lr = opt.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(lr * 1e-3, rel=1e-5)

    def test_lr_decays_after_warmup(self):
        """After warmup, cosine annealing should decrease LR below peak."""
        from voice_clone.train_adapters import _build_scheduler

        lr = 1e-4
        warmup_steps = 50
        total_steps = 500
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = _build_scheduler(opt, warmup_steps, total_steps, lr_min=1e-6)
        for _ in range(warmup_steps + 100):
            sched.step()
        actual_lr = opt.param_groups[0]["lr"]
        assert actual_lr < lr, "LR should decay below peak after warmup"
        assert actual_lr > 1e-6, "LR should be above minimum mid-training"

    def test_lr_reaches_minimum_at_end(self):
        """At total_steps, LR should be at or near lr_min."""
        from voice_clone.train_adapters import _build_scheduler

        lr = 1e-4
        lr_min = 1e-6
        warmup_steps = 50
        total_steps = 500
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = _build_scheduler(opt, warmup_steps, total_steps, lr_min=lr_min)
        for _ in range(total_steps):
            sched.step()
        actual_lr = opt.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(lr_min, rel=1e-3)

    def test_warmup_only_when_no_total_steps(self):
        """When total_steps is None, only LinearLR warmup is used (no cosine decay)."""
        from voice_clone.train_adapters import _build_scheduler

        lr = 1e-4
        warmup_steps = 50
        dummy = nn.Linear(10, 10)
        opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
        sched = _build_scheduler(opt, warmup_steps, total_steps=None, lr_min=1e-6)
        for _ in range(warmup_steps + 100):
            sched.step()
        actual_lr = opt.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(lr, rel=1e-5), "LR should stay at peak with no cosine phase"
