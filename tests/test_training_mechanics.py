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
    from voice_clone.losses import SLMFeatureDiscriminator

    device = torch.device("cpu")
    cfg = TrainConfig()
    kmodel, gst, wavlm, disc, mel_loss, kokoro_cfg = build_models(cfg, device)
    return {
        "kmodel": kmodel,
        "gst": gst,
        "wavlm": wavlm,
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

    def test_wavlm_all_frozen(self, full_stack):
        wavlm = full_stack["wavlm"]
        for name, p in wavlm.named_parameters():
            assert not p.requires_grad, f"WavLM param {name} should be frozen"


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
    def test_no_slm_during_warmup(self, full_stack):
        """generator_loss_backward with include_slm=False gives loss_slm_g=0."""
        from voice_clone.train_adapters import generator_loss_backward
        from voice_clone.wavlm_sv import WavLMSVOutput
        from voice_clone.config import LossWeights

        device = full_stack["device"]
        kmodel = full_stack["kmodel"]
        gst = full_stack["gst"]
        disc = full_stack["disc"]
        mel_loss = full_stack["mel_loss"]
        cfg = full_stack["cfg"]

        # Synthetic WavLM-like output
        fake_pooled = torch.randn(1, 512)
        fake_frames = torch.randn(1, 20, 768, requires_grad=True)
        fake_mask = torch.ones(1, 20)
        fake_attn_mask = torch.ones(1, 24_000)
        wv_gen = WavLMSVOutput(
            pooled_embedding=fake_pooled,
            frame_hidden_states=fake_frames,
            frame_mask=fake_mask,
            attention_mask=fake_attn_mask,
        )

        opt_g = torch.optim.AdamW(
            [p for p in list(gst.parameters()) + list(kmodel.parameters()) if p.requires_grad],
            lr=1e-4,
        )
        opt_g.zero_grad()

        pred_wav = torch.randn(1, 24_000, requires_grad=True)
        ref_pooled = torch.randn(1, 512)
        tgt_24 = torch.randn(1, 24_000)

        metrics = generator_loss_backward(
            kmodel, gst, wv_gen, disc, mel_loss, opt_g,
            pred_wav, ref_pooled, tgt_24,
            cfg.loss_weights,
            scaler=None, use_amp=False,
            include_slm=False,
        )
        assert metrics["loss_slm_g"] == 0.0

    def test_disc_params_frozen_during_g_step(self, full_stack):
        """During generator_loss_backward with include_slm=True, disc requires_grad is toggled off then restored."""
        from voice_clone.train_adapters import generator_loss_backward
        from voice_clone.wavlm_sv import WavLMSVOutput

        device = full_stack["device"]
        kmodel = full_stack["kmodel"]
        gst = full_stack["gst"]
        disc = full_stack["disc"]
        mel_loss = full_stack["mel_loss"]
        cfg = full_stack["cfg"]

        for p in disc.parameters():
            assert p.requires_grad, "D should start trainable"

        fake_pooled = torch.randn(1, 512)
        fake_frames = torch.randn(1, 20, 768, requires_grad=True)
        fake_mask = torch.ones(1, 20)
        fake_attn_mask = torch.ones(1, 24_000)
        wv_gen = WavLMSVOutput(
            pooled_embedding=fake_pooled,
            frame_hidden_states=fake_frames,
            frame_mask=fake_mask,
            attention_mask=fake_attn_mask,
        )

        opt_g = torch.optim.AdamW(
            [p for p in list(gst.parameters()) + list(kmodel.parameters()) if p.requires_grad],
            lr=1e-4,
        )
        opt_g.zero_grad()

        pred_wav = torch.randn(1, 24_000, requires_grad=True)
        ref_pooled = torch.randn(1, 512)
        tgt_24 = torch.randn(1, 24_000)

        generator_loss_backward(
            kmodel, gst, wv_gen, disc, mel_loss, opt_g,
            pred_wav, ref_pooled, tgt_24,
            cfg.loss_weights,
            scaler=None, use_amp=False,
            include_slm=True,
        )

        # After the call, disc requires_grad should be restored
        for p in disc.parameters():
            assert p.requires_grad, "D requires_grad should be restored after G step"


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
