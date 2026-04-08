"""Checkpoint save/load roundtrip tests.

Marked @pytest.mark.slow because they build the full model stack.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def stack_and_opts():
    """Build full stack + optimizers once."""
    from voice_clone.config import TrainConfig
    from voice_clone.train_adapters import (
        build_models,
        generator_trainable_parameters,
        save_checkpoint,
        load_checkpoint,
    )

    device = torch.device("cpu")
    cfg = TrainConfig()
    kmodel, gst, sv_model, disc, mel_loss, kokoro_cfg = build_models(cfg, device)
    params_g = generator_trainable_parameters(kmodel, gst)
    opt_g = torch.optim.AdamW(params_g, lr=cfg.lr_g)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d)
    return {
        "kmodel": kmodel,
        "gst": gst,
        "sv_model": sv_model,
        "disc": disc,
        "opt_g": opt_g,
        "opt_d": opt_d,
        "cfg": cfg,
        "device": device,
    }


class TestSaveLoadRoundtrip:
    def test_step_preserved(self, stack_and_opts, tmp_path):
        from voice_clone.train_adapters import save_checkpoint, load_checkpoint
        from voice_clone.config import TrainConfig
        from voice_clone.train_adapters import build_models, generator_trainable_parameters

        s = stack_and_opts
        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path, gst=s["gst"], disc=s["disc"], kmodel=s["kmodel"],
            opt_g=s["opt_g"], opt_d=s["opt_d"], step=42, cfg=s["cfg"],
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "waveform_discriminator" in ckpt

        device = s["device"]
        cfg2 = TrainConfig()
        km2, gst2, _, disc2, _, _ = build_models(cfg2, device)
        params_g2 = generator_trainable_parameters(km2, gst2)
        og2 = torch.optim.AdamW(params_g2, lr=cfg2.lr_g)
        od2 = torch.optim.AdamW(disc2.parameters(), lr=cfg2.lr_d)

        step = load_checkpoint(
            ckpt_path, gst=gst2, disc=disc2, kmodel=km2,
            opt_g=og2, opt_d=od2, device=device,
        )
        assert step == 42

    def test_gst_state_dict_matches(self, stack_and_opts, tmp_path):
        from voice_clone.train_adapters import save_checkpoint

        s = stack_and_opts
        ckpt_path = tmp_path / "ckpt_gst.pt"

        # Perturb GST so it's not default
        with torch.no_grad():
            s["gst"].to_ref_s.weight.fill_(0.123)

        save_checkpoint(
            ckpt_path, gst=s["gst"], disc=s["disc"], kmodel=s["kmodel"],
            opt_g=s["opt_g"], opt_d=s["opt_d"], step=10, cfg=s["cfg"],
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved_sd = ckpt["segment_gst"]
        live_sd = s["gst"].state_dict()
        for key in live_sd:
            assert key in saved_sd
            assert torch.equal(live_sd[key], saved_sd[key]), f"Mismatch in GST key {key}"

    def test_adapter_state_dicts_present(self, stack_and_opts, tmp_path):
        from voice_clone.train_adapters import save_checkpoint

        s = stack_and_opts
        ckpt_path = tmp_path / "ckpt_adapters.pt"
        save_checkpoint(
            ckpt_path, gst=s["gst"], disc=s["disc"], kmodel=s["kmodel"],
            opt_g=s["opt_g"], opt_d=s["opt_d"], step=1, cfg=s["cfg"],
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["duration_adapters"] is not None
        assert ckpt["decoder_adapters"] is not None
        assert ckpt["generator_adapters"] is not None


class TestTrainConfigFromCheckpointDict:
    def test_reconstructs_config_from_asdict(self):
        from voice_clone.config import TrainConfig, LossWeights, MelLossConfig
        from voice_clone.infer import train_config_from_checkpoint_dict

        cfg = TrainConfig(
            lr_g=2e-4,
            loss_weights=LossWeights(lambda_mel=2.0, lambda_spk=0.3, lambda_adv=0.2, lambda_fm=1.5),
            mel=MelLossConfig(n_fft=2048),
        )
        d = asdict(cfg)
        restored = train_config_from_checkpoint_dict(d)
        assert restored.lr_g == 2e-4
        assert restored.loss_weights.lambda_mel == 2.0
        assert restored.loss_weights.lambda_adv == 0.2
        assert restored.loss_weights.lambda_fm == 1.5
        assert restored.mel.n_fft == 2048

    def test_ignores_unknown_keys(self):
        from voice_clone.config import TrainConfig
        from voice_clone.infer import train_config_from_checkpoint_dict

        d = asdict(TrainConfig())
        d["unknown_future_field"] = 999
        restored = train_config_from_checkpoint_dict(d)
        assert not hasattr(restored, "unknown_future_field")


class TestApplyVoiceCloneCheckpoint:
    def test_loads_gst_and_adapters(self, stack_and_opts, tmp_path):
        from voice_clone.train_adapters import save_checkpoint, build_models
        from voice_clone.infer import apply_voice_clone_checkpoint
        from voice_clone.config import TrainConfig

        s = stack_and_opts
        ckpt_path = tmp_path / "ckpt_apply.pt"

        # Perturb
        with torch.no_grad():
            s["gst"].bank.fill_(0.5)

        save_checkpoint(
            ckpt_path, gst=s["gst"], disc=s["disc"], kmodel=s["kmodel"],
            opt_g=s["opt_g"], opt_d=s["opt_d"], step=5, cfg=s["cfg"],
        )

        device = s["device"]
        cfg2 = TrainConfig()
        km2, gst2, _, _, _, _ = build_models(cfg2, device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        apply_voice_clone_checkpoint(ckpt, gst=gst2, kmodel=km2)

        assert torch.equal(gst2.bank.data, s["gst"].bank.data)

    def test_missing_generator_adapters_key_ok(self, stack_and_opts, tmp_path):
        """Checkpoint without generator_adapters loads without crash."""
        from voice_clone.train_adapters import save_checkpoint, build_models
        from voice_clone.infer import apply_voice_clone_checkpoint
        from voice_clone.config import TrainConfig

        s = stack_and_opts
        ckpt_path = tmp_path / "ckpt_missing.pt"
        save_checkpoint(
            ckpt_path, gst=s["gst"], disc=s["disc"], kmodel=s["kmodel"],
            opt_g=s["opt_g"], opt_d=s["opt_d"], step=1, cfg=s["cfg"],
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        del ckpt["generator_adapters"]

        device = s["device"]
        cfg2 = TrainConfig()
        km2, gst2, _, _, _, _ = build_models(cfg2, device)
        apply_voice_clone_checkpoint(ckpt, gst=gst2, kmodel=km2)
