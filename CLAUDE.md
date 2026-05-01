# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Voice-Clone-Kokoro-TTS

A parameter efficient TTS model capable of zero-shot voice cloning with multilingual support. The system trains adapter layers (SegmentGST) on top of frozen Kokoro and mHuBERT backbones.

## Training Workflow Overview

See `architecture.md` for the full stack description. Key structural facts:

- **Frozen Kokoro backbone** (`hexgrad/Kokoro-82M`) — all parameters frozen via `freeze_kokoro_backbone(...)`
- **Frozen mHuBERT encoder** (`utter-project/mHuBERT-147-base-3rd-iter`) — extracts frame hidden states; training reads cached states from disk
- **SegmentGST** — the *only* trainable generator-side module; projects mHuBERT frame states to Kokoro's 256-D `ref_s` style vector as a residual offset on top of a universal style prior
- **Frozen WeSpeaker** — extracts speaker embeddings for contrastive loss
- **HiFiGAN MPD/MSD discriminator** — present but inactive by default (`disc_start_step = 99_999_999`)

All generator gradients flow through the single 256-D `ref_s` bottleneck.

### Training Pipeline

Each training row processes `ref_wav`, `target_wav`, `text`, and `lang_code`. Per step:

1. Text → phonemes → Kokoro `input_ids` (BOS/EOS padded, ≤ 512 tokens)
2. Reference audio → frozen mHuBERT (cached) → frame hidden states `(B, T_frames, 768)`
3. Frame states → SegmentGST → 256-D `ref_s = (u_dec + Δ_dec) ⊕ (u_pred + Δ_pred)`
4. `input_ids + ref_s` → `KModel.forward_with_tokens(...)` (per-item loop, padded back into batch tensors)
5. Generated 24 kHz waveform vs target → multi-resolution STFT mel-warped reconstruction loss (`auraloss`)
6. Generated waveform → frozen WeSpeaker → InfoNCE contrastive loss against cached target embedding

### Loss Function

`L_G = λ_mel * L_mel + λ_spk * L_spk + λ_dur * L_dur + λ_f0 * L_f0`

Current defaults in `voice_clone/config.py`:
- `lambda_mel = 20.0`
- `lambda_spk_contrastive = 5.0` (raised from 1.0 — 20:1 imbalance caused style collapse in 2026-04-29 run)
- `lambda_dur = 0.0` (inactive)
- `lambda_f0 = 0.0` (diagnostic only)
- `disc_start_step = 99_999_999` (adversarial training off)
- `contrastive_temperature = 0.1` (raised from 0.07 — 14.3× gradient amplification contributed to explosion)
- `grad_clip_norm_g = 1.0` (lowered from 5.0 — pre-clip norm reached 1367 in 2026-04-29 run)
- `warmup_steps = 200` (lowered from 500 — prevents entire short runs from being inside warmup)
- `style_decoder_only_steps = 200` (raised from 0 — pred branch was destabilizing dec branch)

### Caches

Training requires three offline artefacts, all anchored on the manifest stem:

| Directory | Builder | What it stores |
|-----------|---------|----------------|
| `alignments/{stem}/` | `voice_clone/alignment/mfa_pipeline.py` | Raw MFA TextGrids |
| `prosody_cache/{stem}/{row}.pt` | `voice_clone/prosody_targets_builder.py` | `gt_dur_frames`, `duration_targets`, F0 (placeholder), `prosody_enabled`, coverage metadata |
| `cache/{stem}/{row}.pt` | `voice_clone/cache_builder.py` | mHuBERT hidden states, WeSpeaker embeddings, prosody payload |

Schema string `phase1_mfa_v2` is enforced; stale or legacy rows fail fast. Coverage acceptance band is `0.90 ≤ duration_coverage_ratio ≤ 1.10`; rows outside the band load with `prosody_enabled=False`.

### Current Experiment

See `experiment_notebook.md` for the active gate, success criteria, and all live analysis. CLAUDE.md only mirrors the prerequisite order.

**Active Gate:** 6×1 English memorization gate (status: *pending clean English rebuild*).

**Target manifests** (not yet materialized): `manifests/memorization_en_6x1.phonemes.jsonl` and `_val` companion. Only `manifests/memorization_en_6x1_min.phonemes.jsonl` and its `_val` currently exist on disk. Source manifests: `manifests/languages/en_train.phonemes.jsonl` and `en_val.phonemes.jsonl`.

**Prerequisite order** before gate run:
1. Rebuild full English phoneme manifests (`scripts/build_phonemes.sh`).
2. Rebuild full English alignments, prosody cache, and feature cache (`scripts/rebuild_all_caches.sh`).
3. Generate memorization subsets with the deterministic policy in `experiment_notebook.md`.
4. Train the 6×1 English memorization gate.

## Development Commands

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements-training.txt
python scripts/verify_training_env.py
```

### Running Training

`kokoro/` lives directly under the repo root and is imported as a sibling package, so only the repo root needs to be on `PYTHONPATH`. `scripts/train.py` sets this for you.

> **Launcher vs raw CLI:** `scripts/train.py` is the recommended entrypoint. It defaults `EPOCHS=500`, always passes `--amp`, sets sensible `PYTORCH_(CUDA|HIP)_ALLOC_CONF`, and forwards env vars to flags. The raw `voice_clone.train_adapters` CLI defaults `--epochs 1`, `--device cpu`, and AMP off — useful for tests, but don't reach for it in normal training.

**Smoke run:**
```bash
python scripts/train.py --max-steps 75
```

**Single cached step (no launcher, no W&B):**
```bash
python scripts/smoke_cached_train_step.py --manifest manifests/memorization_en_6x1_min.phonemes.jsonl --device cuda
```

**Memorization training (once manifests exist):**
```bash
MANIFEST=manifests/memorization_en_6x1.phonemes.jsonl \
VAL_MANIFEST=manifests/memorization_en_6x1_val.phonemes.jsonl \
EPOCHS=200 \
CHECKPOINT_INTERVAL=25 \
python scripts/train.py
```

**Override defaults via environment variables:**
```bash
export DEVICE=cuda
export BATCH_SIZE=4
export CHECKPOINT_INTERVAL=50
export WANDB_PROJECT="Voice-Clone-Kokoro-TTS"
export WANDB_RUN_NAME="my-voice-clone-exp"
export CKPT_DIR="ckpt/my-voice-clone"
python scripts/train.py
```

### Generate Audio from a Checkpoint

```bash
# Module CLI: explicit, scriptable
python -m voice_clone.infer \
  --checkpoint ckpt/memorization_en_6x1/checkpoint_500.pt \
  --ref-wav   data/en/clips/some_speaker.wav \
  --text      "Hello, world!" \
  --lang      a \
  --out       output/sample.wav

# Convenience wrapper with auto-device detection
python scripts/run_example.py \
  --checkpoint ckpt/memorization_en_6x1/checkpoint_500.pt \
  --ref-wav    data/en/clips/some_speaker.wav \
  --text       "Hello, world!" \
  --lang       a
```

Inference always uses Kokoro's free-running duration head; there is no teacher forcing at inference time.

### Key Arguments

Defaults shown are for `python -m voice_clone.train_adapters`. Defaults applied by `scripts/train.py` are noted in parentheses.

- `--manifest` (required): Path to JSONL manifest.
- `--manifest-root`: Optional base path for resolving relative audio paths.
- `--val-manifest`: Optional validation manifest. Launcher auto-clears it if the file does not exist.
- `--kokoro-repo`: HF repo ID for the backbone (default: `hexgrad/Kokoro-82M`).
- `--epochs`: Default `1` in the CLI, **`500` via the launcher**. Cap actual training with `--max-steps`.
- `--max-steps`: Cap training at N steps (for smoke runs).
- `--device`: `cuda` | `cpu` | `mps`. **Default is `cpu` in the CLI** — pass `DEVICE=cuda` or `--device cuda` explicitly.
- `--batch-size`: Batch size per micro-step (default: 6). `speaker_contrastive_loss` requires batch ≥ 2.
- `--grad-accum-steps`: Gradient accumulation (default: 1).
- `--checkpoint-interval`: Save checkpoint every N steps (default: 100).
- `--save-final-checkpoint` / `--no-save-final-checkpoint`: Launcher exposes this via `SAVE_FINAL_CHECKPOINT={0,1}`.
- `--wandb`: Enable W&B logging. Launcher always passes this.
- `--amp`: **Off by default in the CLI**. Launcher always passes `--amp`. Forces fp32 for STFT regardless because `disable_amp_for_stft=True`.
- `--style-decoder-only-steps`: When `> 0`, pins `ref_s[:, 128:]` to the universal prior for a warmup window; decoder half always trains. `0` (default) disables. Naming is historical.
- `--gst-conv-kernel-size` / `--gst-conv-stride` / `--gst-conv-padding`: SegmentGST temporal-conv geometry overrides.
- `--lambda-mel`, `--lambda-spk-contrastive`, `--contrastive-temperature`, `--grad-clip-norm-g`: Per-run loss-weight / clip overrides. Env-var equivalents: `LAMBDA_MEL`, `LAMBDA_SPK_CONTRASTIVE`, `CONTRASTIVE_TEMPERATURE`, `GRAD_CLIP_NORM_G`.

## W&B Logging Policy

**Train scalars:**
- `train/loss_total`
- `train/loss_mel`
- `train/loss_spk_contrastive`
- `train/grad_norm_g`
- `train/lr_g`

**Validation scalars:**
- `val_free/loss_total`, `val_free/loss_mel`, `val_free/loss_spk_contrastive`, `val_free/len_ratio`
- `val_tf/loss_total`, `val_tf/loss_mel`, `val_tf/loss_spk_contrastive`, `val_tf/len_ratio`
- `val_gap/loss_total`, `val_gap/loss_mel`, `val_gap/len_ratio`

**Collapse diagnostics:**
- `collapse/ref_s_std_mean`
- `collapse/ref_s_delta_norm_mean`
- `collapse/ref_s_pairwise_cos_mean`

Read together: low std + high pairwise cos = style collapse; `delta_norm_mean → 0` = projection glued to universal prior.

Empirical collapse thresholds (from 2026-04-29 failure): `pairwise_cos_mean > 0.5` = significant collapse; `> 0.8` = severe/audio-degrading. `delta_norm_mean > 5` with growing `proj_*_norm_mean` = unbounded residual growth, usually accompanied by gradient explosion.

**GST projection diagnostics:**
- `gst/proj_dec_norm_mean`
- `gst/proj_pred_norm_mean`

L2 norms of `Δ_dec` and `Δ_pred`. Runaway growth during warmup usually means LR is too high or `style_decoder_only_steps` is needed.

**Validation media:**
- `val/audio_table`: step, row_index, speaker_id, text, coverage_ratio, gt_audio, pred_audio_free, pred_audio_tf

## Architecture Notes

See `architecture.md` for the full stack description and `voice_clone/segment_gst.py` for implementation details.

### SegmentGST

- Residual style encoder: output is `concat(u_dec + Δ_dec, u_pred + Δ_pred)` where `u` is a persistent universal style prior and `Δ` are zero-initialized residuals.
- `ref_dim = 256 = 2 × style_dec_dim`; `ref_s[:, :128]` → decoder branch, `ref_s[:, 128:]` → prosody predictor.
- Zero-init means Kokoro receives an in-distribution style vector at step 0, avoiding cold-start instability.
- `style_decoder_only_steps > 0` pins `Δ_pred` to zero for a warmup window; use when prosody-side gradients destabilize the decoder fit.

### mHuBERT

- Frozen, eval-mode. Extracts frame hidden states `(B, T_frames, 768)`.
- `TrainConfig.mhubert_extract_layer = 6`. **The encoder's own constructor default is `extract_layer=9`.** Pass layer explicitly in any tooling that builds `MHuBERTEncoder()` directly.
- Training reads precomputed states from disk; only the cache builder and inference path run mHuBERT online.

### Kokoro Integration

- `KModel.forward_with_tokens(input_ids, ref_s, ...)` produces 24 kHz waveforms.
- `ref_s` shape `(B, 256)`: `[:, :128]` conditions the decoder, `[:, 128:]` conditions the prosody predictor.

### Duration Semantics

- **Free-running:** Kokoro's frozen duration head predicts per-token durations; rounds them for expansion.
- **Teacher-forced (training only):** `gt_dur_frames` from MFA cache replaces rounded durations when `prosody_enabled=True`; otherwise total length is rescaled to match target.
- **Unit:** 1 duration frame = 1/40 s = 600 waveform samples at 24 kHz.

### Loss Components (`voice_clone/losses.py`)

- `MelReconstructionLoss`: Multi-resolution STFT with mel-warped magnitude via `auraloss`. STFT is **forced to fp32** regardless of AMP — fp16 STFT is numerically unstable on ROCm/CUDA.
- `speaker_contrastive_loss`: InfoNCE, temperature 0.07. **Requires batch ≥ 2.**
- `duration_loss_log_space`: MSE in `log1p` space. Active only when `lambda_dur > 0` and `prosody_enabled=True`.
- `masked_l1_loss` (F0): **Not production-ready** — F0 targets are placeholder quality. Keep `lambda_f0 = 0`.
- GAN losses: dormant until `disc_start_step` is reached.

## Performance Tuning

### AMP

- `--amp` is off by default in the raw CLI; the launcher always passes it.
- `disable_amp_for_stft = True` is enforced inside `MelReconstructionLoss` regardless of outer autocast — do not change this.
- For numerical-stability debugging, drop `--amp` (raw CLI) to run fp32 everywhere.

### Mixed Language Training

Supported but requires:
- Every row's `lang_code` supported by the same `--kokoro-repo` G2P stack and vocabulary. Switching checkpoints in a single job is unsafe because vocabularies and `n_token` differ across Kokoro variants.
- `TrainConfig.min_language_speakers = 2`: languages with fewer eligible speakers are dropped so contrastive batches always have a valid negative.
- Phonemes missing from the vocab are silently dropped during `phonemes_to_input_ids`. Validate phonemization for rare languages.

### Cache Layout

```
alignments/{manifest_stem}/...                  # MFA TextGrids
prosody_cache/{manifest_stem}/{row_index}.pt    # gt_dur_frames, F0, prosody_enabled, coverage
cache/{manifest_stem}/{row_index}.pt            # mHuBERT hidden states, WeSpeaker embeddings, prosody payload
```

`{manifest_stem}` = manifest filename without `.jsonl`. Row indices match manifest line order.

### Checkpoint Schema

`save_checkpoint` writes: `step, train_config, segment_gst, waveform_discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d, scheduler_g, scheduler_d, generator_updates, discriminator_updates`.

Legacy keys (`kokoro_lora`, `duration_adapters`, `decoder_adapters`, `generator_adapters`) are **explicitly rejected** — old LoRA checkpoints cannot be migrated; retrain on the SegmentGST graph.

## Common Issues

### Stale or missing cache

```
ValueError: Stale feature cache at cache/<stem>/<row>.pt
ValueError: Cache row <path> missing required keys: [...]
ValueError: Cache row <path> has unsupported feature cache schema '...'
FileNotFoundError: Feature cache row not found: <path>
```

Fix: `scripts/rebuild_all_caches.sh MANIFEST=...`. Targeted rebuilds: `scripts/build_mfa_alignments.sh`, `scripts/build_prosody_targets.sh`, `scripts/build_feature_cache.sh`. Delete `cache/{stem}/`, `prosody_cache/{stem}/`, `alignments/{stem}/` subtrees for a clean rebuild.

### Out-of-vocabulary phonemes

```
ValueError: Phoneme sequence produced no in-vocabulary tokens after filtering unknown graphemes.
ValueError: Phoneme string length <N> exceeds max_phoneme_chars=510.
ValueError: Phoneme sequence too long for model context: <N> > <context_length>
```

Ensure the manifest's `phonemes` field uses Kokoro's `config.json` vocab for the configured `--kokoro-repo`.

### Memory OOM

1. Reduce `BATCH_SIZE`.
2. Increase `GRAD_ACCUM_STEPS` to compensate.
3. Confirm `PYTORCH_(CUDA|HIP)_ALLOC_CONF=expandable_segments:True` is set (launcher does this automatically).

### NaN losses

1. Confirm `target_wav` is finite at load time (`load_audio_mono` raises on non-finite samples).
2. Watch `gst/proj_dec_norm_mean` and `gst/proj_pred_norm_mean` for runaway growth; consider raising `style_decoder_only_steps`.
3. Drop `--amp` (raw CLI) to debug in fp32.

### Validation divergence (canonical decision rule)

- **`val_tf` fails:** acoustic or conditioning path is broken.
- **`val_tf` improves, `val_free` does not:** duration inference is the blocker.
- **`train/loss_spk_contrastive` stays flat:** batch composition or `style_decoder_only_steps` too long.

### Trying to enable `lambda_dur` or `lambda_f0`

- `lambda_dur > 0` is only meaningful for rows with `prosody_enabled=True`.
- `lambda_f0 > 0` is not ready: F0 targets are placeholder quality (see `research/future_plan.md` §1.2–1.6).

## Files to Read First

For **architecture reasoning** and **failure analysis**:
1. `architecture.md` — current stack, duration semantics, validation split.
2. `experiment_notebook.md` — active gate, success criteria, W&B policy.

For **architecture changes** and **model implementation**:
1. `voice_clone/train_adapters.py` — main training loop, optimizer/scheduler, checkpoint I/O.
2. `voice_clone/segment_gst.py` — adapter architecture, residual prior, bipartite split.
3. `voice_clone/config.py` — `TrainConfig`, `LossWeights`, `MelLossConfig` (defaults are authoritative).
4. `voice_clone/losses.py` — mel/MR-STFT, contrastive, duration, F0, GAN losses.
5. `voice_clone/mhubert_encoder.py` — frozen wrapper and frame-mask derivation.
6. `voice_clone/wespeaker_sv.py` — frozen speaker frontend; resample to 16 kHz happens inside.
7. `voice_clone/discriminators/hifigan.py` — HiFiGAN MPD/MSD; dormant unless `disc_start_step` is moved.
8. `voice_clone/infer.py` — canonical inference path and checkpoint schema enforcement.
9. `kokoro/kokoro/model.py` — `KModel.forward_with_tokens`, AdaIN, duration head; read-only.

For **writing new scripts and experiments**:
1. `scripts/train.py` — training launcher; env-var → CLI map.
2. `scripts/smoke_cached_train_step.py` — minimal forward/backward against an existing cache.
3. `scripts/rebuild_all_caches.sh` — alignments → prosody cache → feature cache, end to end.
4. `scripts/build_single_manifest_same_speaker_subset.py` — deterministic memorization-subset selection.
5. `voice_clone/DATASET.md` — manifest schema (authoritative).

## Writing a New Experiment or Script

1. **Manifest.** Build or extend a `*.jsonl` manifest. For memorization subsets, use `scripts/build_single_manifest_same_speaker_subset.py` for a reproducible `.meta.json`.
2. **Phonemize.** `scripts/build_phonemes.sh manifests/<name>.jsonl` → `manifests/<name>.phonemes.jsonl`.
3. **Caches.** `scripts/rebuild_all_caches.sh manifests/<name>.phonemes.jsonl`. Set `MANIFEST_ROOT=` if audio paths are relative. `SKIP_EXISTING=1` (default); pass `CLEAN=1` only when alignments are stale.
4. **Smoke check.** `python scripts/smoke_cached_train_step.py --manifest manifests/<name>.phonemes.jsonl --device cuda`. Failure here means data pipeline, not optimizer.
5. **Train.** Run via `scripts/train.py`. Keep run names machine-readable (`exp01-layer06-baseline`).
6. **Inspect.** `python scripts/run_example.py --checkpoint ... --ref-wav ...` for audio sanity.
7. **Record.** Append to `experiment_notebook.md`.

When *adding model code*, also touch:
- `voice_clone/config.py` (new flag → `TrainConfig` field).
- `voice_clone/train_adapters.py:parse_args` (new flag → CLI argument).
- `scripts/train.py` (new flag → env var → CLI passthrough).
- A test in `tests/`.

## Documentation Conventions

- All config defaults in `voice_clone/config.py` must be reflected in CLI help.
- Manifest schema is authoritative in `voice_clone/DATASET.md`.
- Environment variables map directly to CLI arguments via `scripts/train.py`.
- When the launcher adds new env-var handling, update both the launcher docstring and this file's *Key Arguments* / *Writing a New Experiment or Script* sections.
