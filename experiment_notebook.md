# Experiment Notebook

Canonical working note for the active gate experiment and the next queue. Update this file first when concluding a run or promoting a new baseline.

Last updated: 2026-04-30

## Current Experiment

Active question: does the fixed English-only MFA pipeline support a clean single-batch memorization gate before any larger English or multilingual runs?

### Run Identity

- Experiment type: active memorization gate
- Status: pending clean English rebuild
- Goal: verify that the current Phase-1 stack can fit a 6-speaker x 1-train-row English probe and show non-degenerate teacher-forced plus free-running validation behavior

### Proof Point Before This Gate

- The MFA duration projection bug is treated as fixed for English.
- The accepted proof point is the rebuilt English debug path:
  - `manifests/debug/en_train_64.phonemes.jsonl`
  - `manifests/debug/en_val_16.phonemes.jsonl`
- Those 64/16 English debug runs are the current evidence that the corrected projection path is good enough to scale to the full English rebuild.

### Config Delta From Baseline

- Training subset: `manifests/memorization_en_6x1.phonemes.jsonl`
- Validation subset: `manifests/memorization_en_6x1_val.phonemes.jsonl`
- Selected speakers: top 6 speakers present in both clean English train and val manifests after `prosody_enabled=True` filtering
- Train shape: 1 utterance per selected speaker from `en_train`
- Val shape: 1 different utterance per same selected speaker from `en_val`
- `batch_size = 6`
- `grad_accum_steps = 1`
- `warmup_steps = 200`
- `style_decoder_only_steps = 200`
- `gst_conv_kernel_size = 5`
- `gst_conv_stride = 2`
- `gst_conv_padding = 2`
- `checkpoint_interval = 25`
- `max_steps = 2000`
- `lambda_spk_contrastive = 5.0` (raised from 1.0 — see run post-mortem below)
- `contrastive_temperature = 0.1` (raised from 0.07 — see run post-mortem below)
- `grad_clip_norm_g = 1.0` (lowered from 5.0 — see run post-mortem below)

### Deterministic Subset Policy

- Source manifests:
  - full clean `manifests/languages/en_train.phonemes.jsonl`
  - full clean `manifests/languages/en_val.phonemes.jsonl`
- Allowed speakers: present in both splits with at least one usable `prosody_enabled=True` row in each split
- Speaker ranking:
  - train-row count descending
  - val-row count descending
  - speaker ID ascending
- Row selection within each chosen speaker:
  - `prosody_enabled=True` only
  - shortest text first
  - source index ascending as the final tie-break
- Metadata requirement:
  - write `.meta.json` with selected speaker IDs, source indices, source manifests, and the exact selection policy

### Success Criteria

Treat this gate as successful only if most of the following hold:

- `train/loss_mel` drops materially.
- `train/loss_spk_contrastive` trends down from early steps.
- `val_tf/loss_mel` drops clearly.
- `val_free/loss_mel` does not stay flat or fully decoupled from `val_tf/loss_mel`.
- Validation audio table shows intelligible teacher-forced speech and speaker-dependent differences.
- No NaN/Inf events or chronic optimization instability.

For failure interpretation, see "Validation divergence" in `CLAUDE.md`.

## Required Coverage Reporting

Record this after the full English rebuild and before subset generation:

- English summary:
  - total rows
  - alignment-attempted rows
  - coverage-accepted rows
  - coverage-rejected rows
  - rejection reasons by category
- Gate condition:
  - do not generate `memorization_en_6x1*` from stale or mixed multilingual caches
  - do not train until the English acceptance/rejection summary looks sane and only coverage-accepted rows remain `prosody_enabled=True`

## W&B Logging Policy

See `CLAUDE.md` for the canonical scalar and diagnostic list. Additional audio table columns beyond what `CLAUDE.md` lists:

- `gt_samples_full`
- `gt_samples_tf`
- `pred_samples_free`
- `pred_samples_tf`

Metrics to remove from W&B:

- per-item `val_free/audio_pred_*`
- per-item `val_tf/audio_pred_*`
- per-item `val_free/audio_gt_*`
- per-item `val_tf/audio_gt_*`
- per-item `val/text_*`
- `audio_samples_*`
- `duration_frames_*`
- collapse min/max metrics
- universal-prior norm constants
- duplicate export-only metrics that do not affect gate decisions

Local CSV export only for:

- `train/loss_mel`
- `train/loss_spk_contrastive`
- `val_free/loss_spk_contrastive`
- `val_tf/loss_spk_contrastive`

## Current Experiment Results

### Run: `memorization_en_6x1-train-20260429-024824` — FAILED

**Manifest used:** `memorization_en_6x1_min.phonemes.jsonl` (min variant, not the target full `en_6x1` — full rebuild still pending)
**Steps:** 500 | **Batch size:** 6 | **Max steps:** 500 | **Warmup steps:** 500

**Observed behavior:**
- `train/loss_mel`: 4.05 → 2.0 by step ~150, then hard plateau through step 500.
- `train/loss_spk_contrastive`: highly unstable; spiked to 3.95 at steps 38–41, never consistently converged.
- `val_free/loss_spk_contrastive`: WORSENED from 1.19 at step 25 to 2.05 at step 500.
- Audio: collapsed to droning noise — confirmed style collapse.

**Key diagnostic metrics at step 500:**

| Metric | Value | Threshold concern |
|--------|-------|-------------------|
| `train/collapse/ref_s_pairwise_cos_mean` | 0.818 | >0.5 = severe collapse |
| `val_free/collapse/ref_s_pairwise_cos_mean` | 0.919 | >0.5 = severe collapse |
| `train/grad_norm_g` (pre-clip) | 1367 | clip was 5.0 → 273× clip threshold |
| `train/gst/proj_dec_norm_mean` | 5.92 | unbounded growth |
| `train/gst/proj_pred_norm_mean` | 7.23 | unbounded growth |

**Root causes identified (ordered by confidence):**

1. **warmup_steps = max_steps = 500** — the entire run was inside the LR warmup ramp (1e-7 → 1e-4). No step ever trained at the target LR. This alone invalidates the run as a meaningful gate test.

2. **Loss imbalance (20:1 mel:spk)** — `lambda_mel=20` vs `lambda_spk_contrastive=1` drives the model to find a single averaged style that satisfies mel reconstruction across all 6 speakers. The contrastive signal is too weak to maintain speaker diversity.

3. **Gradient explosion through the contrastive path** — `grad_through_input=True` in the WeSpeaker call lets gradients flow through the entire generated-waveform pipeline. Combined with `F.normalize + temperature=0.07 (14.3× amplification)`, this created the observed 1367-norm gradient explosion that was never resolved even after clipping.

4. **style_decoder_only_steps=0** — no warmup guard. Both `to_style_dec` and `to_style_pred` trained simultaneously from step 0. The pred projection grew larger (7.23) than dec (5.92), consistent with pred destabilizing the dec branch.

5. **Temperature 0.07 too aggressive** — for batch_size=6, this creates a very sharp loss surface. Any small perturbation in embedding direction is amplified 14.3×, causing the observed high-variance oscillation.

**What was confirmed NOT a cause:** `universal_style_vector.pt` exists and is non-zero (shape (256,), norm 1.87).

**Config changes applied for next run** (now defaults in `voice_clone/config.py`):
- `warmup_steps`: 500 → 200
- `style_decoder_only_steps`: 0 → 200
- `lambda_spk_contrastive`: 1.0 → 5.0
- `contrastive_temperature`: 0.07 → 0.1
- `grad_clip_norm_g`: 5.0 → 1.0

**Status:** DISCARDED. Config updated. Waiting for full English rebuild before re-running gate.

## Next Experiment Queue

1. 6x1 English memorization gate.
2. Larger English memorization subset if the 6x1 gate passes.
3. Multilingual rebuild and re-entry only after English passes.

## Update Rule

When a run finishes:

1. Update `Current Experiment Results`.
2. Mark the run as promoted, discarded, or revisited.
3. If promoted, update `architecture.md`, `CLAUDE.md`, and `Current Experiment`.
4. Keep only the next 3 concrete experiments in `Next Experiment Queue`.
