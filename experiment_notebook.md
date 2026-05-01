# Experiment Notebook

Canonical working note for the active gate experiment and the next queue. Update this file first when concluding a run or promoting a new baseline.

Last updated: 2026-05-01

## Current Experiment

Active question: can detaching the waveform before WeSpeaker (breaking the gradient explosion path) combined with a feature swap to WeSpeaker frame features fix the aggregation collapse identified in the diagnostic run?

### Run Identity

- Experiment type: candidate fix run — gradient detach + feature swap
- Status: ready to design (pending WeSpeaker frame extraction implementation)
- Goal: confirm that fixing the two root causes (gradient explosion + query diffusion in aggregation) allows the model to learn speaker-discriminative style vectors on the `memorization_en_6x1_min` set

### Scope Caveat

Continue on the existing `memorization_en_6x1_min` cache (12 rows — 6 train, 6 val). This is **not** the intended deterministic-policy en_6x1 set. All conclusions are scoped to "behavior on this 12-row set." No gate pass/fail decision is made here. Full English rebuild and gate run are deferred until a candidate fix is confirmed.

### Changes From Diagnostic Run

Two root causes to address, in order of confidence:

1. **Detach waveform before WeSpeaker** — break the gradient path: InfoNCE ← WeSpeaker ← generated waveform ← Kokoro ← SegmentGST. The pre-clip gradient norm reached 1300–7300; clipping to 1.0 distorts the update direction without removing the explosion source. Gradients from the contrastive loss should flow only through the speaker embedding normalization, not back through the full Kokoro graph.

2. **Swap mHuBERT layer 6 → WeSpeaker frame features** (cheapest feature swap per decision table) — the aggregation collapse is caused by query diffusion: `gst/attn_entropy_mean` stayed near log(1024) = 6.931 throughout, meaning all utterances produce near-uniform attention over the bank and thus near-identical pooled styles. mHuBERT layer 6 frame queries are insufficiently speaker-discriminative at the frame level for the MHA attention to select different bases across speakers.

### Diagnostic Objectives

This run does not pass or fail the gate. Treat it as informative when:

- `gst/pooled_style_pairwise_cos_mean` drops clearly below 0.5 and holds there without reversing.
- `gst/attn_entropy_mean` shows a meaningful decrease from log(1024) = 6.931, indicating attention is sharpening.
- Pre-clip `train/grad_norm_g` is below 10 (not 1300+).
- `train/loss_spk_contrastive` continues trending down through step 200 and does not reverse after pred branch unlocks.

### Deferred: Full English Gate Run

Do not proceed with the gate until a candidate fix is identified from this diagnostic run. When ready:

1. Run `scripts/build_phonemes.sh` for English (already done — `en_train.phonemes.jsonl` and `en_val.phonemes.jsonl` exist).
2. Run `scripts/rebuild_all_caches.sh` on the English manifests.
3. Generate deterministic en_6x1 subsets per the policy below.
4. Train with the gate config and evaluate against the success criteria below.

#### Deterministic Subset Policy

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

#### Gate Success Criteria

Treat the gate as successful only if most of the following hold:

- `train/loss_mel` drops materially.
- `train/loss_spk_contrastive` trends down from early steps.
- `val_tf/loss_mel` drops clearly.
- `val_free/loss_mel` does not stay flat or fully decoupled from `val_tf/loss_mel`.
- Validation audio table shows intelligible teacher-forced speech and speaker-dependent differences.
- No NaN/Inf events or chronic optimization instability.

For failure interpretation, see "Validation divergence" in `CLAUDE.md`.

## Required Coverage Reporting

**Deferred.** Do not generate `memorization_en_6x1*` from stale or mixed multilingual caches. Do not run the English gate until a candidate fix is identified from the diagnostic run and the full English rebuild and cache rebuild are complete.

When the rebuild is ready, record before subset generation:

- English summary:
  - total rows
  - alignment-attempted rows
  - coverage-accepted rows
  - coverage-rejected rows
  - rejection reasons by category
- Gate condition:
  - only coverage-accepted rows with `prosody_enabled=True` remain

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
- `train/gst/pooled_style_pairwise_cos_mean`
- `train/gst/attn_entropy_mean`
- `train/gst/style_dec_pairwise_cos_mean`
- `train/gst/style_pred_pairwise_cos_mean`
- `train/gst/pred_norm_mean`
- `train/gst/proj_dec_norm_mean`

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

**Status:** DISCARDED. Config updated. Diagnostic run on `_min` set scheduled as the immediate next run.

### Post-Mortem Addendum (added 2026-04-30)

**Prior loss history:** Cosine similarity to a cached target speaker embedding was the first speaker loss tried, before InfoNCE contrastive loss. It also failed to produce speaker-discriminative output. The contrastive run failed more spectacularly — gradient explosion, drone audio, pairwise_cos_mean=0.82 — but the underlying inability to produce per-speaker output predates the contrastive choice. The five patches applied to config defaults (warmup, loss weighting, temperature, grad clip, style_decoder_only_steps) fix real problems in the contrastive run, and they stand. But they do not address what went wrong in the cosine run.

**Implication:** the speaker loss function is not the only problem. The conditioning pathway — mHuBERT features → SegmentGST → 256-D ref_s → frozen Kokoro AdaIN — appears unable to transmit speaker identity into the audio output, regardless of how that path is supervised. Both loss formulations tried so far failed; the supervision signal is apparently not reaching the audio in a speaker-discriminative way.

**Three candidate collapse locations, in priority order for testing:**

1. **SegmentGST bank attention mode-collapsing** — different utterances may be producing similar `pooled_style` vectors regardless of input. If the soft attention weights assign similar mass to the same basis banks for every utterance, the pooled output is effectively speaker-agnostic before the projections ever run. This is the cheapest hypothesis to confirm: we need to log the actual attention distribution and the pairwise cosine similarity of `pooled_style` across the batch.

2. **mHuBERT layer 6 features insufficient for this aggregation and bottleneck** — layer 6 was empirically chosen as the layer with the best probeable speaker information, with layer 1 as a close second. However, "best probeable by a linear classifier on frozen features" is not the same as "transmits speaker identity through soft attention pooling and a 256-D bottleneck into a frozen decoder." The features may be speaker-discriminative in isolation but too entangled with phonetic content at layer 6 for this aggregation path to extract a clean speaker signal.

3. **Kokoro's frozen 256-D ref_s interface fundamentally too narrow** — if the frozen AdaIN conditioning space is already strongly peaked around the training distribution, even a correct speaker embedding arriving at the `ref_s` input may be unable to steer the decoder far enough to produce meaningfully different speakers. This is the hardest hypothesis to test without modifying Kokoro, so it is lowest priority unless (1) and (2) are ruled out.

**Decision:** the next run will not change losses or model architecture. It will use the existing `_min` cache with new SegmentGST internal diagnostics to localize where collapse occurs. Full English rebuild and gate decision deferred until a candidate fix is identified from this diagnostic run.

**Decision rule based on new diagnostics:**

| Observed pattern | Conclusion | Next step |
|-----------------|-----------|-----------|
| `gst/pooled_style_pairwise_cos_mean` low, `collapse/ref_s_pairwise_cos_mean` high | Collapse happens in projection layers after pooling | Investigate `to_style_dec` / `to_style_pred` capacity or regularization |
| `gst/pooled_style_pairwise_cos_mean` also high | SegmentGST aggregation itself is collapsing | Swap conditioning features — WeSpeaker frame features as cheapest test |
| `gst/attn_entropy_mean` very low, same bases selected across utterances | Bank collapse — all utterances attending to the same basis vectors | Investigate bank initialization or attention temperature |
| All SegmentGST internals look healthy, only `ref_s` collapses | Projection layers are the bottleneck | Consider whether ref_s bottleneck is tractable or Kokoro interface is the constraint |

### Run: `memorization_en_6x1_min-diag-20260501-091048` — DISCARDED

**Manifest used:** `memorization_en_6x1_min.phonemes.jsonl` (12 rows — 6 train, 6 val)
**Steps:** 101–240 (resumed from `checkpoint_100.pt`) | **Batch size:** 6 | **Warmup steps:** 200 | **style_decoder_only_steps:** 200

**Key diagnostic metrics:**

| Metric | Step 101 | Step 200 | Step 239 | Interpretation |
|--------|----------|----------|----------|----------------|
| `gst/pooled_style_pairwise_cos_mean` | 0.993 | 0.781 | 0.836 | Slowly improving during dec-only phase; reverses after pred unlock |
| `gst/attn_entropy_mean` | 6.931 | 6.923 | 6.921 | Near log(1024)=6.931 throughout — near-uniform attention, query diffusion |
| `gst/attn_top1_index_mode_count` | 3/6 | 3/6 | 3/6 | 3 of 6 batch items share the same top-1 bank basis every step |
| `gst/style_dec_pairwise_cos_mean` | 0.997 | 0.811 | 0.774 | Dec branch slowly improving throughout (genuine progress) |
| `gst/style_pred_pairwise_cos_mean` | 1.000 | 1.000 | 0.921 | 1.0 during frozen phase (expected); collapses back toward 1.0 rapidly after unlock |
| `gst/proj_dec_norm_mean` | 3.09 | 5.74 | 5.51 | Unhealthy but slowing |
| `gst/pred_norm_mean` | 0.00 | 0.00 | 6.20 | Rockets to 6.2 in 40 steps after pred unlock |
| `gst/pooled_style_norm_mean` | 8.51 | 12.91 | 17.01 | Bank growing in magnitude throughout |
| `train/grad_norm_g` (pre-clip) | ~1300 | — | ~7300 | Catastrophic explosion; same root cause as April 29 run |
| `train/loss_spk_contrastive` | 0.591 | 1.337 | 1.152 | Improving during warmup; reverses sharply at step 200 |

**Collapse localization (decision table):**

`gst/pooled_style_pairwise_cos_mean` is HIGH throughout → **Row 2: SegmentGST aggregation itself is collapsing.**

Mechanism: `gst/attn_entropy_mean` ≈ log(1024) = 6.931 throughout the run, meaning the attention is near-uniform over all 1024 bank bases. This is **query diffusion** — all utterances produce nearly identical attention distributions, so all get `pooled_style ≈ mean(bank)` regardless of speaker. This is the opposite of bank collapse (which would show very low entropy). The aggregation is failing because mHuBERT layer 6 frame queries are not sufficiently speaker-discriminative to select different bases across speakers.

Secondary factor: the pred branch immediately collapses after unlock — `gst/pred_norm_mean` reaches 6.2 in 40 steps, `gst/style_pred_pairwise_cos_mean` rebounds from 0.90 to 0.95. The pred unlock coincides with warmup end, creating a double perturbation at step 200.

**What the dec branch confirmed:** `gst/style_dec_pairwise_cos_mean` improved from 0.997 to 0.774 over 139 steps. The projection layer CAN learn given better pooled input; the bottleneck is aggregation, not projection capacity.

**Gradient explosion (unresolved):** Pre-clip norm 1300 → 7300. Clipping to 1.0 constrains step size but does not fix the explosion source. The InfoNCE ← WeSpeaker ← waveform ← Kokoro ← SegmentGST gradient path remains active.

**Audio observations:**
- Step 100: Intelligible speech, clear timing distortion (words stretching/slurring). Pred branch frozen → prosody predictor receives universal prior, not per-speaker style.
- Step 200: Voice quality breaking down, timing worse. Pred branch unlocked with exploding gradients.

**Status:** DISCARDED. Diagnostic objective met — collapse localized to aggregation (query diffusion). Config unchanged. Candidate fix (gradient detach + feature swap) scheduled as next run.

## Next Experiment Queue

1. **Gradient detach + feature swap on `memorization_en_6x1_min`** — detach waveform before WeSpeaker call and replace mHuBERT layer 6 frame features with WeSpeaker frame features as SegmentGST input. Run until collapse diagnostics show `gst/pooled_style_pairwise_cos_mean` < 0.5 or until confirming the feature swap doesn't help.
2. Full English rebuild + en_6x1 gate run — deferred until candidate fix passes `memorization_en_6x1_min` diagnostics.
3. Larger English memorization subset if the en_6x1 gate passes.

## Update Rule

When a run finishes:

1. Update `Current Experiment Results`.
2. Mark the run as promoted, discarded, or revisited.
3. If promoted, update `architecture.md`, `CLAUDE.md`, and `Current Experiment`.
4. Keep only the next 3 concrete experiments in `Next Experiment Queue`.
