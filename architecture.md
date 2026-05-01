# Architecture: Current Voice-Clone Stack

This document reflects the code currently implemented under `voice_clone/`, `kokoro/`, and `scripts/train.py`.

The current system is a SegmentGST-only voice-cloning stack on top of frozen `hexgrad/Kokoro-82M`:

- Kokoro backbone weights stay frozen.
- `SegmentGST` is the only trainable generator-side module.
- Reference conditioning comes from frozen `mHuBERT`.
- Speaker supervision comes from frozen `WeSpeaker`.
- Duration and F0 targets are cached, but only mel and speaker losses are active by default.
- Training uses MFA-backed per-token teacher forcing when available and total-length forcing as the fallback; inference stays free-running.

## 1. High-level pipeline

Training row:

- `ref_wav`: reference speaker audio, loaded at 16 kHz mono
- `target_wav`: supervision waveform, loaded at 24 kHz mono
- `text`
- `lang_code`

Training step:

1. `text` -> phonemes -> Kokoro `input_ids`
2. `ref_wav_16k` -> frozen `mHuBERT` -> frame hidden states
3. mHuBERT frame states -> `SegmentGST` -> 256-D Kokoro style vector `ref_s`
4. `input_ids + ref_s` -> `KModel.forward_with_tokens(...)`
5. During training, Kokoro duration expansion uses cached `gt_dur_frames` when `prosody_enabled=True`; otherwise it falls back to forcing only the total generated length
6. generated 24 kHz waveform vs target 24 kHz waveform -> log-mel reconstruction loss
7. generated 24 kHz waveform -> frozen `WeSpeaker` -> contrastive speaker loss against cached target embeddings

Inference path:

1. rebuild `Kokoro + SegmentGST + mHuBERT`
2. text -> phonemes -> `input_ids`
3. reference audio -> `mHuBERT` -> `SegmentGST` -> `ref_s`
4. `KModel.forward_with_tokens(...)` runs free-running durations and returns 24 kHz waveform

## 2. Conditioning and Style Path

### 2.1 Frozen mHuBERT

Implemented in `voice_clone/mhubert_encoder.py`.

- Model: `MHuBERTEncoder`
- Input: mono 16 kHz waveform
- Output:
  - `hidden_states`: `(B, T_frames, hidden_size)`
  - `frame_mask`: `(B, T_frames)`
- The wrapper keeps the underlying HuBERT model frozen and in eval mode.
- The current training stack reads cached mHuBERT states from disk during training rather than recomputing them online.

### 2.2 SegmentGST

Implemented in `voice_clone/segment_gst.py`.

- Input: mHuBERT frame states plus frame mask
- Output: Kokoro conditioning vector `ref_s` with shape `(B, 256)`
- `ref_s[:, :128]` conditions the decoder branch
- `ref_s[:, 128:]` conditions Kokoro’s prosody predictor branch

Important implementation details:

- `to_style_dec` and `to_style_pred` are small-normal initialized (`std=0.01`) with zero bias
- a persistent `universal_style_vector` buffer is loaded from disk
- forward computes a residual style offset on top of that universal prior
- train and validation logging now report projection-vs-prior norms so early overshoot is visible
- all generator gradients flow through this one `ref_s` bottleneck, because Kokoro itself is frozen

Operational implication:

- `SegmentGST` is currently responsible for all trainable adaptation capacity in the generator path
- if the stack cannot memorize a tiny set cleanly under the current Phase-1 loss regime, that is evidence of a capacity or conditioning problem, not a prosody-supervision problem

## 3. Frozen Kokoro Generator

Implemented in `kokoro/kokoro/model.py` and built from `voice_clone/train_adapters.py`.

- `KModel` is loaded from `hexgrad/Kokoro-82M`
- `freeze_kokoro_backbone(...)` sets `requires_grad_(False)` for all Kokoro parameters
- Kokoro still runs in `train(True)` during training, but no backbone parameters receive gradients
- batched training is still executed as a per-item loop over `forward_with_tokens(...)`, with outputs padded back into batch tensors afterward

Current duration behavior:

- Free-running path:
  - Kokoro predicts per-token durations from its frozen duration head
  - rounded durations are used to expand text states into the decoder/F0 time axis
- Training-time forced path:
  - Kokoro still predicts duration logits
  - if cached MFA supervision is present, `gt_dur_frames` directly replace the rounded durations
  - otherwise rounded durations are rescaled so their total frame count matches the target waveform duration
  - inference and `val_free` remain fully free-running

Duration-frame unit:

- one Kokoro duration frame corresponds to `1 / 40` seconds
- at 24 kHz, that is `600` waveform samples per duration frame

## 4. Loss Stack

Implemented in `voice_clone/losses.py` and `voice_clone/train_adapters.py`.

Generator objective:

`L_G = lambda_mel * L_mel + lambda_spk_contrastive * L_spk + lambda_dur * L_dur + lambda_f0 * L_f0`

Current defaults in `voice_clone/config.py`:

- `lambda_mel = 20.0`
- `lambda_spk_contrastive = 1.0`
- `lambda_dur = 0.0`
- `lambda_f0 = 0.0`
- adversarial training remains off by default because `disc_start_step` is effectively set out of reach

### 4.1 Active losses

`L_mel`

- log-mel L1 reconstruction between predicted and target 24 kHz waveforms
- training path benefits from teacher-forced duration alignment

`L_spk`

- contrastive speaker loss between generated-speaker embeddings and cached target embeddings
- target embeddings come from the target waveform and are cached offline
- generated waveforms are resampled inside `WeSpeakerSV` to the encoder’s 16 kHz domain before speaker embedding extraction

### 4.2 Logged but inactive losses

`L_dur`

- log-domain loss against cached duration targets
- currently inactive because `lambda_dur = 0`

`L_f0`

- masked L1 against cached F0 targets
- currently inactive because `lambda_f0 = 0`

Practical reading of the loss stack:

- `L_mel` is the main fitting objective
- `L_spk` is the main identity regularizer
- `L_dur` is available for later prosody work but intentionally off in Phase 1
- `L_f0` remains diagnostic-only because its cached target path is still placeholder quality

## 5. Offline Caches

### 5.1 Feature cache

Implemented in `voice_clone/cache_builder.py`.

Per row it stores:

- reference mHuBERT hidden states and mask
- target WeSpeaker embedding
- cached duration targets and mask
- `gt_dur_frames` and `prosody_enabled`
- cached F0 targets and mask
- cache-schema metadata
- manifest fingerprint metadata

`VoiceCloneManifestDataset` requires this cache to exist and validates freshness against the manifest fingerprint.

Each training sample loads:

- `ref_hidden_states`
- `ref_frame_mask`
- `target_wespeaker_embedding`
- `duration_targets`
- `duration_mask`
- `gt_dur_frames`
- `prosody_enabled`
- `f0_targets`
- `f0_mask`
- raw `ref_wav_16k` and `target_wav_24k`

Old-schema cache rows are expected to fail fast and must be rebuilt rather than loaded through compatibility shims.

### 5.2 Prosody cache

Implemented in `voice_clone/prosody_targets_builder.py`.

Current behavior:

- if MFA alignments are available, duration targets are derived from cached TextGrid intervals and projected onto the Kokoro token sequence
- unsupported or failed rows fall back to heuristic uniform durations with `prosody_enabled=False`
- F0 is still extracted onto a placeholder target grid and remains a compatibility field rather than trusted supervision

This cache now supports Phase-1 teacher forcing for duration alignment, but it does not yet provide reliable F0 supervision.

## 6. Training, Validation, and Inference Semantics

Training loop behavior:

- generator parameters are only the `SegmentGST` parameters
- Kokoro, mHuBERT, and WeSpeaker remain frozen
- training computes mel, speaker, duration, and F0 scalar metrics every step, but only mel and speaker contribute gradient by default
- gradient clipping is applied to generator parameters with `grad_clip_norm_g = 5.0`

Validation reports two views:

- `val_free/*`: Kokoro runs free-running durations, matching inference semantics
- `val_tf/*`: Kokoro uses teacher-forced durations, preferring MFA `gt_dur_frames` and falling back to forced total length

Recommended interpretation:

- `val_tf/*` tells you whether the acoustic/style pathway can match the target once duration drift is removed
- `val_free/*` tells you whether the actual inference path is improving end to end
- if `val_tf/loss_mel` improves but `val_free/loss_mel` stays poor, the remaining problem is mostly duration control rather than acoustic rendering

Inference always uses the free-running path. There is no teacher forcing at inference time.
