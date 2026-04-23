# Architecture: Current Voice-Clone Stack

This document reflects the code currently implemented under `voice_clone/` and the launcher in `scripts/train.py`.

The system is a voice-cloning fine-tune on top of `hexgrad/Kokoro-82M` with:

- frozen `Kokoro` backbone weights,
- trainable Kokoro `L-adapters`,
- frozen `mHuBERT` for reference-audio conditioning,
- frozen `WeSpeaker` for speaker-consistency loss,
- trainable `SegmentGST` as the reference-style bottleneck,
- optional waveform GAN (`HiFi-GAN`-style MPD+MSD discriminator), which is disabled by default in `TrainConfig`.

## 1. High-level pipeline

Training data row:

- `ref_wav`: reference speaker audio, loaded and resampled to mono 16 kHz,
- `target_wav`: supervision waveform, loaded and resampled to mono 24 kHz,
- `text`,
- `lang_code`.

Training step:

1. `text` -> phonemes via `KPipeline(model=False)` -> Kokoro `input_ids`
2. `ref_wav_16k` -> frozen `mHuBERT` -> frame hidden states
3. mHuBERT frame states -> `SegmentGST` -> Kokoro `ref_s` style vector
4. `input_ids` + `ref_s` -> `KModel.forward_with_tokens(...)` -> predicted 24 kHz waveform
5. `ref_wav_16k` -> frozen `WeSpeaker` -> detached reference speaker embedding
6. predicted waveform -> differentiable mel -> frozen `WeSpeaker` -> generated speaker embedding
7. optimize generator with mel loss + speaker cosine loss + optional GAN losses

Inference path:

1. load checkpoint,
2. rebuild `Kokoro + SegmentGST + mHuBERT`,
3. text -> phonemes -> `input_ids`,
4. reference audio -> `mHuBERT` -> `SegmentGST` -> `ref_s`,
5. `Kokoro.forward_with_tokens(...)` -> 24 kHz waveform.

## 2. Conditioning encoder: frozen mHuBERT

Implemented in `voice_clone/mhubert_encoder.py`.

- Class: `MHuBERTEncoder`
- Backend: `transformers.HubertModel.from_pretrained(...)`
- Default repo: `utter-project/mHuBERT-147-base-3rd-iter`
- Input: mono waveform at 16 kHz
- Output:
  - `hidden_states`: `(B, T_frames, hidden_size)`
  - `frame_mask`: `(B, T_frames)` boolean mask
- Freezing behavior:
  - `requires_grad_(False)`
  - wrapper forces the underlying HuBERT model to stay in `eval()`

Current default extract layer in `TrainConfig` is `mhubert_extract_layer=1`.

## 3. Speaker-loss encoder: frozen WeSpeaker

Implemented in `voice_clone/wespeaker_sv.py`.

- Class: `WeSpeakerSV`
- Checkpoint layout:
  - default path `voice_clone/encoder-ckpts/wespeaker-ckpt/models/avg_model.pt`
  - sibling `config.yaml` is required in the same model directory
- Input domain: mono waveform at 16 kHz
- Output:
  - `pooled_embedding`: `(B, 256)`, L2-normalized
  - optional frame features / masks

Usage split:

- reference branch:
  - frozen forward,
  - detached embedding target for speaker loss
- generated branch:
  - waveform -> differentiable mel frontend,
  - `forward_from_mel(..., grad_through_input=True)`,
  - speaker-loss gradients flow back into the generator

WeSpeaker is used for the speaker objective only. It is not used to condition Kokoro.

## 4. Style bottleneck: SegmentGST

Implemented in `voice_clone/segment_gst.py`.

Architecture:

- learnable bank: `(num_bases=512, embed_dim=256)`
- query projection: `Linear(frame_dim -> 256)`
- attention: `MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)`
- post-attention: `LayerNorm + Dropout`
- temporal reduction: mask-weighted mean pooling over reference frames
- readout: `to_ref_s: Linear(256 -> 256)`

Important implementation detail:

- `to_ref_s` is zero-initialized
- a persistent `universal_style_vector` buffer is loaded from `TrainConfig.universal_style_vector_path`
- the forward pass computes:

`ref_s = to_ref_s(pooled_style) + 0.1 * universal_style_vector`

So at initialization the style path is effectively a scaled universal-style prior, not the full vector.

Kokoro split:

- `ref_s[:, :128]` -> decoder style branch
- `ref_s[:, 128:]` -> predictor / prosody branch
- the full 256-D `ref_s` is also used as `z_style` for all inserted adapters

Current `TrainConfig` default dropout is `gst_dropout=0.0`, so although `SegmentGST` supports dropout, the default training configuration disables it.

## 5. Generator: frozen Kokoro backbone with trainable L-adapters

Backbone:

- `kokoro.model.KModel`
- default repo: `hexgrad/Kokoro-82M`

Adapter implementation:

- file: `voice_clone/adapters.py`
- module: `ResidualAdapter`
- formula:

`h' = h + W_up(ReLU(W_down([h || z_style])))`

Initialization:

- `W_up` is zero-initialized
- adapters start as identity perturbations

Adapter placement is derived from the Kokoro config:

- duration/text encoder adapters: one per `n_layer`
- decoder adapters: one per `DECODER_L_ADAPTER_HIDDEN_DIMS`
- generator adapters: one per `generator_l_adapter_hidden_dims(...)`

Only `SegmentGST` and the adapter modules are trainable. The original Kokoro backbone weights are frozen.

## 6. Kokoro freeze policy

Implemented in `freeze_kokoro_except_adapters(...)` inside `voice_clone/train_adapters.py`.

The code does not keep Kokoro in `eval()` during training. Instead it:

- freezes all original Kokoro parameters,
- re-enables gradients only for adapter modules,
- keeps the model in `train(True)`,
- sets all `nn.Dropout.p = 0.0`,
- sets `nn.RNNBase.dropout = 0.0`.

This is an implementation workaround for ROCm/MIOpen training behavior where some recurrent backward paths require training mode.

## 7. Waveform discriminator

Implemented in `voice_clone/discriminators/hifigan.py`.

Class:

- `HiFiGANMPDMSDDiscriminator`

Composition:

- MPD periods: `[2, 3, 5, 7, 11]`
- MSD scales: `[1, 2, 4]`

Domain:

- operates directly on predicted and target waveforms in the 24 kHz output domain

Losses:

- discriminator: `discriminator_loss_lsgan`
- generator adversarial term: `generator_loss_lsgan`
- feature matching: `feature_matching_loss`

GAN training is gated by `global_step >= cfg.disc_start_step`.

## 8. Loss stack

Implemented in `voice_clone/losses.py` and `voice_clone/train_adapters.py`.

Generator objective:

`L_G = lambda_mel * L_mel + lambda_spk * L_spk + lambda_adv * L_adv + lambda_fm * L_fm`

Current `TrainConfig` defaults:

- `lambda_mel = 20.0`
- `lambda_spk = 10.0`
- `lambda_adv = 1.0`
- `lambda_fm = 2.0`

### 8.1 Mel reconstruction

- module: `MelReconstructionLoss`
- target domain: 24 kHz waveform
- transform:
  - `MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024)`
  - log-mel
- criterion:
  - L1 on log-mels
  - optional L2 exists in the module but defaults to `0.0`
- variable-length batches are handled with explicit `pred_lengths` and `target_lengths`

### 8.2 Speaker consistency

- function: `speaker_cosine_loss`
- formula: mean `1 - cosine(ref_embedding, gen_embedding)`

### 8.3 GAN terms

- discriminator: LSGAN on waveform logits
- generator: adversarial LSGAN + feature matching
- discriminator crops are random waveform segments with `segment_size = 16384`

## 9. Training loop

Implemented in `voice_clone/train_adapters.py`.

Per optimizer step:

1. zero generator and discriminator grads
2. run `cfg.grad_accum_steps` micro-steps
3. on each micro-step:
   - build masks from padded lengths
   - run frozen `mHuBERT`
   - run `SegmentGST`
   - run frozen `WeSpeaker` on reference audio
   - run Kokoro synthesis per item in the batch
   - stack and pad generated waveforms
   - run differentiable WeSpeaker path on generated audio
   - optionally run discriminator
   - backprop generator loss
4. clip grads
5. step `AdamW` optimizers
6. step schedulers if configured

Important mechanics:

- Kokoro forward is still effectively batch-1:
  - the outer batch is handled by looping `forward_with_tokens(...)` per item and re-padding outputs
- mel loss uses full predicted and target waveforms with length-aware masking
- GAN uses cropped waveform segments
- NaN/Inf guard:
  - if a micro-step produces non-finite losses, the optimizer step is skipped and `global_step` still advances

Optimizers:

- generator: `AdamW(params_g, lr=1e-4, betas=(0.8, 0.99), weight_decay=0.0)`
- discriminator: `AdamW(disc.parameters(), lr=5e-5, betas=(0.8, 0.99), weight_decay=0.0)`

Schedulers:

- optional linear warmup + cosine decay
- current `TrainConfig` defaults effectively keep LR constant because:
  - `warmup_steps = 0`
  - `lr_min_g == lr_g`
  - `lr_min_d == lr_d`

GAN activation:

- current `TrainConfig` default is `disc_start_step = 99999999`
- so the discriminator is disabled by default unless explicitly overridden

## 10. Data path

Implemented in `voice_clone/dataset.py`.

Manifest requirements per JSONL row:

- `ref_wav`
- `target_wav`
- `text`
- `lang_code`

Optional fields:

- `phonemes`
- `speaker_id`

Text/token pipeline:

1. normalize `lang_code`
2. phonemize text with `KPipeline(model=False)` unless manifest already provides `phonemes`
3. map phoneme characters into Kokoro vocab ids
4. add BOS/EOS token id `0`

Audio pipeline:

- reference audio -> mono 16 kHz
- target audio -> mono 24 kHz

Validation checks:

- files must exist
- text must be non-empty
- phonemes must be non-empty when provided
- no NaN/Inf waveform samples
- minimum reference length: `4800` samples at 16 kHz
- minimum target length: `4800` samples at 24 kHz

Batch collation:

- pads `input_ids`, `ref_wav_16k`, and `target_wav_24k`
- returns `input_ids_lengths`, `ref_lengths`, and `target_lengths`

## 11. Inference path

Implemented in `voice_clone/infer.py`.

Checkpoint loading:

- rebuild `TrainConfig` from `ckpt["train_config"]` when present
- rebuild `Kokoro + SegmentGST + mHuBERT`
- load:
  - `segment_gst`
  - duration adapters
  - decoder adapters
  - generator adapters

Inference does not use WeSpeaker or the discriminator.

Output:

- mono waveform at 24 kHz
- final waveform is clamped to `[-1, 1]`

## 12. Current default configuration

From `voice_clone/config.py`:

- `kokoro_repo_id = "hexgrad/Kokoro-82M"`
- `mhubert_repo_id = "utter-project/mHuBERT-147-base-3rd-iter"`
- `mhubert_extract_layer = 1`
- `wespeaker_checkpoint_path = "voice_clone/encoder-ckpts/wespeaker-ckpt/models/avg_model.pt"`
- `wespeaker_embedding_dim = 256`
- `wespeaker_sample_rate = 16000`
- `universal_style_vector_path = "voice_clone/universal_style_vector.pt"`
- `disable_amp_for_stft = True`
- `adapter_bottleneck = 64`
- `lambda_mel = 20.0`
- `lambda_spk = 10.0`
- `lambda_adv = 1.0`
- `lambda_fm = 2.0`
- `lr_g = 1e-4`
- `lr_d = 5e-5`
- `weight_decay_g = 0.0`
- `weight_decay_d = 0.0`
- `batch_size = 1`
- `grad_accum_steps = 1`
- `disc_start_step = 99999999`
- `warmup_steps = 0`
- `gst_dropout = 0.0`
- `grad_clip_norm_g = 5.0`
- `grad_clip_norm_d = 1.0`

## 13. Launcher behavior

`scripts/train.py` adds a few operational defaults on top of `TrainConfig`:

- default manifest: `manifests/memorize_train.jsonl`
- default val manifest: `manifests/memorize_val.jsonl` if present
- default checkpoint dir: `ckpt/memorize`
- default resume checkpoint: `ckpt/memorize/checkpoint_300.pt`
- default epochs: `300`
- default AMP flag: enabled via `--amp`
- optional CLI/env overrides for:
  - batch size
  - grad accumulation
  - warmup
  - discriminator start step
  - checkpoint cadence

That launcher changes how training is typically run, but it does not change the underlying architecture described above.
