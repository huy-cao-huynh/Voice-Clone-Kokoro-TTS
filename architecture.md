# Architecture: Voice Cloning with Kokoro TTS

This document reflects the current `voice_clone` codepath: a frozen **WeSpeaker-style speaker frontend** + trainable **SegmentGST** + trainable Kokoro **L-adapters**, with a waveform **HiFi-GAN-style MPD+MSD discriminator** trained using LSGAN and feature matching.

---

## 1. Frozen speaker frontend: WeSpeakerSV

Implemented in `voice_clone/wespeaker_sv.py`.

- **Model wrapper**: `WeSpeakerSV` (frozen encoder + differentiable mel frontend).
- **Input**: mono reference audio at **16 kHz** (`TrainConfig.wespeaker_sample_rate`).
- **Backend encoder**:
  - preferred: `_TorchvisionResNet34Encoder` (single-channel mel input),
  - fallback: `_FallbackSpeakerEncoder` if torchvision model construction fails.
- **Checkpoint source**: `TrainConfig.wespeaker_checkpoint_path` (default `wespeaker_ckpt/models/avg_model.pt`; avoids shadowing the PyPI `wespeaker` package).
- **Outputs** (`WeSpeakerSVOutput`):
  - `pooled_embedding`: `(B, 256)` L2-normalized embedding (speaker loss input),
  - `frame_features`: `(B, T_frames, C)` frame sequence for SegmentGST conditioning.

During training:

- **Reference path**: WeSpeaker forward with `grad_through_input=False` to extract conditioning and detached speaker target.
- **Generated path**: differentiable mel + `forward_from_mel(..., grad_through_input=True)` so speaker-loss gradients flow back to the generator.

---

## 2. Trainable bottleneck: SegmentGST

Implemented in `voice_clone/segment_gst.py`.

- **Bank**: learnable `bank` parameter `(num_bases=512, embed_dim=256)`.
- **Query projection**: `q_proj` maps frame features into `embed_dim` (current build expects `frame_dim=1024`).
- **Attention**: `nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)`.
- **Post-attention**: `LayerNorm` + `Dropout(0.1)`.
- **Pooling**: mask-weighted mean over time.
- **Readout**: `to_ref_s: Linear(256 -> 256)` with zero-initialized weight/bias.
- **Kokoro split**:
  - `ref_s[:, :128]` -> decoder AdaIN style branch,
  - `ref_s[:, 128:]` -> prosody branch,
  - full `ref_s` (256-D) -> `z_style` for all L-adapters.

Universal style handling:

- `SegmentGST` supports a persistent `universal_style_vector` buffer added to `delta_ref_s`.
- If not provided at construction time, it defaults to zeros.
- `TrainConfig.universal_style_vector_path` exists, but the current `build_models` path does not yet load/inject this file into `SegmentGST`.

---

## 3. Frozen backbone + trainable injections: Kokoro + L-adapters

- **Backbone**: `KModel` from `hexgrad/Kokoro-82M` (default `TrainConfig.kokoro_repo_id`).
- **Trainable parameters**: SegmentGST + adapter ModuleLists only.
- **Frozen parameters**: all original Kokoro backbone weights (`requires_grad_(False)`).

Adapter definition (`voice_clone/adapters.py`):

`h' = h + W_up(ReLU(W_down([h || z_style])))`

- `h`: `(B, C, T)` feature map at each insertion point.
- `z_style`: `(B, 256)` broadcast over time.
- `W_up` is zero-initialized so each adapter starts as an identity perturbation.

Adapter placement comes from Kokoro config:

- duration encoder adapters: `n_layer` blocks,
- decoder adapters: `DECODER_L_ADAPTER_HIDDEN_DIMS`,
- generator adapters: `generator_l_adapter_hidden_dims(upsample_initial_channel, num_upsamples)`.

For Kokoro-82M defaults, this is still 3 (duration) + 5 (decoder) + 2 (generator) = 10 adapters.

---

## 4. Waveform discriminator

Implemented in `voice_clone/discriminators/hifigan.py`:

- `HiFiGANMPDMSDDiscriminator = MPD + MSD`
- MPD periods: `[2, 3, 5, 7, 11]`
- MSD scales: `[1, 2, 4]`
- Operates on waveform outputs directly (24 kHz domain).

Used with:

- `discriminator_loss_lsgan`
- `generator_loss_lsgan`
- `feature_matching_loss`

from `voice_clone/losses.py`.

---

## 5. Loss stack

Generator objective:

`L_G = lambda_mel * L_mel + lambda_spk * L_spk + lambda_adv * L_adv + lambda_fm * L_fm`

Defaults from `LossWeights`:

- `lambda_mel=1.0`
- `lambda_spk=1.0`
- `lambda_adv=0.1`
- `lambda_fm=1.0`

### 5.1 Mel reconstruction

- `MelReconstructionLoss` on 24 kHz predicted/target waveforms.
- Mel settings from `MelLossConfig` (defaults: `n_fft=1024`, `hop=256`, `win=1024`).
- Loss is log-mel L1 plus optional L2 (default L2 weight 0.0).

### 5.2 Speaker consistency

- `speaker_cosine_loss(ref_embedding, gen_embedding)`.
- Uses normalized pooled embeddings from WeSpeakerSV.
- Generated embedding path is differentiable to waveform.

### 5.3 GAN terms

- Discriminator: LSGAN real/fake on waveform logits.
- Generator: adversarial LSGAN + feature matching against detached real features.

---

## 6. Training loop architecture

Implemented in `voice_clone/train_adapters.py`.

High-level order per effective step:

1. Zero generator/discriminator grads.
2. Run `grad_accum_steps` micro-steps:
  - WeSpeaker ref forward -> SegmentGST -> `ref_s`
  - Kokoro `forward_with_tokens(input_ids, ref_s, speed)`
  - Clamp audio to `[-1, 1]`
  - WeSpeaker gen forward (differentiable mel path)
  - Optional D forward/backward when `global_step >= disc_start_step`
  - G backward (`mel + spk + optional adv/fm`)
3. Step optimizers (D first if active, then G), scaler updates if AMP enabled.
4. Increment `global_step`, step schedulers.

Additional mechanics:

- **Gradient accumulation**: default `grad_accum_steps=8` (batch size is currently fixed to 1 in collate).
- **GAN warmup**: discriminator starts at `disc_start_step` (default **250**).
- **Schedulers**: warmup then cosine (`SequentialLR(LinearLR -> CosineAnnealingLR)`).
- **NaN guard**: if non-finite generator loss appears in a micro-step, skip optimizer step and advance safely.

Kokoro freeze mode policy:

- `freeze_kokoro_except_adapters` sets model to eval and freezes params.
- `apply_kokoro_freeze_mode_policy` currently supports `eval_then_fallback_train_dropout_zero`.

---

## 7. Data and tokenization path

Implemented in `voice_clone/dataset.py`.

- Manifest rows require: `ref_wav`, `target_wav`, `text`, `lang_code`.
- Audio loading:
  - reference -> mono 16 kHz,
  - target -> mono 24 kHz.
- Validation checks:
  - finite waveform samples only,
  - min reference length: 4800 samples @16k,
  - min target length: 4800 samples @24k.
- Text -> phonemes via `KPipeline(model=False)`; then phonemes -> Kokoro `input_ids`.
- Current collate function enforces `batch_size == 1`.

---

## 8. Inference architecture

Implemented in `voice_clone/infer.py`.

1. Load checkpoint and reconstruct `TrainConfig` when available.
2. Build stack with `build_models`: `KModel + SegmentGST + WeSpeakerSV`.
3. Load adapter + SegmentGST weights from checkpoint.
4. Convert input text to phonemes and `input_ids`.
5. Load reference waveform (16 kHz), run WeSpeaker -> SegmentGST -> `ref_s`.
6. Run Kokoro synthesis with `forward_with_tokens(input_ids, ref_s, speed)`.
7. Clamp output to `[-1, 1]`, save 24 kHz mono waveform.

---

## 9. Current defaults summary

From `TrainConfig` in `voice_clone/config.py`:

- `kokoro_repo_id`: `hexgrad/Kokoro-82M`
- `wespeaker_checkpoint_path`: `wespeaker_ckpt/models/avg_model.pt`
- `wespeaker_embedding_dim`: `256`
- `wespeaker_sample_rate`: `16000`
- `adapter_bottleneck`: `64`
- `lr_g`: `1e-4`
- `lr_d`: `5e-5`
- `grad_accum_steps`: `8`
- `disc_start_step`: `250`
- `warmup_steps`: `50`
- `lr_min_g`: `1e-6`
- `lr_min_d`: `1e-7`
- `checkpoint_interval`: `250`
- `speed`: `1.0`
- `disable_amp_for_stft`: `True`

AMP note:

- `TrainConfig.use_amp` dataclass default is `True`.
- CLI training entrypoint currently constructs config with `use_amp=args.amp`, so without `--amp` it runs with AMP disabled.