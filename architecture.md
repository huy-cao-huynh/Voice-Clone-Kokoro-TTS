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
- **Checkpoint source**: `TrainConfig.wespeaker_checkpoint_path` (default `wespeaker/models/avg_model`).
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
- `wespeaker_checkpoint_path`: `wespeaker/models/avg_model`
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
# Architecture: Voice Cloning with Kokoro TTS

This document matches the current `voice_clone` training and inference code: frozen [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (intermediate-layer speaker conditioning) plus trainable **SegmentGST** and Kokoro **L-adapters**, with a waveform MPD+MSD discriminator (HiFi-GAN style) trained with LSGAN and feature matching.

---

## 1. Frozen speaker front-end: XLS-R (Intermediate-layer SV frames)

- **Checkpoint**: `facebook/wav2vec2-xls-r-300m` (`TrainConfig.xlsr_model_id`, `XLSRSV`), tapping encoder `layer_idx` (`TrainConfig.xlsr_layer_idx`, default **12**).
- **Wrapper**: Hugging Face `Wav2Vec2Model` + `Wav2Vec2FeatureExtractor` (frozen weights; tensor preprocessing keeps waveforms on-device).
- **Reference audio**: mono, **16 kHz** (hard requirement of the XLS-R wrapper).
- **Outputs** (see `XLSRSVOutput` in `voice_clone/xlsr_sv.py`):
  - **Pooled speaker embedding** `pooled_embedding`: shape **(B, 1024)**, L2-normalized for cosine speaker loss.
  - **Frame hidden states**: **(B, T_frames, 1024)** — used as SegmentGST queries.
  - **Frame mask** **(B, T_frames)** derived from the feature extractor attention mask and the model’s feature extractor length/stride.

There is **no ECAPA-TDNN** in this repo; speaker identity for training losses comes from the same frozen **XLS-R pooled embedding** (1024-D).

For **training and inference on device**, `XLSRSV` (`voice_clone/xlsr_sv.py`) keeps XLS-R weights frozen and supports switching gradient flow with `grad_through_input` (used for generated-audio speaker loss and SSL grad clipping).

---

## 2. Trainable bottleneck: SegmentGST

Implemented in `voice_clone/segment_gst.py`.

- **Learnable bank** `B`: **(512, 256)** — `num_bases=512`, `embed_dim=256`, initialized `N(0, 0.02)`.
- **Queries**: XLS-R frames projected with `nn.Linear(1024 → 256)`.
- **Attention**: `nn.MultiheadAttention` with **8 heads**, `batch_first=True`, **keys/values = bank** (expanded per batch). Frame validity is **not** passed as `key_padding_mask`; invalid frames are zeroed out **after** MHA via masked mean pooling.
- **Post-MHA**: `LayerNorm(256)`, `Dropout(0.1)`.
- **Pooling**: mask-weighted mean over time → **(B, 256)** `pooled_style`.
- **Readout**: `Linear(256 → 256)` → `**ref_s`** **(B, 256)**, **zero-initialized** (weight and bias) so `ref_s` starts as a zero vector (neutral AdaIN conditioning) rather than random noise.
- **Kokoro split** (matches `KModel.forward_with_tokens`):
  - `ref_s[:, :128]` → **decoder AdaIN** style `s` (timbre / decoder conditioning).
  - `ref_s[:, 128:]` → **prosody** path as `s` into duration/F0/N blocks (`style_dim=128` in Kokoro `config.json`).
  - Full `**ref_s`** is `**z_style**` for all **L-adapters** (256-D concatenation in adapter MLPs).

---

## 3. Frozen backbone: Kokoro-82M + L-adapter injection sites

- **Default repo**: `hexgrad/Kokoro-82M` (`TrainConfig.kokoro_repo_id`).
- **Frozen**: full `KModel` except the injected adapter parameters (`freeze_kokoro_except_adapters` in `voice_clone/train_adapters.py`).

Numeric backbone settings come from Kokoro’s published `config.json` (current HF snapshot):


| Setting                                        | Value                           |
| ---------------------------------------------- | ------------------------------- |
| PLBERT hidden size                             | 768                             |
| `hidden_dim` (`d_hid`, duration/encoder width) | 512                             |
| `n_layer` (duration encoder depth)             | 3                               |
| Kokoro `style_dim` (AdaIN / prosody `s`)       | 128                             |
| `n_mels`                                       | 80                              |
| ISTFTNet `upsample_initial_channel`            | 512                             |
| ISTFTNet `upsample_rates`                      | [10, 6] → **2** upsample stages |


**Forward sketch** (see `kokoro/kokoro/model.py`): phoneme `input_ids` → PLBERT → linear to **512**-D → prosody/duration → alignment → decoder → **24 kHz** waveform.

### 3.1 Residual L-adapter definition

In `voice_clone/adapters.py`, each adapter implements:


h' = h + W_{\text{up}}\bigl(\mathrm{ReLU}(W_{\text{down}}([h  z_{\text{style}}]))\bigr)


with `h` **(B, C, T)** and `z_style` **(B, 256)** broadcast along **T**. Per adapter:

- `W_down`: **Linear(C + 256 → 64)** (`adapter_bottleneck=64` from `TrainConfig`).
- `W_up`: **Linear(64 → C)**, **zero-initialized** (weight and bias) so the adapter starts as identity and the pretrained Kokoro behavior is preserved at step 0.

### 3.2 Where adapters attach (block-level)


| Subsystem                                                              | # adapters                      | Channel width **C** at each site                                    | When it runs                                                                                                                                         |
| ---------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Duration encoder** (`DurationEncoder` in `kokoro/kokoro/modules.py`) | **3** (= `n_layer`)             | **512** each                                                        | After each **AdaLayerNorm** block (after every 1-layer BiLSTM stage), using full `**z_style` = `ref_s**`.                                            |
| **ISTFTNet decoder** (`Decoder` in `kokoro/kokoro/istftnet.py`)        | **5**                           | **(1024, 1024, 1024, 1024, 512)** = `DECODER_L_ADAPTER_HIDDEN_DIMS` | Index **0**: after **encode** (`AdainResBlk1d` to 1024 ch). Indices **1–4**: after each **decode** `AdainResBlk1d` (last block upsamples to 512 ch). |
| **Generator** (ISTFTNet `Generator`)                                   | **2** (= `len(upsample_rates)`) | **(256, 128)** from `generator_l_adapter_hidden_dims(512, 2)`       | After each upsample/resblock segment, in order.                                                                                                      |


**Total L-adapters**: 3 + 5 + 2 = **10**, all conditioned on `**ref_s` (256-D)**.

---

## 4. Tensor shapes through the conditioning path

Rough data flow for one training step (batch size **B**):

1. **Reference waveform** `ref_wav_16k`: **(B, N_16k)** → frozen XLS-R → frame hidden states **(B, T_w, 1024)**, mask **(B, T_w)**.
2. **SegmentGST** → `ref_s` **(B, 256)**; internally MHA output **(B, T_w, 256)** → pooled **(B, 256)**.
3. **Kokoro** `forward_with_tokens(input_ids, ref_s)`:
  - `input_ids`: **(B, L)** (phoneme token ids including boundaries).
  - PLBERT sequence: **(B, L, 768)** → `bert_encoder` → **(B, 512, L)** (duration input layout as in `KModel`).
  - Duration path features stay at hidden width **512** at adapter sites; decoder feature maps use the channel counts in §3.2.
4. **Synthesis output** `pred_wav`: **(B, T_24k)** mono at **24 kHz**, hard-clamped to **[-1, 1]** via `.clamp(-1.0, 1.0)` in both training and inference to prevent out-of-range samples without gradient saturation (unlike `tanh`).

During **training**, that Kokoro forward runs **once per micro-step** under the generator graph (XLS-R ref -> SegmentGST -> `forward_with_tokens`). After synthesis, `pred_wav` is resampled **24 → 16 kHz** and run through frozen XLS-R **once with grad** to produce `wv_gen` (intermediate-layer pooled embedding). `wv_gen` drives the cosine speaker loss and provides gradients for the waveform GAN losses. There is **no XLS-R forward** on the target waveform; the waveform discriminator operates on 24 kHz waveforms only (`disc(tgt_24)` and `disc(pred_wav.detach())` for the D step). This yields **2 total XLS-R forwards per micro-step** (ref, gen). See §6 for the full step order.

---

## 5. Losses (training)

Composite generator objective:


\mathcal{L}*G = \lambda*{\text{mel}}\mathcal{L}*{\text{mel}} + \lambda*{\text{spk}}\mathcal{L}*{\text{spk}} + \lambda*{\text{adv}}\mathcal{L}_{\text{adv},G} + \lambda*{\text{fm}}\mathcal{L}_{\text{fm},G}


with defaults `LossWeights`: **λ_mel = 1.0**, **λ_spk = 1.0**, **λ_adv = 0.1**, **λ_fm = 1.0** (`voice_clone/config.py`). `λ_spk` is set equal to `λ_mel` to prevent mel gradients from overwhelming the speaker identity signal during training.

### 5.1 Mel reconstruction (`MelReconstructionLoss`)

- **Signal**: Kokoro output vs target waveform, both **24 kHz**, cropped to matching length.
- **Frontend**: `torchaudio.transforms.MelSpectrogram` with `MelLossConfig`: **n_fft = 1024**, **hop_length = 256**, **win_length = 1024**, **n_mels** from Kokoro config (**80**), `f_min = 0`, `f_max = None` (Nyquist), `center=True`, power **1.0**.
- **Loss**: **mean L1** on **log** mels (with `log_floor = 1e-5`). **l2_weight = 0** (L2 term not used unless changed in code).

### 5.2 Speaker consistency (`speaker_cosine_loss`)

- Embeddings: frozen XLS-R **pooled embeddings** **(B, 1024)** (L2-normalized).
- **Reference**: `ref_wav_16k` through XLS-R with **no grad** through audio.
- **Generated**: resampled predicted waveform, with `**grad_through_input=True**` so gradients reach the generator.
- **Loss**: mean over batch of **1 − cos(a, b)** with **L2 re-normalization** of both vectors (cosine in 1024-D).

### 5.3 Waveform GAN (`HiFiGANMPDMSDDiscriminator`) (LSGAN + feature matching)

- **Discriminator**: `HiFiGANMPDMSDDiscriminator` (`voice_clone/discriminators/hifigan.py`) operating directly on **24 kHz waveforms**.
- **MPD**: periods **[2, 3, 5, 7, 11]**.
- **MSD**: scales **[1, 2, 4]** (raw audio plus average-pooled downsampled variants).
- **Adversarial loss**: **LSGAN (MSE)** (`discriminator_loss_lsgan` + `generator_loss_lsgan`).
- **Feature matching**: generator matches intermediate discriminator feature maps (`feature_matching_loss`).
- **D step**: `disc(tgt_24)` vs `disc(pred_wav.detach())`, discriminator optimizer only.
- **G step**: `disc(pred_wav)` with real features detached, adding `λ_adv * L_adv + λ_fm * L_fm` to the generator loss.

### 5.4 Numerical safeguard: ISTFTNet exp clamping

The `Generator` in `kokoro/kokoro/istftnet.py` converts log-magnitude to linear magnitude via `torch.exp()` before the inverse STFT. To prevent overflow (especially under AMP fp16 where `exp(x)` overflows for x > ~11), the exponent input is clamped: `torch.exp(x.clamp(max=15.0))`. `exp(15) ≈ 3.27 M` is well above any physically meaningful STFT magnitude for 24 kHz audio. The conservative cap also limits backward gradient amplification through the exp (compared to a naïve `max=30`, the maximum gradient is reduced by a factor of ~3.3 × 10⁹), which matters during early training when adapter perturbations or random style vectors may produce extreme `conv_post` outputs.

---

## 6. Optimization and training loop

- **Frozen Kokoro in train() mode**: `freeze_kokoro_except_adapters` calls `kmodel.train()` (not `eval()`) to keep MIOpen RNN backward working on ROCm. Freezing is controlled by `requires_grad_(False)`, not by `eval()`. This means `nn.Dropout` layers in frozen Kokoro blocks remain active during training, adding stochastic regularization. The trainable L-adapters (`ResidualAdapter`) contain no dropout or batchnorm.
- **Generator trainable parameters**: SegmentGST + all Kokoro L-adapters (duration, decoder, generator).
- **Discriminator trainable parameters**: `HiFiGANMPDMSDDiscriminator` only.
- **Optimizers**: **AdamW** for both; **lr_g = 1e-4**, **lr_d = 5e-5** (TTUR — Two Time-scale Update Rule: the discriminator learns at half the generator rate to prevent D from overpowering G early in training); **weight_decay_g = weight_decay_d = 0** (defaults).
- **Gradient clipping**: **1.0** L2 norm on generator params and on discriminator params when not `None` (`grad_clip_g`, `grad_clip_d`).
- **AMP**: optional CUDA autocast + `GradScaler` when `TrainConfig.use_amp=True` (default **False**). **Separate** `GradScaler` instances are used for generator and discriminator (`scaler_g`, `scaler_d`) so that inf/NaN in one optimizer does not force the other to drop its scale factor. Each scaler calls `update()` after its own optimizer step. Scaler states are saved and restored in checkpoints.
- **Learning rate schedule**: `SequentialLR(LinearLR → CosineAnnealingLR)` ramps `opt_g` from `1e-3 × lr_g` to full `lr_g` over `warmup_steps` (default **50**), then cosine-decays to `lr_min_g` (default **1e-6**) over the remaining steps to `max_steps`. `opt_d` gets its own schedule starting from `disc_start_step`, decaying to `lr_min_d` (default **1e-7**). When `max_steps` is not provided, the cosine horizon is estimated from `len(dataloader) * epochs / grad_accum_steps`. When resuming from a checkpoint, schedulers are fast-forwarded to match the resumed step.
- **Weight-norm stripping**: `strip_weight_norm_frozen` bakes `weight_norm` decompositions in frozen Kokoro `ConvTranspose1d` layers into static parameters at model-build time, avoiding AMP autocast cache misses on ROCm. Safe because the Kokoro model is frozen (`requires_grad=False`).

### 6.1 Gradient accumulation

Batch size stays **1** in the DataLoader (Kokoro's `forward_with_tokens` constraint), but effective batch size is increased via **gradient accumulation** over `grad_accum_steps` micro-steps (default **8**). Gradients from each micro-step are scaled by `1/grad_accum_steps` before `.backward()`, then accumulated into a single optimizer step. `global_step` increments once per effective step (not per micro-step), and LR schedulers step once per effective step.

### 6.2 Phased training (generator-only warmup)

For the first `disc_start_step` effective steps (default **500**), the discriminator is disabled:

- Waveform GAN forward/backward is skipped: `generator_loss_backward` is called with `real_disc_features=None`, so `loss_adv_g = 0` and `loss_fm = 0`.
- `opt_d.step()` and `sched_d.step()` are skipped; `loss_d` is logged as 0.

Once `global_step >= disc_start_step`, the discriminator activates and the full adversarial objective is used. The discriminator's LR warmup begins at this point.

### 6.3 Per-step order

The training loop (`train_loop` in `voice_clone/train_adapters.py`) uses a two-level structure — one outer effective step wrapping N micro-steps:

1. `**opt_g.zero_grad`** and **`opt_d.zero_grad`** once per effective step.
2. **For each micro-step** (N = `grad_accum_steps`):
  1. **XLS-R ref** (`xlsr_ref`): reference **16 kHz** → frozen XLS-R (no grad) → frame hidden states + pooled embedding; SegmentGST → `ref_s`.
  2. **Kokoro forward** (`kokoro_fwd`): `KModel.forward_with_tokens(input_ids, ref_s)` → `pred_wav` (**24 kHz**), under autocast when AMP is on.
  3. **XLS-R gen** (`xlsr_gen`): resample `pred_wav` **24 → 16 kHz** and run XLS-R **with grad** (`grad_through_input=True`) → `wv_gen` (`XLSRSVOutput`). Gradients flow into the generator. A **per-tensor gradient clamp hook** (max L2 norm `ssl_grad_max_norm`, default **1.0**) is registered on `wv_gen.frame_hidden_states` and `wv_gen.pooled_embedding`.
  4. **Waveform discriminator** (`disc`): if `global_step >= disc_start_step`, compute `real_logits, real_feats = disc(tgt_24)` and `fake_logits_detached, _ = disc(pred_wav.detach())` (D step uses detached fake).
  5. **D backward**: compute `discriminator_loss_lsgan(real_logits, fake_logits_detached)` and call `.backward()` scaled by `1/N`.
  6. **G backward** (`gen_backward`): `generator_loss_backward` computes mel + speaker, and (if D is active) adds adversarial + feature matching losses, scaled by `1/N`, and calls `.backward()`.
3. **Deferred optimizer steps** (after all micro-steps): unscale (AMP), clip gradients, `opt_d.step()` (if D is active), `opt_g.step()`, `scaler_d.update()` + `scaler_g.update()` (AMP).
4. **`global_step += 1`**.
5. **LR scheduler step**: `sched_g.step()` every step; `sched_d.step()` only when `global_step > disc_start_step`.

This gives **2 XLS-R forward passes per micro-step** (ref, gen). The waveform discriminator forward is run only when `global_step >= disc_start_step`.

**NaN guard**: after each micro-step's generator losses, the loop checks all loss values with `np.isfinite`. If any component is NaN or Inf, the `nan_detected` flag is set, the remaining micro-steps are skipped, both optimizer gradients are zeroed, and `global_step` advances without an optimizer step. A `train/nan_skips` counter is logged to wandb when active. This prevents a single corrupted sample from poisoning the accumulated gradient.

**Dataset-level audio validation** (`voice_clone/dataset.py`): `load_audio_mono` verifies `torch.isfinite(wav).all()` after resampling; non-finite samples raise `ValueError`. `VoiceCloneManifestDataset.__getitem__` enforces minimum durations: reference audio ≥ 4800 samples at 16 kHz (~~0.3 s, enough XLS-R frames for a meaningful speaker embedding) and target audio ≥ 4800 samples at 24 kHz (~~0.2 s).

- **Waveform GAN schedule**: discriminator activation is controlled by `disc_start_step` (G-only warmup before that).
- **Kokoro inference timing**: `speed` default **1.0** (`TrainConfig.speed`), passed into `forward_with_tokens`.

Logging defaults: `**log_interval = 1`**, `**checkpoint_interval = 500**`. Metrics logged are averages over the N micro-steps in each effective step.

**Validation**: when `--val-manifest` is provided, validation metrics (`val/mel_l1`, `val/mel_l2`, `val/speaker_loss`) and audio samples are computed from the held-out set instead of the training set. This enables overfitting detection.

---

## 7. Hyperparameter summary (chosen defaults)


| Group        | Parameter                                        | Value                                      |
| ------------ | ------------------------------------------------ | ------------------------------------------ |
| Models       | Kokoro repo                                      | `hexgrad/Kokoro-82M`                       |
|              | XLS-R repo                                       | `facebook/wav2vec2-xls-r-300m`             |
| Adapters     | `adapter_bottleneck`                             | **64**                                     |
| SegmentGST   | `num_bases`                                      | **512**                                    |
|              | `embed_dim`                                      | **256**                                    |
|              | `frame_dim`                                      | **1024**                                   |
|              | `num_heads`                                      | **8**                                      |
|              | `ref_dim` / `style_dec_dim`                      | **256** / **128**                          |
|              | `dropout`                                        | **0.1**                                    |
| Loss weights | `lambda_mel` / `lambda_spk` / `lambda_adv` / `lambda_fm` | **1.0** / **1.0** / **0.1** / **1.0** |
| Mel loss     | `sample_rate`                                    | **24000**                                  |
|              | `n_fft` / `hop_length` / `win_length`            | **1024** / **256** / **1024**              |
| Waveform D (HiFi-GAN) | MPD periods | **[2, 3, 5, 7, 11]** |
|              | MSD scales                                      | **[1, 2, 4]** |
| Optim        | `lr_g`                                           | **1e-4**                                   |
|              | `lr_d`                                           | **5e-5** (TTUR)                            |
|              | `weight_decay_g`, `weight_decay_d`               | **0**                                      |
|              | `grad_clip_g`, `grad_clip_d`                     | **1.0**                                    |
|              | `ssl_grad_max_norm`                             | **1.0** (per-tensor hook on XLS-R outputs) |
|              | `grad_accum_steps`                               | **8**                                      |
| Schedule     | `warmup_steps`                                   | **50**                                     |
|              | `lr_min_g`                                       | **1e-6** (cosine floor for G)              |
|              | `lr_min_d`                                       | **1e-7** (cosine floor for D)              |
|              | `disc_start_step`                                | **500** (G-only warmup)                    |
|              | `xlsr_layer_idx`                                | **12**                                     |
|              | `speed`                                          | **1.0**                                    |
| Run          | `use_amp`                                        | **False**                                  |
|              | `log_interval`                                   | **1**                                      |
|              | `checkpoint_interval`                            | **1000**                                   |


CLI overrides (e.g. `--kokoro-repo`, `--amp`) patch `TrainConfig` in `voice_clone/train_adapters.py` and should be treated as run-time changes to the table above.