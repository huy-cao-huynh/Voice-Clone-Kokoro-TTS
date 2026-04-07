# Architecture: Voice Cloning with Kokoro TTS

This document matches the current `voice_clone` training and inference code: frozen [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) plus trainable **SegmentGST** and Kokoro **L-adapters**, with a separate **SLM feature discriminator**.

---

## 1. Frozen speaker front-end: WavLM-Base+ SV

- **Checkpoint**: `microsoft/wavlm-base-plus-sv` (`TrainConfig.wavlm_model_id`, `WavLMSV`).
- **Wrapper**: Hugging Face `WavLMForXVector` + `Wav2Vec2FeatureExtractor` (weights frozen, module stays in `eval()` even when training adapters).
- **Reference audio**: mono, **16 kHz** (hard requirement of this checkpoint).
- **Outputs** (see `WavLMSVOutput` in `voice_clone/wavlm_sv.py`):
  - **Pooled x-vector** `embeddings`: shape **(B, 512)**, L2-normalized when `normalize_embeddings=True` (default for speaker loss).
  - **Last encoder layer frame features**: **(B, T_frames, 768)** — WavLM-Base+ hidden size; used as SegmentGST queries and as SLM real/fake feature inputs.
  - **Frame mask** **(B, T_frames)** derived from the feature extractor’s sample-level attention mask and WavLM’s `_get_feat_extract_output_lengths`.

There is **no ECAPA-TDNN** in this repo; speaker identity for training losses comes from the same frozen **WavLM-SV x-vector head** (512-D), not from a separate ECAPA stack.

For **training and inference on device**, `WavLMSV` (`voice_clone/wavlm_sv.py`) applies **GPU-side preprocessing** that matches Hugging Face `Wav2Vec2FeatureExtractor`: padding to batch max length, attention mask, sampling rate semantics, and optional per-audio normalization when the checkpoint sets `do_normalize`, so tensor waveforms are not sent through the NumPy/CPU extractor on the hot path. The normalization function (`_zero_mean_unit_var_norm_torch`) uses **ε = 1e-5** in the variance denominator (matching `LayerNorm` / `InstanceNorm1d` conventions) to limit backward gradient amplification when generated audio has near-zero variance. The **training loop** (`voice_clone/train_adapters.py`) runs **one** Kokoro `forward_with_tokens` per step on the live generator graph, then runs **one** WavLM forward on the generated audio (with grad) whose output is shared: the SLM discriminator receives **detached** features while mel, speaker, and SLM generator losses use the **grad-tracked** features for backward through adapters and GST (see §4 and §6 for shapes and optimizer order).

---

## 2. Trainable bottleneck: SegmentGST

Implemented in `voice_clone/segment_gst.py`.

- **Learnable bank** `B`: **(512, 256)** — `num_bases=512`, `embed_dim=256`, initialized `N(0, 0.02)`.
- **Queries**: WavLM frames projected with `nn.Linear(768 → 256)`.
- **Attention**: `nn.MultiheadAttention` with **8 heads**, `batch_first=True`, **keys/values = bank** (expanded per batch). Frame validity is **not** passed as `key_padding_mask`; invalid frames are zeroed out **after** MHA via masked mean pooling.
- **Post-MHA**: `LayerNorm(256)`, `Dropout(0.1)`.
- **Pooling**: mask-weighted mean over time → **(B, 256)** `pooled_style`.
- **Readout**: `Linear(256 → 256)` → **`ref_s`** **(B, 256)**, **zero-initialized** (weight and bias) so `ref_s` starts as a zero vector (neutral AdaIN conditioning) rather than random noise.
- **Kokoro split** (matches `KModel.forward_with_tokens`):
  - `ref_s[:, :128]` → **decoder AdaIN** style `s` (timbre / decoder conditioning).
  - `ref_s[:, 128:]` → **prosody** path as `s` into duration/F0/N blocks (`style_dim=128` in Kokoro `config.json`).
  - Full **`ref_s`** is **`z_style`** for all **L-adapters** (256-D concatenation in adapter MLPs).

---

## 3. Frozen backbone: Kokoro-82M + L-adapter injection sites

- **Default repo**: `hexgrad/Kokoro-82M` (`TrainConfig.kokoro_repo_id`).
- **Frozen**: full `KModel` except the injected adapter parameters (`freeze_kokoro_except_adapters` in `voice_clone/train_adapters.py`).

Numeric backbone settings come from Kokoro’s published `config.json` (current HF snapshot):

| Setting | Value |
|--------|--------|
| PLBERT hidden size | 768 |
| `hidden_dim` (`d_hid`, duration/encoder width) | 512 |
| `n_layer` (duration encoder depth) | 3 |
| Kokoro `style_dim` (AdaIN / prosody `s`) | 128 |
| `n_mels` | 80 |
| ISTFTNet `upsample_initial_channel` | 512 |
| ISTFTNet `upsample_rates` | [10, 6] → **2** upsample stages |

**Forward sketch** (see `kokoro/kokoro/model.py`): phoneme `input_ids` → PLBERT → linear to **512**-D → prosody/duration → alignment → decoder → **24 kHz** waveform.

### 3.1 Residual L-adapter definition

In `voice_clone/adapters.py`, each adapter implements:

\[
h' = h + W_{\text{up}}\bigl(\mathrm{ReLU}(W_{\text{down}}([h \,\|\, z_{\text{style}}]))\bigr)
\]

with `h` **(B, C, T)** and `z_style` **(B, 256)** broadcast along **T**. Per adapter:

- `W_down`: **Linear(C + 256 → 64)** (`adapter_bottleneck=64` from `TrainConfig`).
- `W_up`: **Linear(64 → C)**, **zero-initialized** (weight and bias) so the adapter starts as identity and the pretrained Kokoro behavior is preserved at step 0.

### 3.2 Where adapters attach (block-level)

| Subsystem | # adapters | Channel width **C** at each site | When it runs |
|-----------|-------------|----------------------------------|--------------|
| **Duration encoder** (`DurationEncoder` in `kokoro/kokoro/modules.py`) | **3** (= `n_layer`) | **512** each | After each **AdaLayerNorm** block (after every 1-layer BiLSTM stage), using full **`z_style` = `ref_s`**. |
| **ISTFTNet decoder** (`Decoder` in `kokoro/kokoro/istftnet.py`) | **5** | **(1024, 1024, 1024, 1024, 512)** = `DECODER_L_ADAPTER_HIDDEN_DIMS` | Index **0**: after **encode** (`AdainResBlk1d` to 1024 ch). Indices **1–4**: after each **decode** `AdainResBlk1d` (last block upsamples to 512 ch). |
| **Generator** (ISTFTNet `Generator`) | **2** (= `len(upsample_rates)`) | **(256, 128)** from `generator_l_adapter_hidden_dims(512, 2)` | After each upsample/resblock segment, in order. |

**Total L-adapters**: 3 + 5 + 2 = **10**, all conditioned on **`ref_s` (256-D)**.

---

## 4. Tensor shapes through the conditioning path

Rough data flow for one training step (batch size **B**):

1. **Reference waveform** `ref_wav_16k`: **(B, N_16k)** → WavLM → frames **(B, T_w, 768)**, mask **(B, T_w)**.
2. **SegmentGST** → `ref_s` **(B, 256)**; internally MHA output **(B, T_w, 256)** → pooled **(B, 256)**.
3. **Kokoro** `forward_with_tokens(input_ids, ref_s)`:
   - `input_ids`: **(B, L)** (phoneme token ids including boundaries).
   - PLBERT sequence: **(B, L, 768)** → `bert_encoder` → **(B, 512, L)** (duration input layout as in `KModel`).
   - Duration path features stay at hidden width **512** at adapter sites; decoder feature maps use the channel counts in §3.2.
4. **Synthesis output** `pred_wav`: **(B, T_24k)** mono at **24 kHz**, hard-clamped to **[-1, 1]** via `.clamp(-1.0, 1.0)` in both training and inference to prevent out-of-range samples without gradient saturation (unlike `tanh`).

During **training**, that Kokoro forward runs **once per step** under the generator graph (WavLM reference → SegmentGST → `forward_with_tokens`). After synthesis, `pred_wav` is resampled **24 → 16 kHz** and run through frozen WavLM **once with grad** to produce `wv_gen` (`WavLMSVOutput`). The SLM discriminator receives **`wv_gen.frame_hidden_states.detach()`** so **D** does not backprop into Kokoro, while the **same** grad-tracked `wv_gen` feeds the speaker and SLM generator losses for backward through adapters and GST. Target audio is processed separately (**no grad**) via the optimized base-encoder path. This yields **3 total WavLM forwards per step** (ref, gen, tgt). See §6 for the full step order.

---

## 5. Losses (training)

Composite generator objective:

\[
\mathcal{L}_G = \lambda_{\text{mel}}\mathcal{L}_{\text{mel}} + \lambda_{\text{spk}}\mathcal{L}_{\text{spk}} + \lambda_{\text{slm}}\mathcal{L}_{\text{slm},G}
\]

with defaults `LossWeights`: **λ_mel = 1.0**, **λ_spk = 1.0**, **λ_slm = 0.05** (`voice_clone/config.py`). `λ_spk` is set equal to `λ_mel` to prevent mel gradients from overwhelming the speaker identity signal during training, which was observed to cause speaker loss regression after LR warmup.

### 5.1 Mel reconstruction (`MelReconstructionLoss`)

- **Signal**: Kokoro output vs target waveform, both **24 kHz**, cropped to matching length.
- **Frontend**: `torchaudio.transforms.MelSpectrogram` with `MelLossConfig`: **n_fft = 1024**, **hop_length = 256**, **win_length = 1024**, **n_mels** from Kokoro config (**80**), `f_min = 0`, `f_max = None` (Nyquist), `center=True`, power **1.0**.
- **Loss**: **mean L1** on **log** mels (with `log_floor = 1e-5`). **l2_weight = 0** (L2 term not used unless changed in code).

### 5.2 Speaker consistency (`speaker_cosine_loss`)

- Embeddings: frozen WavLM-SV **pooled x-vectors** **(B, 512)**.
- **Reference**: `ref_wav_16k` through WavLM with **no grad** through audio.
- **Generated**: resampled predicted waveform, with **`grad_through_input=True`** so gradients reach the generator.
- **Loss**: mean over batch of **1 − cos(a, b)** with **L2 re-normalization** of both vectors (cosine in 512-D).

### 5.3 SLM feature GAN (`SLMFeatureDiscriminator` + hinge)

- **Not** a WavLM-as-discriminator setup: a **small trainable conv discriminator** (`voice_clone/losses.py`) runs on **frozen WavLM last-layer frame features** **(B, T, 768)**.
- **Architecture**: `LayerNorm(768)` → **4** layers of **Conv1d** (kernel **7**, padding 3), hidden **256**, **LeakyReLU(0.2)** → **Conv1d → 1** channel; logits pooled to **one scalar per utterance** via **mask-weighted mean** over time. All **Conv1d** layers are wrapped with **spectral normalization** (`torch.nn.utils.spectral_norm`) when `SLMDiscriminatorConfig.use_spectral_norm=True` (default), constraining the discriminator's Lipschitz constant and stabilizing the hinge loss.
- **D step** (`slm_discriminator_loss_hinge`): hinge loss on **real** (target 16 kHz) vs **fake** (generated 16 kHz, **detached** from the shared generator forward). Optimizer **AdamW** on discriminator only.
- **G step** (`slm_generator_loss_hinge`): **−E[D(fake_features)]** (discriminator weights **frozen** / `requires_grad=False` for this forward), using WavLM features of the **same** non-detached `pred_wav` as mel and speaker losses.

**Discriminator hyperparameters** (`SLMDiscriminatorConfig`): `hidden_channels=256`, `num_layers=4`, `kernel_size=7`, `use_spectral_norm=True`.

### 5.4 Numerical safeguard: ISTFTNet exp clamping

The `Generator` in `kokoro/kokoro/istftnet.py` converts log-magnitude to linear magnitude via `torch.exp()` before the inverse STFT. To prevent overflow (especially under AMP fp16 where `exp(x)` overflows for x > ~11), the exponent input is clamped: `torch.exp(x.clamp(max=15.0))`. `exp(15) ≈ 3.27 M` is well above any physically meaningful STFT magnitude for 24 kHz audio. The conservative cap also limits backward gradient amplification through the exp (compared to a naïve `max=30`, the maximum gradient is reduced by a factor of ~3.3 × 10⁹), which matters during early training when adapter perturbations or random style vectors may produce extreme `conv_post` outputs.

---

## 6. Optimization and training loop

- **Frozen Kokoro in train() mode**: `freeze_kokoro_except_adapters` calls `kmodel.train()` (not `eval()`) to keep MIOpen RNN backward working on ROCm. Freezing is controlled by `requires_grad_(False)`, not by `eval()`. This means `nn.Dropout` layers in frozen Kokoro blocks remain active during training, adding stochastic regularization. The trainable L-adapters (`ResidualAdapter`) contain no dropout or batchnorm.
- **Generator trainable parameters**: SegmentGST + all Kokoro L-adapters (duration, decoder, generator).
- **Discriminator trainable parameters**: `SLMFeatureDiscriminator` only.
- **Optimizers**: **AdamW** for both; **lr_g = 1e-4**, **lr_d = 5e-5** (TTUR — Two Time-scale Update Rule: the discriminator learns at half the generator rate to prevent D from overpowering G early in training); **weight_decay_g = weight_decay_d = 0** (defaults).
- **Gradient clipping**: **1.0** L2 norm on generator params and on discriminator params when not `None` (`grad_clip_g`, `grad_clip_d`).
- **AMP**: optional CUDA autocast + `GradScaler` when `TrainConfig.use_amp=True` (default **False**). **Separate** `GradScaler` instances are used for generator and discriminator (`scaler_g`, `scaler_d`) so that inf/NaN in one optimizer does not force the other to drop its scale factor. Each scaler calls `update()` after its own optimizer step. Scaler states are saved and restored in checkpoints.
- **Learning rate schedule**: `SequentialLR(LinearLR → CosineAnnealingLR)` ramps `opt_g` from `1e-3 × lr_g` to full `lr_g` over `warmup_steps` (default **50**), then cosine-decays to `lr_min_g` (default **1e-6**) over the remaining steps to `max_steps`. `opt_d` gets its own schedule starting from `disc_start_step`, decaying to `lr_min_d` (default **1e-7**). When `max_steps` is not provided, the cosine horizon is estimated from `len(dataloader) * epochs / grad_accum_steps`. When resuming from a checkpoint, schedulers are fast-forwarded to match the resumed step.
- **Weight-norm stripping**: `strip_weight_norm_frozen` bakes `weight_norm` decompositions in frozen Kokoro `ConvTranspose1d` layers into static parameters at model-build time, avoiding AMP autocast cache misses on ROCm. Safe because the Kokoro model is frozen (`requires_grad=False`).

### 6.1 Gradient accumulation

Batch size stays **1** in the DataLoader (Kokoro's `forward_with_tokens` constraint), but effective batch size is increased via **gradient accumulation** over `grad_accum_steps` micro-steps (default **8**). Gradients from each micro-step are scaled by `1/grad_accum_steps` before `.backward()`, then accumulated into a single optimizer step. `global_step` increments once per effective step (not per micro-step), and LR schedulers step once per effective step.

### 6.2 Phased training (generator-only warmup)

For the first `disc_start_step` effective steps (default **500**), the discriminator is disabled:

- `slm_discriminator_backward` is skipped entirely.
- `generator_loss_backward` runs with `include_slm=False`, so `loss_g = λ_mel · L_mel + λ_spk · L_spk` (no SLM term).
- `opt_d.step()` and `sched_d.step()` are skipped; `loss_d` and `loss_slm_g` are logged as 0.

Once `global_step >= disc_start_step`, the discriminator activates and the full adversarial objective is used. The discriminator's LR warmup begins at this point.

### 6.3 Per-step order

The training loop (`train_loop` in `voice_clone/train_adapters.py`) uses a two-level structure — one outer effective step wrapping N micro-steps:

  1. **`opt_g.zero_grad`** and **`opt_d.zero_grad`** once per effective step.
  2. **For each micro-step** (N = `grad_accum_steps`):
     1. **WavLM ref** (`wavlm_ref`): reference **16 kHz** → frozen WavLM (no grad) → frame features + pooled x-vector; SegmentGST → `ref_s`.
     2. **Kokoro forward** (`kokoro_fwd`): `KModel.forward_with_tokens(input_ids, ref_s)` → `pred_wav` (**24 kHz**), under autocast when AMP is on.
     3. **WavLM gen** (`wavlm_gen`): resample `pred_wav` **24 → 16 kHz**, one WavLM forward **with grad** (`grad_through_input=True`) → `wv_gen` (`WavLMSVOutput`). Computed **once**; the same output feeds both D and G losses. A **per-tensor gradient clamp hook** (max L2 norm `wavlm_grad_max_norm`, default **1.0**) is registered on `wv_gen.frame_hidden_states` and `wv_gen.pooled_embedding`, preventing extreme upstream gradients from being amplified through the 12-layer WavLM backward pass.
     4. **WavLM tgt** (`wavlm_tgt`): resample `tgt_24` **24 → 16 kHz**, `wavlm.frame_hidden_states(...)` **without grad** → `real_feats`, `real_mask`. Uses the optimized base-encoder-only path (skips the x-vector head).
     5. **D backward** (`disc`): if `global_step >= disc_start_step`, `slm_discriminator_backward` computes hinge loss on **detached** features and calls `.backward()` scaled by `1/N`.
     6. **G backward** (`gen_backward`): `generator_loss_backward` computes mel + speaker + conditional SLM losses, scales by `1/N`, and calls `.backward()`.
  3. **Deferred optimizer steps** (after all micro-steps): unscale (AMP), clip gradients, `opt_d.step()` (if D is active), `opt_g.step()`, `scaler_d.update()` + `scaler_g.update()` (AMP).
  4. **`global_step += 1`**.
  5. **LR scheduler step**: `sched_g.step()` every step; `sched_d.step()` only when `global_step > disc_start_step`.

This gives **3 WavLM forward passes per micro-step** (ref, gen, tgt), because the generated-audio WavLM output is computed once and shared between D (detached) and G (with grad). Partial final accumulation (when the dataloader runs out mid-accumulation) still triggers the optimizer step on whatever gradients were accumulated.

**NaN guard**: after each micro-step's generator losses, the loop checks all loss values with `np.isfinite`. If any component is NaN or Inf, the `nan_detected` flag is set, the remaining micro-steps are skipped, both optimizer gradients are zeroed, and `global_step` advances without an optimizer step. A `train/nan_skips` counter is logged to wandb when active. This prevents a single corrupted sample from poisoning the accumulated gradient.

**Dataset-level audio validation** (`voice_clone/dataset.py`): `load_audio_mono` verifies `torch.isfinite(wav).all()` after resampling; non-finite samples raise `ValueError`. `VoiceCloneManifestDataset.__getitem__` enforces minimum durations: reference audio ≥ 4800 samples at 16 kHz (~0.3 s, enough WavLM frames for a meaningful speaker embedding) and target audio ≥ 4800 samples at 24 kHz (~0.2 s).

- **SLM schedule**: **`slm_d_steps_per_g_step`** (default **1**) is how many **D** backward calls run within each micro-step.
- **Kokoro inference timing**: `speed` default **1.0** (`TrainConfig.speed`), passed into `forward_with_tokens`.

Logging defaults: **`log_interval = 1`**, **`checkpoint_interval = 500`**. Metrics logged are averages over the N micro-steps in each effective step.

**Validation**: when `--val-manifest` is provided, validation metrics (`val/mel_l1`, `val/mel_l2`, `val/speaker_loss`) and audio samples are computed from the held-out set instead of the training set. This enables overfitting detection.

---

## 7. Hyperparameter summary (chosen defaults)

| Group | Parameter | Value |
|--------|-----------|--------|
| Models | Kokoro repo | `hexgrad/Kokoro-82M` |
| | WavLM repo | `microsoft/wavlm-base-plus-sv` |
| Adapters | `adapter_bottleneck` | **64** |
| SegmentGST | `num_bases` | **512** |
| | `embed_dim` | **256** |
| | `frame_dim` | **768** |
| | `num_heads` | **8** |
| | `ref_dim` / `style_dec_dim` | **256** / **128** |
| | `dropout` | **0.1** |
| Loss weights | `lambda_mel` / `lambda_spk` / `lambda_slm` | **1.0** / **1.0** / **0.05** |
| Mel loss | `sample_rate` | **24000** |
| | `n_fft` / `hop_length` / `win_length` | **1024** / **256** / **1024** |
| SLM D | `hidden_channels` / `num_layers` / `kernel_size` | **256** / **4** / **7** |
| | `use_spectral_norm` | **True** |
| Optim | `lr_g` | **1e-4** |
| | `lr_d` | **5e-5** (TTUR) |
| | `weight_decay_g`, `weight_decay_d` | **0** |
| | `grad_clip_g`, `grad_clip_d` | **1.0** |
| | `wavlm_grad_max_norm` | **1.0** (per-tensor hook on WavLM outputs) |
| | `grad_accum_steps` | **8** |
| Schedule | `warmup_steps` | **50** |
| | `lr_min_g` | **1e-6** (cosine floor for G) |
| | `lr_min_d` | **1e-7** (cosine floor for D) |
| | `disc_start_step` | **500** (G-only warmup) |
| | `slm_d_steps_per_g_step` | **1** |
| | `speed` | **1.0** |
| Run | `use_amp` | **False** |
| | `log_interval` | **1** |
| | `checkpoint_interval` | **1000** |

CLI overrides (e.g. `--kokoro-repo`, `--amp`) patch `TrainConfig` in `voice_clone/train_adapters.py` and should be treated as run-time changes to the table above.
