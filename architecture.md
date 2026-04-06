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

For **training and inference on device**, `WavLMSV` (`voice_clone/wavlm_sv.py`) applies **GPU-side preprocessing** that matches Hugging Face `Wav2Vec2FeatureExtractor`: padding to batch max length, attention mask, sampling rate semantics, and optional per-audio normalization when the checkpoint sets `do_normalize`, so tensor waveforms are not sent through the NumPy/CPU extractor on the hot path. The **training loop** (`voice_clone/train_adapters.py`) runs **one** Kokoro `forward_with_tokens` per step on the live generator graph; mel, speaker, and SLM generator losses reuse that waveform, while the SLM discriminator consumes **`pred_wav.detach()`** (see §4 and §6 for shapes and optimizer order).

---

## 2. Trainable bottleneck: SegmentGST

Implemented in `voice_clone/segment_gst.py`.

- **Learnable bank** `B`: **(512, 256)** — `num_bases=512`, `embed_dim=256`, initialized `N(0, 0.02)`.
- **Queries**: WavLM frames projected with `nn.Linear(768 → 256)`.
- **Attention**: `nn.MultiheadAttention` with **8 heads**, `batch_first=True`, **keys/values = bank** (expanded per batch). Frame validity is **not** passed as `key_padding_mask`; invalid frames are zeroed out **after** MHA via masked mean pooling.
- **Post-MHA**: `LayerNorm(256)`, `Dropout(0.1)`.
- **Pooling**: mask-weighted mean over time → **(B, 256)** `pooled_style`.
- **Readout**: `Linear(256 → 256)` → **`ref_s`** **(B, 256)**.
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
- `W_up`: **Linear(64 → C)**.

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
4. **Synthesis output** `pred_wav`: **(B, T_24k)** mono at **24 kHz**.

During **training**, that Kokoro forward runs **once per step** under the generator graph (WavLM reference → SegmentGST → `forward_with_tokens`). The SLM discriminator sees **`pred_wav.detach()`** (and detached target audio) so **D** does not backprop into Kokoro; mel, speaker, and SLM generator losses still use the **same** `pred_wav` for backward through adapters and GST.

For **SLM / speaker losses**, `pred_wav` is resampled **24 kHz → 16 kHz** (`torchaudio.functional.resample`) before re-entering frozen WavLM so frame tensors stay aligned with the SV model.

---

## 5. Losses (training)

Composite generator objective:

\[
\mathcal{L}_G = \lambda_{\text{mel}}\mathcal{L}_{\text{mel}} + \lambda_{\text{spk}}\mathcal{L}_{\text{spk}} + \lambda_{\text{slm}}\mathcal{L}_{\text{slm},G}
\]

with defaults `LossWeights`: **λ_mel = 1.0**, **λ_spk = 0.1**, **λ_slm = 0.05** (`voice_clone/config.py`).

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
- **Architecture**: `LayerNorm(768)` → **4** layers of **Conv1d** (kernel **7**, padding 3), hidden **256**, **LeakyReLU(0.2)** → **Conv1d → 1** channel; logits pooled to **one scalar per utterance** via **mask-weighted mean** over time.
- **D step** (`slm_discriminator_loss_hinge`): hinge loss on **real** (target 16 kHz) vs **fake** (generated 16 kHz, **detached** from the shared generator forward). Optimizer **AdamW** on discriminator only.
- **G step** (`slm_generator_loss_hinge`): **−E[D(fake_features)]** (discriminator weights **frozen** / `requires_grad=False` for this forward), using WavLM features of the **same** non-detached `pred_wav` as mel and speaker losses.

**Discriminator hyperparameters** (`SLMDiscriminatorConfig`): `hidden_channels=256`, `num_layers=4`, `kernel_size=7`.

---

## 6. Optimization and training loop

- **Generator trainable parameters**: SegmentGST + all Kokoro L-adapters (duration, decoder, generator).
- **Discriminator trainable parameters**: `SLMFeatureDiscriminator` only.
- **Optimizers**: **AdamW** for both; **lr_g = lr_d = 1e-4**; **weight_decay_g = weight_decay_d = 0** (defaults).
- **Gradient clipping**: **1.0** L2 norm on generator params and on discriminator params when not `None` (`grad_clip_g`, `grad_clip_d`).
- **AMP**: optional CUDA autocast + `GradScaler` when `TrainConfig.use_amp=True` (default **False**).
- **Per-step order** (`train_loop` / `generator_loss_backward_step` in `voice_clone/train_adapters.py`):
  1. **`opt_g.zero_grad`** once.
  2. **One differentiable forward**: reference **16 kHz** → frozen WavLM → SegmentGST → **`KModel.forward_with_tokens`** → `pred_wav` (**24 kHz**), under autocast when AMP is on.
  3. **`slm_d_steps_per_g_step`** times: **D** forward/backward/step on **real vs `pred_wav.detach()`** (WavLM features at 16 kHz); **`opt_d.zero_grad`** inside the D step.
  4. **G** losses (mel + speaker + SLM generator hinge) from the **same** `pred_wav` tensor, then backward and **`opt_g.step()`** (no second Kokoro forward).
- **SLM schedule**: **`slm_d_steps_per_g_step`** (default **1**) is how many **D** updates run **after** that shared forward and **before** each **G** update.
- **Kokoro inference timing**: `speed` default **1.0** (`TrainConfig.speed`), passed into `forward_with_tokens`.

Logging defaults: **`log_interval = 10`**, **`checkpoint_interval = 500`**.

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
| Loss weights | `lambda_mel` / `lambda_spk` / `lambda_slm` | **1.0** / **0.1** / **0.05** |
| Mel loss | `sample_rate` | **24000** |
| | `n_fft` / `hop_length` / `win_length` | **1024** / **256** / **1024** |
| SLM D | `hidden_channels` / `num_layers` / `kernel_size` | **256** / **4** / **7** |
| Optim | `lr_g`, `lr_d` | **1e-4** |
| | `weight_decay_g`, `weight_decay_d` | **0** |
| | `grad_clip_g`, `grad_clip_d` | **1.0** |
| Schedule | `slm_d_steps_per_g_step` | **1** |
| | `speed` | **1.0** |
| Run | `use_amp` | **False** |
| | `log_interval` | **10** |
| | `checkpoint_interval` | **500** |

CLI overrides (e.g. `--kokoro-repo`, `--amp`) patch `TrainConfig` in `voice_clone/train_adapters.py` and should be treated as run-time changes to the table above.
