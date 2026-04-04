# Architecture: Voice Cloning with Kokoro TTS

## Phase 1: Feature Extraction and Aggregation

The initial stage of the pipeline involves processing the unseen reference audio to extract a robust, content-agnostic speaker identity vector.

### SSL front-end

Utilize a pre-trained WavLM-Base+ or WavLM-Large model. The reference waveform (resampled to 16 kHz to match WavLM’s expected input) is passed through the frozen WavLM encoder. The output is a sequence of frame-level hidden states $H \in \mathbb{R}^{T \times 1024}$, where $T$ is the number of acoustic frames.

### Attentive pooling aggregation

To convert the variable-length sequence $H$ into a fixed-size vector without losing critical temporal characteristics, pass the representations through an ECAPA-TDNN head. The attentive statistics pooling layer calculates the attention weights for each frame, producing a global speaker embedding $z_{\text{speaker}} \in \mathbb{R}^{256}$. This dimensionality must be chosen to tightly match the latent spaces typical of Kokoro / StyleTTS2 condition vectors.

---

## Phase 2: Bottleneck layer regularization

Drawing directly from the findings of the Google VT module research, the $z_{\text{speaker}}$ embedding must be heavily constrained before being injected into the Kokoro backbone. Feeding unconstrained, high-dimensional embeddings extracted by WavLM directly into an 82M-parameter model will almost certainly overwhelm the decoder, leading to a catastrophic degradation of linguistic content and causing the model to output garbled noise.

The optimal design choice is to implement a **SegmentGST** (Segment Global Style Token) bottleneck.

- Initialize a learnable embedding bank (the simplex) $B \in \mathbb{R}^{N \times d}$, where $N$ is the number of voice bases (e.g., 512 or 1024) and $d$ is the embedding dimensionality (e.g., 256).
- Rather than using a single global vector, allow the bottleneck’s multi-head attention mechanism to query the bank $B$ using the compressed sequential outputs of the ECAPA-TDNN before the final pooling stage.
- The attention weights map the reference audio to a convex combination of the learned bases, producing a highly regularized style vector $\tilde{z}_{\text{style}}$.

This bottleneck enforces that any unseen speaker is mathematically represented strictly as a combination of acoustic archetypes already understood by the training paradigm. By constraining the embedding to lie within this learned simplex, the adapter is prevented from hallucinating out-of-distribution acoustic features that the Kokoro decoder cannot process.

---

## Phase 3: Residual adapter architecture and injection routing

Kokoro-82M utilizes a decoder-only architecture heavily reliant on textual token embeddings and grapheme-to-phoneme (G2P) alignments provided by the misaki library. To adapt the model without altering its foundational weights, residual adapters (specifically Layer adapters, or L-adapters) must be carefully inserted into the execution graph.

A standard residual adapter consists of a down-projection linear layer, a non-linear activation function, and an up-projection linear layer, wrapped in a residual connection. For this application, the adapter must fuse the hidden states of the backbone with the regularized speaker embedding:

$$
h_{\text{adapted}} = h_{\text{input}} + W_{\text{up}}\bigl(\mathrm{ReLU}\bigl(W_{\text{down}}\bigl([h_{\text{input}} \parallel \tilde{z}_{\text{style}}]\bigr)\bigr)\bigr)
$$

The strategic choice of where to inject these adapters within Kokoro determines the success of the voice cloning, as demonstrated by the Google VT architecture.

### Duration predictor injection

Kokoro relies on predicting the length of each phoneme to establish the temporal rhythm of the speech. Injecting an adapter into the hidden layers of the duration predictor ensures that the synthesized speech adopts the target speaker’s unique pacing, speech rate, and pause structures.

### Prosody and feature decoder injection

To clone the actual timbre, pitch, and vocal tract characteristics, a second set of residual adapters must be inserted into the transformer blocks of the feature decoder, prior to the ISTFTNet vocoder. This residual addition forces the latent acoustic features to shift toward the target speaker’s spectral envelope without overriding the linguistic intent.

---

## Phase 4: Training regime and objective functions

The paramount advantage of this architecture is parameter efficiency. Because the WavLM encoder, the ECAPA-TDNN extractor, and the entire Kokoro-82M backbone are kept frozen, the only parameters being updated during training are the SegmentGST bottleneck and the lightweight residual adapters. These represent a fraction of a percent of the total pipeline weights.

The training dataset should comprise a diverse, multi-speaker corpus (such as LibriTTS or VCTK). During each training step:

1. A reference utterance and a target text sequence from the same speaker are sampled.
2. The reference utterance is passed through the frozen WavLM and the trainable adapter pipeline to produce the conditioned latent variables.
3. Kokoro generates the predicted mel-spectrogram or waveform based on the text and the adapted condition.

The loss function must carefully balance acoustic reconstruction with zero-shot speaker similarity:

### Reconstruction loss

Standard L1/L2 loss on the predicted mel-spectrograms against the ground truth to ensure the text is properly spoken.

### Speaker consistency loss

A pre-trained speaker verification network (e.g., a frozen WavLM–ECAPA model) evaluates the generated audio against the reference. The cosine similarity between the embedding of the generated audio and the original reference is maximized:

$$
\mathcal{L}_{\text{speaker}} = 1 - \cos\bigl(\mathrm{Emb}_{\text{target}}, \mathrm{Emb}_{\text{generated}}\bigr)
$$

### Adversarial SLM loss

Following the original StyleTTS 2 methodology, a frozen WavLM model can act as a feature-level discriminator. This ensures the generated speech matches the high-level naturalness of real human speech, penalizing the adapters if they produce robotic or degraded artifacts.
