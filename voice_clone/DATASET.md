# Multilingual training data

Voice-clone training expects **one Kokoro checkpoint** per run (`TrainConfig.kokoro_repo_id` / `--kokoro-repo`). Text is turned into token ids with **`KPipeline(lang_code=..., repo_id=..., model=False)`** (quiet G2P only), then the same vocabulary as that checkpoint’s `config.json`.

## Manifest format (JSONL)

One UTF-8 JSON object per line. Required fields:

| Field | Meaning |
|--------|--------|
| `ref_wav` | Path to reference speaker clip (any rate; loaded as **mono 16 kHz**). |
| `target_wav` | Path to target utterance for reconstruction loss (any rate; loaded as **mono 24 kHz**). |
| `text` | Orthographic text for the **target** utterance (language-specific G2P). |
| `lang_code` | Kokoro language id (e.g. `a` American English, `b` British, `z` Mandarin, …) or an alias such as `en-us`. Must be supported by `KPipeline` (`kokoro.pipeline.LANG_CODES`). |

Optional fields:

| Field | Meaning |
|--------|--------|
| `phonemes` | If set, **skips G2P**; must be a phoneme string compatible with the chosen `repo_id` vocabulary (same character-wise mapping as `KModel.forward`). |
| `speaker_id` | Optional string for logging; passed through the collate fn when present. |

Paths may be relative; they are resolved against `--manifest-root` if given, otherwise against the manifest file’s directory.

Example line:

```json
{"ref_wav": "clips/spk1/ref.wav", "target_wav": "clips/spk1/utt0001.wav", "text": "Hello world.", "lang_code": "a", "speaker_id": "spk1"}
```

## Alignment and length

- **One row = one supervised pair**: `text` should describe the **full** `target_wav`. If `KPipeline` splits the line into **multiple** chunks (long English tokenization or long non-English segments), `VoiceCloneManifestDataset` raises by default so you do not silently misalign text and audio. Keep lines short enough for a **single** G2P segment, or pre-split audio and text into multiple rows.
- Phoneme strings are capped at **510** characters before BOS/EOS (Kokoro pipeline limit). The model context is `plbert.max_position_embeddings` from `config.json` (typically 512 tokens including BOS/EOS).

## `repo_id` and vocabulary constraints

- **`kokoro_repo_id` must match the weights you train with.** `VoiceCloneManifestDataset` passes the same `repo_id` into `KPipeline` so Chinese G2P (`lang_code` `z`) picks the correct Misaki backend (e.g. `Kokoro-82M` vs `Kokoro-82M-v1.1-zh`) in line with upstream Kokoro.
- **Do not mix checkpoints in a single training job.** Different Hugging Face repos can differ in **`vocab`** and **`n_token`**; token ids from one checkpoint are invalid for another. Filter your manifest (or run separate jobs) so every row is compatible with the job’s `--kokoro-repo`.
- **Multilingual rows** (`lang_code` varies) are allowed **only** when every language is supported by that same checkpoint’s G2P stack and vocabulary. If a phoneme character is missing from `vocab`, it is dropped when building `input_ids` (same behavior as `KModel.forward`), which can hurt quality—validate phonemization for rare languages.

## Usage

```bash
# From repo root (with kokoro + voice_clone on PYTHONPATH or editable installs)
python -m voice_clone.train_adapters --manifest path/to/train.jsonl --kokoro-repo hexgrad/Kokoro-82M
```

Optional: `--manifest-root /data/audio` to resolve relative paths.
