# BYOS (Bring Your Own State)

BYOS is a standalone research prototype for long-context language modeling with an explicit persistent state memory `S` that complements a fixed local attention window.

The core question this repo explores is:

If a model cannot afford a huge local context window at inference, can it learn to compress useful information into a persistent state and reuse it later?

This repo is under active development. A more complete evaluation suite will be added later.

## Idea (Stateful LM)

In a standard decoder-only LM (vanilla GPT), each forward pass attends over a local window of tokens.
In BYOS, we also maintain a persistent state `S`:

- `S` is a small set of learned memory slots (`n_state` slots) stored in token-embedding space.
- Each inference step "pops" a token from the local KV window and writes it into `S` via a predictor module (routing + update).
- The transformer reads a projected version of `S` as a prefix on every block, but `S` itself persists between steps.

This design makes "long context" depend on how well `S` can store and refresh information, rather than on making `n_local` huge.

## Repo Structure

- Model + predictor + inference logic: `BYOS/byos_prototype.py`
- Training loop (single-GPU, AMP, gradient accumulation, checkpointing): `BYOS/byos_train.py`
- Data:
  - local text-file dataset with token cache: `BYOS/byos_data.py`
  - HF streaming + rolling token buffer sampler (for large corpora): `BYOS/byos_data.py`
- Eval / generation utilities: `BYOS/byos_eval.py`
- Monitoring & diagnostics: `BYOS/byos_monitor.py`
- Notebook (experiments / notes): `BYOS/byos_v0_5.ipynb`
- Configs: `BYOS/cfg_byos*.yaml`

## Key Concepts / Knobs

- `n_local`: local window length (tokens). This is what you would normally call `block_size` / context length in a vanilla GPT setup.
- `n_state`: number of persistent state slots.
  - Setting `n_state=0` enables a vanilla-GPT branch (no state).
- `h_len`: number of "history" tokens used to build an initial state for training batches.
  - Training samples `(h_ids, x_ids, y_ids)` where the span length is `h_len + n_local + 1`.
- `routing_temp`: temperature applied to the state routing logits (controls how sharp the slot assignment is).
- `tokens_per_step`: sets the effective batch size via gradient accumulation (stable way to scale tokens/update without changing `batch_size`).

## Installation

Editable install:

```bash
pip install -e .
```

Minimal dependency notes:

- If you train on Dolma via HF scripts, you likely need `datasets<4.0.0` and `hf_trust_remote_code: true` (see below).

## Training

Run training with a YAML config:

```bash
python BYOS/byos_train.py BYOS/cfg_byos.yaml
```

Checkpoint output is controlled by `ckpt_dir` and `experiment_name` in the config.

### PG19 (Local Text Files)

Use a local dataset directory layout with `train/` and `validation/` splits containing `.txt` files.
The loader will build a token cache under `_token_cache/` inside the dataset dir (or `token_cache_dir` if set).

Relevant config keys:

- `dataset_dir`, `train_split`, `val_split`
- `max_train_files`, `max_val_files`
- `max_train_items`, `max_val_items` (only used for the non-streaming HF text path)

### Dolma (HF Streaming)

The training loop supports HF streaming with a rolling CPU token buffer that enables random-span sampling without random-access files.

Use `BYOS/cfg_byos_dolma.yaml` as a starting point.

Important keys:

- `hf_streaming: true`
- `hf_trust_remote_code: true`
- `hf_text_field: text`
- `hf_token_buffer_size`, `hf_prefill_tokens`, `hf_refresh_tokens`
- `hf_max_doc_tokens` (0 means "no cap", otherwise truncate docs to this many tokens)

Notes:

- Many streaming datasets do not provide a clean `validation` split. If the requested `val_split` is missing, the code prints:
  `INFO: NO VAL -> Autofallback to train split for validation`.
  In this fallback mode, "val loss" is an in-sample proxy and not a true generalization metric.

## Monitoring

The repo includes simple monitoring utilities for routing/state dynamics (entropy, mass on top-k slots, state update norms).
These are meant to answer questions like:

- Is routing collapsing (always writing to the same slot)?
- Is state actually changing (non-trivial `delta_s_norm`)?
- Does the state "age" during long inference runs?

See: `BYOS/byos_monitor.py`.

## torch.compile Notes

BYOS samples variable `h_len` during training. To keep `torch.compile(..., dynamic=False)` usable (static graphs),
the compiled path excludes the predictor and variable-length history. Concretely:

- Predictor + history-to-state (`h_ids -> s0`) runs eagerly.
- The compiled function runs the fixed-shape transformer stack given `(x_ids, s0, y_ids)` and fixed RoPE caches for `n_local`.

If you are on the VRAM cliff, `torch.compile` can increase peak memory. Reducing microbatch `batch_size` (and keeping
`tokens_per_step` constant via accumulation) is the most reliable way to keep training stable.

## Layout

This section is kept for quick reference:

- Model + predictor + inference: `BYOS/byos_prototype.py`
- Training: `BYOS/byos_train.py`
- Data: `BYOS/byos_data.py`
- Eval: `BYOS/byos_eval.py`
- Monitoring: `BYOS/byos_monitor.py`
- Notebook: `BYOS/byos_v0_5.ipynb`
- Configs: `BYOS/cfg_byos*.yaml`
