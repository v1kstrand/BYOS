import json
import hashlib
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import itertools

TOKENIZER_CONFIGS = {
    "gpt2": {
        "hf_id": "gpt2",
        "checkpoint_dir": "/notebooks/BYOS/gpt2_tokenizer",
        "token_cache_dtype": "int32",
    },
    "sentencepiece": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "checkpoint_dir": "/notebooks/BYOS/mistral_tokenizer",
        "token_cache_dtype": "int32",
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "checkpoint_dir": "/notebooks/BYOS/mistral_tokenizer",
        "token_cache_dtype": "int32",
    },
}


def _ensure_tokenizer_dir(path: Path, hf_id: str):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tok_json = path / "tokenizer.json"
    tok_spm = path / "tokenizer.model"
    if tok_json.exists() or tok_spm.exists():
        return path
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to download tokenizers. Install with: pip install transformers"
        ) from exc
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.save_pretrained(path)
    return path


def resolve_tokenizer_config(name: str | None):
    key = str(name or "sentencepiece").strip().lower()
    if key not in TOKENIZER_CONFIGS:
        raise ValueError(f"unsupported tokenizer setting: {key}")
    cfg = TOKENIZER_CONFIGS[key]
    checkpoint_dir = _ensure_tokenizer_dir(cfg["checkpoint_dir"], cfg["hf_id"])
    return {
        "name": key,
        "checkpoint_dir": checkpoint_dir,
        "token_cache_dtype": cfg["token_cache_dtype"],
    }


def parse_h_len_cfg(h_len_cfg):
    if isinstance(h_len_cfg, str):
        cleaned = h_len_cfg.strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            parts = cleaned[1:-1].split(",")
            return [int(p.strip()) for p in parts if p.strip()]
        if cleaned.startswith("[") and cleaned.endswith("]"):
            parts = cleaned[1:-1].split(",")
            return [int(p.strip()) for p in parts if p.strip()]
    return h_len_cfg


def sample_h_len(h_len_cfg, *, generator: torch.Generator | None = None) -> int:
    if isinstance(h_len_cfg, (list, tuple)):
        if len(h_len_cfg) != 2:
            raise ValueError("h_len must be an int or a 2-item list/tuple")
        h_min, h_max = int(h_len_cfg[0]), int(h_len_cfg[1])
        if h_min < 0 or h_max < h_min:
            raise ValueError("h_len range must satisfy 0 <= min <= max")
        if h_min == h_max:
            return h_min
        return int(torch.randint(h_min, h_max + 1, (1,), generator=generator).item())
    return int(h_len_cfg)


def _h_len_bounds(h_len_cfg):
    if isinstance(h_len_cfg, (list, tuple)):
        return int(h_len_cfg[0]), int(h_len_cfg[1])
    val = int(h_len_cfg)
    return val, val


def list_split_files(dataset_dir: Path, split: str):
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    return sorted(split_dir.rglob("*.txt"))


def select_files(files, max_files: int, seed: int):
    if max_files is None or max_files <= 0 or max_files >= len(files):
        return files
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    return files[:max_files]


def _cache_paths(txt_path: Path, dataset_dir: Path, cache_dir: Path):
    rel = txt_path.relative_to(dataset_dir).with_suffix("")
    safe = str(rel).replace(os.sep, "__")
    bin_path = cache_dir / f"{safe}.bin"
    meta_path = cache_dir / f"{safe}.json"
    return bin_path, meta_path


def _load_meta(meta_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return int(meta["length"]), np.dtype(meta["dtype"])


def _build_cache(txt_path: Path, bin_path: Path, meta_path: Path, tokenizer, dtype):
    text = txt_path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text, device=torch.device("cpu")).to(dtype=torch.long)
    arr = tokens.cpu().numpy().astype(dtype, copy=False)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = bin_path.with_suffix(".tmp")
    arr.tofile(tmp_path)
    tmp_path.replace(bin_path)
    meta = {"length": int(arr.size), "dtype": str(arr.dtype)}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return int(arr.size), arr.dtype


def _open_memmap(bin_path: Path, length: int, dtype):
    return np.memmap(bin_path, mode="r", dtype=dtype, shape=(length,))


class MemmapCache:
    def __init__(self, max_items: int = 4):
        self.max_items = max_items
        self.cache = {}
        self.order = []

    def get(self, key, loader):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        value = loader()
        if self.max_items > 0 and len(self.cache) >= self.max_items:
            oldest = self.order.pop(0)
            self.cache.pop(oldest, None)
        self.cache[key] = value
        self.order.append(key)
        return value


class TokenFileSampler(IterableDataset):
    def __init__(
        self,
        files,
        *,
        dataset_dir: Path,
        tokenizer,
        batch_size: int,
        n_local: int,
        h_len_cfg,
        seed: int,
        cache_dir: Path,
        cache_dtype,
        mem_cache_items: int = 4,
        max_retries: int = 20,
    ):
        super().__init__()
        self.files = list(files)
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_local = n_local
        self.h_len_cfg = h_len_cfg
        self.seed = seed
        self.cache_dir = cache_dir
        self.cache_dtype = cache_dtype
        self.mem_cache_items = mem_cache_items
        self.max_retries = max_retries
        h_min, _ = _h_len_bounds(h_len_cfg)
        self.min_span = h_min + n_local + 1

    def _get_tokens(self, path: Path, mem_cache: MemmapCache, meta_cache: dict):
        bin_path, meta_path = _cache_paths(path, self.dataset_dir, self.cache_dir)

        if path in meta_cache:
            length, dtype = meta_cache[path]
        else:
            if bin_path.exists() and meta_path.exists():
                length, dtype = _load_meta(meta_path)
            else:
                length, dtype = _build_cache(
                    path, bin_path, meta_path, self.tokenizer, self.cache_dtype
                )
            meta_cache[path] = (length, dtype)

        def _loader():
            return _open_memmap(bin_path, length, dtype)

        tokens = mem_cache.get(bin_path, _loader)
        return tokens, length

    def __iter__(self):
        info = get_worker_info()
        seed = self.seed if info is None else self.seed + info.id
        gen = torch.Generator()
        gen.manual_seed(seed)

        if not self.files:
            raise ValueError("no dataset files found")

        mem_cache = MemmapCache(max_items=self.mem_cache_items)
        meta_cache = {}
        too_short = set()
        file_count = len(self.files)

        while True:
            h_len = sample_h_len(self.h_len_cfg, generator=gen)
            span = h_len + self.n_local + 1
            h_ids = []
            x_ids = []
            y_ids = []

            for _ in range(self.batch_size):
                for _ in range(self.max_retries):
                    file_idx = int(torch.randint(0, file_count, (1,), generator=gen).item())
                    path = self.files[file_idx]
                    if path in too_short:
                        continue
                    tokens, length = self._get_tokens(path, mem_cache, meta_cache)
                    if length < self.min_span:
                        too_short.add(path)
                        continue
                    if length <= span:
                        continue
                    start = int(torch.randint(0, length - span, (1,), generator=gen).item())
                    h_slice = tokens[start : start + h_len]
                    x_slice = tokens[start + h_len : start + h_len + self.n_local]
                    y_slice = tokens[start + h_len + 1 : start + h_len + 1 + self.n_local]
                    h_ids.append(torch.from_numpy(np.asarray(h_slice).copy()))
                    x_ids.append(torch.from_numpy(np.asarray(x_slice).copy()))
                    y_ids.append(torch.from_numpy(np.asarray(y_slice).copy()))
                    break
                else:
                    raise RuntimeError("failed to sample a valid span from dataset")

            yield torch.stack(h_ids, dim=0), torch.stack(x_ids, dim=0), torch.stack(
                y_ids, dim=0
            ), h_len


def build_dataloaders_from_dir(
    *,
    dataset_dir: Path,
    tokenizer,
    batch_size: int,
    n_local: int,
    h_len_cfg,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    train_split: str = "train",
    val_split: str = "validation",
    cache_dir: Path | None = None,
    cache_dtype: str = "int32",
    max_train_files: int = 0,
    max_val_files: int = 0,
):
    cache_dir = cache_dir or (dataset_dir / "_token_cache")
    cache_dir = Path(cache_dir)

    train_files = list_split_files(dataset_dir, train_split)
    val_files = list_split_files(dataset_dir, val_split)

    train_files = select_files(train_files, max_train_files, seed)
    val_files = select_files(val_files, max_val_files, seed + 1)
    print(
        f"INFO: dataset_dir={dataset_dir} train_files={len(train_files)} val_files={len(val_files)} cache_dir={cache_dir}"
    )

    train_ds = TokenFileSampler(
        train_files,
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        n_local=n_local,
        h_len_cfg=h_len_cfg,
        seed=seed,
        cache_dir=cache_dir / "train",
        cache_dtype=np.dtype(cache_dtype),
    )
    val_ds = TokenFileSampler(
        val_files,
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        n_local=n_local,
        h_len_cfg=h_len_cfg,
        seed=seed + 1000,
        cache_dir=cache_dir / "validation",
        cache_dtype=np.dtype(cache_dtype),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


class _TokenRingBuffer:
    """Fixed-capacity ring buffer for token ids (CPU).

    This supports cheap appends and random contiguous span sampling, which is
    needed for HF streaming datasets where random access is not available.
    """

    def __init__(self, capacity: int, *, dtype: torch.dtype = torch.int32):
        if capacity <= 0:
            raise ValueError("token buffer capacity must be > 0")
        self.capacity = int(capacity)
        self.buf = torch.empty((self.capacity,), dtype=dtype, device="cpu")
        self.start = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    def append(self, tokens: torch.Tensor) -> None:
        if tokens is None:
            return
        tokens = tokens.detach().to(device="cpu")
        n = int(tokens.numel())
        if n <= 0:
            return

        if tokens.dtype != self.buf.dtype:
            tokens = tokens.to(dtype=self.buf.dtype)

        if n >= self.capacity:
            # Keep the most recent `capacity` tokens.
            tokens = tokens[-self.capacity :]
            self.buf.copy_(tokens)
            self.start = 0
            self.size = self.capacity
            return

        overflow = max(0, self.size + n - self.capacity)
        if overflow:
            self.start = (self.start + overflow) % self.capacity
            self.size -= overflow

        end = (self.start + self.size) % self.capacity
        first = min(n, self.capacity - end)
        self.buf[end : end + first] = tokens[:first]
        rest = n - first
        if rest:
            self.buf[0:rest] = tokens[first:]
        self.size += n

    def sample_span(self, span: int, *, generator: torch.Generator) -> torch.Tensor | None:
        span = int(span)
        if span <= 0:
            raise ValueError("span must be > 0")
        if self.size < span:
            return None
        max_off = self.size - span
        off = int(torch.randint(0, max_off + 1, (1,), generator=generator).item())
        idx = (self.start + off) % self.capacity
        if idx + span <= self.capacity:
            return self.buf[idx : idx + span].clone()
        first = self.capacity - idx
        return torch.cat([self.buf[idx:], self.buf[: span - first]]).clone()


class HFStreamingTokenBufferSampler(IterableDataset):
    def __init__(
        self,
        *,
        dataset: str,
        dataset_config: str,
        split: str,
        tokenizer,
        text_field: str,
        batch_size: int,
        n_local: int,
        h_len_cfg,
        seed: int,
        trust_remote_code: bool,
        storage_block_size: int,
        streaming_read_max_retries: int,
        streaming_read_retry_interval_s: float,
        holdout_mode: str,
        holdout_role: str,
        holdout_salt: str,
        holdout_id_field: str,
        val_pct: float,
        test_pct: float,
        restart_on_stream_error: bool,
        restart_sleep_s: float,
        max_restarts: int,
        shuffle_buffer: int,
        token_buffer_size: int,
        prefill_tokens: int,
        refresh_tokens: int,
        max_doc_tokens: int,
        add_eos: bool,
        max_items: int = 0,
    ):
        super().__init__()
        self.dataset = str(dataset)
        self.dataset_config = str(dataset_config)
        self.split = str(split)
        self.tokenizer = tokenizer
        self.text_field = str(text_field or "text")
        self.batch_size = int(batch_size)
        self.n_local = int(n_local)
        self.h_len_cfg = h_len_cfg
        self.seed = int(seed)
        self.trust_remote_code = bool(trust_remote_code)
        self.storage_block_size = int(storage_block_size)
        self.streaming_read_max_retries = int(streaming_read_max_retries)
        self.streaming_read_retry_interval_s = float(streaming_read_retry_interval_s)
        self.holdout_mode = str(holdout_mode or "none").strip().lower()
        self.holdout_role = str(holdout_role or "train").strip().lower()
        self.holdout_salt = str(holdout_salt or "")
        self.holdout_id_field = str(holdout_id_field or "id")
        self.val_pct = float(val_pct)
        self.test_pct = float(test_pct)
        self.restart_on_stream_error = bool(restart_on_stream_error)
        self.restart_sleep_s = float(restart_sleep_s)
        self.max_restarts = int(max_restarts)
        self.shuffle_buffer = int(shuffle_buffer)
        self.token_buffer_size = int(token_buffer_size)
        self.prefill_tokens = int(prefill_tokens)
        self.refresh_tokens = int(refresh_tokens)
        self.max_doc_tokens = int(max_doc_tokens)
        self.add_eos = bool(add_eos)
        self.max_items = int(max_items)

        h_min, _ = _h_len_bounds(h_len_cfg)
        self.min_span = int(h_min + self.n_local + 1)

        # Ensure we can sample at least one full batch after prefill.
        min_prefill = self.min_span * max(2, self.batch_size)
        if self.prefill_tokens < min_prefill:
            self.prefill_tokens = min_prefill

    def _holdout_bucket(self, ex) -> str:
        if self.holdout_mode in ("", "none", "off", "false", "0"):
            return "train"
        if self.holdout_mode not in ("hash_id", "hash", "id_hash"):
            raise ValueError(f"unsupported holdout_mode: {self.holdout_mode}")
        if not self.holdout_salt:
            raise ValueError("holdout_salt must be set when holdout_mode=hash_id")

        if not isinstance(ex, dict):
            return "train"
        ex_id = ex.get(self.holdout_id_field)
        if ex_id is None:
            return "train"

        # Stable mapping to [0, 1). Do not use Python's built-in hash().
        key = f"{self.holdout_salt}:{ex_id}".encode("utf-8", errors="ignore")
        h = hashlib.sha256(key).digest()
        u = int.from_bytes(h[:8], "big", signed=False) / float(2**64)

        test_frac = max(0.0, min(1.0, self.test_pct / 100.0))
        val_frac = max(0.0, min(1.0, self.val_pct / 100.0))
        if u < test_frac:
            return "test"
        if u < (test_frac + val_frac):
            return "val"
        return "train"

    @staticmethod
    def _load_dataset_streaming(
        *,
        dataset: str,
        dataset_config: str,
        split: str,
        trust_remote_code: bool,
        storage_block_size: int,
        streaming_read_max_retries: int,
        streaming_read_retry_interval_s: float,
    ):
        try:
            import datasets
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets is required for HF streaming. Install with: pip install datasets"
            ) from exc

        # Reduce time spent in HF's internal retry loop. BYOS wraps streaming with an outer restart guard.
        if hasattr(datasets, "config") and hasattr(datasets.config, "STREAMING_READ_MAX_RETRIES"):
            datasets.config.STREAMING_READ_MAX_RETRIES = int(streaming_read_max_retries)
        if hasattr(datasets, "config") and hasattr(datasets.config, "STREAMING_READ_RETRY_INTERVAL"):
            datasets.config.STREAMING_READ_RETRY_INTERVAL = float(streaming_read_retry_interval_s)

        storage_options = {"block_size": int(storage_block_size)}

        # Newer `datasets` exposes `storage_options` directly. Older versions require `DownloadConfig`.
        try:
            return load_dataset(
                dataset,
                dataset_config,
                split=split,
                streaming=True,
                trust_remote_code=trust_remote_code,
                storage_options=storage_options,
            )
        except TypeError:
            try:
                from datasets import DownloadConfig

                dl_cfg = DownloadConfig(storage_options=storage_options)
                return load_dataset(
                    dataset,
                    dataset_config,
                    split=split,
                    streaming=True,
                    trust_remote_code=trust_remote_code,
                    download_config=dl_cfg,
                )
            except Exception as exc:
                raise RuntimeError(
                    "HF streaming failed while configuring storage_options. "
                    "This can happen if your installed `datasets`/`fsspec` stack doesn't support "
                    "storage_options for streaming. Try upgrading: pip install -U datasets fsspec."
                ) from exc

    def _make_stream(self, *, epoch_seed: int, num_shards: int, shard_index: int):
        ds = self._load_dataset_streaming(
            dataset=self.dataset,
            dataset_config=self.dataset_config,
            split=self.split,
            trust_remote_code=self.trust_remote_code,
            storage_block_size=self.storage_block_size,
            streaming_read_max_retries=self.streaming_read_max_retries,
            streaming_read_retry_interval_s=self.streaming_read_retry_interval_s,
        )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=epoch_seed)
        if num_shards > 1:
            shard = getattr(ds, "shard", None)
            if callable(shard):
                ds = ds.shard(num_shards=num_shards, index=shard_index)
        return ds

    def _iter_examples_forever(self, *, base_seed: int, num_shards: int, shard_index: int):
        epoch = 0
        restarts = 0
        while True:
            epoch_seed = base_seed + epoch
            ds = self._make_stream(
                epoch_seed=epoch_seed, num_shards=num_shards, shard_index=shard_index
            )
            it = iter(ds)
            if self.max_items > 0:
                it = itertools.islice(it, self.max_items)
            try:
                yielded = 0
                for ex in it:
                    if self.holdout_role in ("", "all", "any"):
                        bucket_ok = True
                    else:
                        bucket_ok = self._holdout_bucket(ex) == self.holdout_role
                    if not bucket_ok:
                        continue
                    yield ex
                    yielded += 1
                    if self.max_items > 0 and yielded >= self.max_items:
                        break
                epoch += 1
                restarts = 0
            except Exception as exc:
                if not self.restart_on_stream_error:
                    raise
                restarts += 1
                if self.max_restarts > 0 and restarts > self.max_restarts:
                    raise RuntimeError(
                        f"HF streaming exceeded max_restarts={self.max_restarts}"
                    ) from exc

                max_tag = "inf" if self.max_restarts <= 0 else str(self.max_restarts)
                sleep_s = max(self.restart_sleep_s, 0.0)
                print(
                    f"INFO: HF stream error ({type(exc).__name__}) -> restarting in {sleep_s:.1f}s "
                    f"[{restarts}/{max_tag}]"
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
                epoch += 1

    def _extract_text(self, ex) -> str:
        if isinstance(ex, dict):
            if self.text_field in ex:
                return str(ex[self.text_field])
            # Fallback: pick the first string-ish field.
            for v in ex.values():
                if isinstance(v, str):
                    return v
        return str(ex)

    def _encode_doc(self, text: str) -> torch.Tensor:
        max_len = self.max_doc_tokens if self.max_doc_tokens > 0 else -1
        # Keep state/history in pure token id space; we optionally add EOS between docs.
        return self.tokenizer.encode(
            text,
            device=torch.device("cpu"),
            bos=False,
            eos=self.add_eos,
            max_length=max_len,
        ).to(dtype=torch.int32)

    def _ingest_tokens(self, ring: _TokenRingBuffer, ex_iter, *, target_tokens: int) -> int:
        ingested = 0
        last_log = 0
        log_every = 250_000
        while ingested < target_tokens:
            ex = next(ex_iter)
            text = self._extract_text(ex)
            if not text:
                continue
            toks = self._encode_doc(text)
            if toks.numel() == 0:
                continue
            ring.append(toks)
            ingested += int(toks.numel())
            if (
                self.holdout_role == "val"
                and target_tokens >= log_every
                and ingested - last_log >= log_every
            ):
                last_log = ingested
                print(
                    f"INFO: HF val prefill: {ingested}/{target_tokens} tokens "
                    f"(buffer={self.token_buffer_size} refresh={self.refresh_tokens})"
                )
        return ingested

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else int(info.id)
        num_workers = 1 if info is None else int(info.num_workers)

        gen = torch.Generator()
        gen.manual_seed(self.seed + worker_id)

        ring = _TokenRingBuffer(self.token_buffer_size, dtype=torch.int32)
        ex_iter = self._iter_examples_forever(
            base_seed=self.seed + 17 * worker_id,
            num_shards=num_workers,
            shard_index=worker_id,
        )

        # Prefill buffer to enable random span sampling.
        if self.holdout_role == "val":
            print(
                "INFO: HF val sampler prefill start: "
                f"prefill_tokens={self.prefill_tokens} token_buffer_size={self.token_buffer_size} "
                f"shuffle_buffer={self.shuffle_buffer} storage_block_size={self.storage_block_size}"
            )
        self._ingest_tokens(ring, ex_iter, target_tokens=self.prefill_tokens)
        if self.holdout_role == "val":
            print("INFO: HF val sampler prefill done")

        while True:
            h_len = sample_h_len(self.h_len_cfg, generator=gen)
            span = int(h_len + self.n_local + 1)
            if span > self.token_buffer_size:
                raise ValueError(
                    f"requested span {span} exceeds hf_token_buffer_size={self.token_buffer_size}"
                )

            while len(ring) < max(span, self.min_span * self.batch_size):
                self._ingest_tokens(ring, ex_iter, target_tokens=self.refresh_tokens)

            h_ids = []
            x_ids = []
            y_ids = []
            for _ in range(self.batch_size):
                span_ids = ring.sample_span(span, generator=gen)
                if span_ids is None:
                    self._ingest_tokens(ring, ex_iter, target_tokens=self.refresh_tokens)
                    span_ids = ring.sample_span(span, generator=gen)
                if span_ids is None:
                    raise RuntimeError("failed to sample span from token ring buffer")
                h_ids.append(span_ids[:h_len].to(dtype=torch.long))
                x_ids.append(span_ids[h_len : h_len + self.n_local].to(dtype=torch.long))
                y_ids.append(
                    span_ids[h_len + 1 : h_len + 1 + self.n_local].to(dtype=torch.long)
                )

            # Advance the buffer to avoid repeatedly resampling the same window.
            if self.refresh_tokens > 0:
                self._ingest_tokens(ring, ex_iter, target_tokens=self.refresh_tokens)

            yield (
                torch.stack(h_ids, dim=0),
                torch.stack(x_ids, dim=0),
                torch.stack(y_ids, dim=0),
                h_len,
            )


def build_dataloaders_from_hf_streaming(
    *,
    dataset: str,
    dataset_config: str,
    tokenizer,
    batch_size: int,
    n_local: int,
    h_len_cfg,
    seed: int,
    train_seed: int | None = None,
    val_seed: int | None = None,
    num_workers: int,
    pin_memory: bool,
    train_split: str = "train",
    val_split: str = "validation",
    trust_remote_code: bool = False,
    text_field: str = "text",
    storage_block_size: int = 0,
    streaming_read_max_retries: int = 20,
    streaming_read_retry_interval_s: float = 5.0,
    holdout_mode: str = "none",
    holdout_salt: str = "",
    holdout_id_field: str = "id",
    val_pct: float = 0.0,
    test_pct: float = 0.0,
    restart_on_stream_error: bool = True,
    restart_sleep_s: float = 5.0,
    max_restarts: int = 0,
    shuffle_buffer: int = 0,
    val_shuffle_buffer: int = 0,
    token_buffer_size: int = 2_000_000,
    prefill_tokens: int = 500_000,
    refresh_tokens: int = 50_000,
    max_doc_tokens: int = 0,
    add_eos: bool = True,
    max_train_items: int = 0,
    max_val_items: int = 0,
):
    # Backward-compat: callers can pass a single `seed` and we derive split seeds.
    if train_seed is None:
        train_seed = int(seed)
    if val_seed is None:
        val_seed = int(seed) + 1000

    holdout_mode = str(holdout_mode or "none").strip().lower()
    if holdout_mode not in ("", "none", "off", "false", "0", "hash_id", "hash", "id_hash"):
        raise ValueError(f"unsupported holdout_mode: {holdout_mode}")

    if holdout_mode in ("hash_id", "hash", "id_hash"):
        if not str(holdout_salt or ""):
            raise ValueError("holdout_salt must be set when holdout_mode=hash_id")
        if val_pct < 0 or test_pct < 0:
            raise ValueError("val_pct and test_pct must be >= 0")
        if (val_pct + test_pct) >= 100.0:
            raise ValueError("val_pct + test_pct must be < 100")
        # In holdout mode, val/test are derived from the train split.
        resolved_val_split = train_split
    else:
        # Probe val split early so we can fallback cleanly.
        resolved_val_split = val_split
        try:
            HFStreamingTokenBufferSampler._load_dataset_streaming(
                dataset=dataset,
                dataset_config=dataset_config,
                split=val_split,
                trust_remote_code=trust_remote_code,
                storage_block_size=storage_block_size,
                streaming_read_max_retries=streaming_read_max_retries,
                streaming_read_retry_interval_s=streaming_read_retry_interval_s,
            )
        except Exception:
            print("INFO: NO VAL -> Autofallback to train split for validation")
            resolved_val_split = train_split

    train_ds = HFStreamingTokenBufferSampler(
        dataset=dataset,
        dataset_config=dataset_config,
        split=train_split,
        tokenizer=tokenizer,
        text_field=text_field,
        batch_size=batch_size,
        n_local=n_local,
        h_len_cfg=h_len_cfg,
        seed=train_seed,
        trust_remote_code=trust_remote_code,
        storage_block_size=storage_block_size,
        streaming_read_max_retries=streaming_read_max_retries,
        streaming_read_retry_interval_s=streaming_read_retry_interval_s,
        holdout_mode=holdout_mode,
        holdout_role="train",
        holdout_salt=holdout_salt,
        holdout_id_field=holdout_id_field,
        val_pct=val_pct,
        test_pct=test_pct,
        restart_on_stream_error=restart_on_stream_error,
        restart_sleep_s=restart_sleep_s,
        max_restarts=max_restarts,
        shuffle_buffer=shuffle_buffer,
        token_buffer_size=token_buffer_size,
        prefill_tokens=prefill_tokens,
        refresh_tokens=refresh_tokens,
        max_doc_tokens=max_doc_tokens,
        add_eos=add_eos,
        max_items=max_train_items,
    )
    val_ds = HFStreamingTokenBufferSampler(
        dataset=dataset,
        dataset_config=dataset_config,
        split=resolved_val_split,
        tokenizer=tokenizer,
        text_field=text_field,
        batch_size=batch_size,
        n_local=n_local,
        h_len_cfg=h_len_cfg,
        seed=val_seed,
        trust_remote_code=trust_remote_code,
        storage_block_size=storage_block_size,
        streaming_read_max_retries=streaming_read_max_retries,
        streaming_read_retry_interval_s=streaming_read_retry_interval_s,
        holdout_mode=holdout_mode,
        holdout_role="val" if holdout_mode in ("hash_id", "hash", "id_hash") else "all",
        holdout_salt=holdout_salt,
        holdout_id_field=holdout_id_field,
        val_pct=val_pct,
        test_pct=test_pct,
        restart_on_stream_error=restart_on_stream_error,
        restart_sleep_s=restart_sleep_s,
        max_restarts=max_restarts,
        shuffle_buffer=val_shuffle_buffer,
        token_buffer_size=token_buffer_size,
        prefill_tokens=prefill_tokens,
        refresh_tokens=refresh_tokens,
        max_doc_tokens=max_doc_tokens,
        add_eos=add_eos,
        max_items=max_val_items,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def build_dataloader_from_hf_streaming_role(
    *,
    dataset: str,
    dataset_config: str,
    split: str,
    tokenizer,
    batch_size: int,
    n_local: int,
    h_len_cfg,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    trust_remote_code: bool = False,
    text_field: str = "text",
    storage_block_size: int = 0,
    streaming_read_max_retries: int = 20,
    streaming_read_retry_interval_s: float = 5.0,
    holdout_mode: str = "none",
    holdout_role: str = "all",
    holdout_salt: str = "",
    holdout_id_field: str = "id",
    val_pct: float = 0.0,
    test_pct: float = 0.0,
    restart_on_stream_error: bool = True,
    restart_sleep_s: float = 5.0,
    max_restarts: int = 0,
    shuffle_buffer: int = 0,
    token_buffer_size: int = 2_000_000,
    prefill_tokens: int = 500_000,
    refresh_tokens: int = 50_000,
    max_doc_tokens: int = 0,
    add_eos: bool = True,
    max_items: int = 0,
):
    ds = HFStreamingTokenBufferSampler(
        dataset=dataset,
        dataset_config=dataset_config,
        split=split,
        tokenizer=tokenizer,
        text_field=text_field,
        batch_size=batch_size,
        n_local=n_local,
        h_len_cfg=h_len_cfg,
        seed=seed,
        trust_remote_code=trust_remote_code,
        storage_block_size=storage_block_size,
        streaming_read_max_retries=streaming_read_max_retries,
        streaming_read_retry_interval_s=streaming_read_retry_interval_s,
        holdout_mode=holdout_mode,
        holdout_role=holdout_role,
        holdout_salt=holdout_salt,
        holdout_id_field=holdout_id_field,
        val_pct=val_pct,
        test_pct=test_pct,
        restart_on_stream_error=restart_on_stream_error,
        restart_sleep_s=restart_sleep_s,
        max_restarts=max_restarts,
        shuffle_buffer=shuffle_buffer,
        token_buffer_size=token_buffer_size,
        prefill_tokens=prefill_tokens,
        refresh_tokens=refresh_tokens,
        max_doc_tokens=max_doc_tokens,
        add_eos=add_eos,
        max_items=max_items,
    )
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
