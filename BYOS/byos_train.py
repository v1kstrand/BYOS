import contextlib
import math
import os
import sys
import random
import time
import warnings
from pathlib import Path

import torch
from tqdm.auto import tqdm
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from BYOS.tokenizer import Tokenizer

from BYOS.byos_data import (
    build_dataloaders_from_dir,
    build_dataloaders_from_hf_streaming,
    resolve_tokenizer_config,
)
from BYOS.byos_prototype import BYOSv1


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_hf_text(dataset: str, config: str, split: str, max_items: int) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for HF loading. Install with: pip install datasets"
        ) from exc

    ds = load_dataset(dataset, config, split=split)
    if "text" in ds.column_names:
        texts = ds["text"]
    else:
        col = ds.column_names[0]
        texts = [str(x) for x in ds[col]]
    if max_items is not None:
        texts = texts[: max_items]
    return "\n".join(texts)


def split_tokens(tokens: torch.Tensor, val_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")
    n = tokens.numel()
    split = int(n * (1.0 - val_ratio))
    return tokens[:split], tokens[split:]


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


class TokenBatchIterableDataset(IterableDataset):
    def __init__(self, tokens, *, batch_size: int, n_local: int, h_len_cfg, seed: int):
        super().__init__()
        self.tokens = tokens
        self.batch_size = batch_size
        self.n_local = n_local
        self.h_len_cfg = h_len_cfg
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        seed = self.seed if info is None else self.seed + info.id
        gen = torch.Generator()
        gen.manual_seed(seed)

        tokens = self.tokens
        total = tokens.numel()

        while True:
            h_len = sample_h_len(self.h_len_cfg, generator=gen)
            span = h_len + self.n_local + 1
            if total <= span:
                raise ValueError("not enough tokens for the requested h_len and n_local")
            max_start = total - span
            starts = torch.randint(0, max_start, (self.batch_size,), generator=gen)

            h_ids = []
            x_ids = []
            y_ids = []
            for s in starts.tolist():
                s = int(s)
                h_ids.append(tokens[s : s + h_len])
                x_ids.append(tokens[s + h_len : s + h_len + self.n_local])
                y_ids.append(tokens[s + h_len + 1 : s + h_len + 1 + self.n_local])

            yield (
                torch.stack(h_ids, dim=0),
                torch.stack(x_ids, dim=0),
                torch.stack(y_ids, dim=0),
                h_len,
            )


def make_dataloader(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    n_local: int,
    h_len_cfg,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TokenBatchIterableDataset(
        tokens,
        batch_size=batch_size,
        n_local=n_local,
        h_len_cfg=h_len_cfg,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def configure_optimizer(
    model: torch.nn.Module, *, lr: float, weight_decay: float, fused: bool = False
):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if param.ndim == 1 or name_lower.endswith(".bias") or "norm" in name_lower:
            no_decay.append(param)
        else:
            decay.append(param)
    params = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    try:
        return torch.optim.AdamW(params, lr=lr, fused=fused)
    except TypeError:
        return torch.optim.AdamW(params, lr=lr)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
):
    min_lr = min(min_lr, base_lr)
    if total_steps <= 1:
        total_steps = 1
    warmup_steps = max(0, min(warmup_steps, total_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    warmup_start_lr = 1e-10
    base_lr = optimizer.param_groups[0]["lr"]
    start_factor = warmup_start_lr / max(base_lr, 1e-20)
    start_factor = min(1.0, max(1e-12, start_factor))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=min_lr,
    )
    if warmup_steps == 0:
        return cosine
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def setup_comet(cfg: dict):
    project_name = str(cfg.get("comet_project_name", "")).strip()
    exp_name = str(cfg.get("experiment_name", "")).strip()
    exp_key = cfg.get("comet_experiment_key", None)
    optional = bool(cfg.get("comet_optional", True))

    if not project_name:
        return None
    if not os.getenv("COMET_API_KEY"):
        if optional:
            return None
        raise RuntimeError("COMET_API_KEY is required for Comet logging")

    try:
        import comet_ml
    except ImportError:
        if optional:
            return None
        raise
    
    comet_exp = comet_ml.start(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=project_name,
        experiment_key=exp_key or None,
    )
    if exp_name:
        comet_exp.set_name(exp_name)
    return comet_exp


def save_checkpoint(
    path: Path, model, optimizer, scheduler, scaler, step: int, val_loss: float, cfg: dict
):
    payload = {
        "step": step,
        "val_loss": float(val_loss),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg,
    }
    import shutil

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    bak_path = path.with_name(f"{path.stem}_bak{path.suffix}")

    if path.exists():
        shutil.copy2(path, bak_path)

    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        if bak_path.exists():
            bak_path.unlink(missing_ok=True)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def cleanup_checkpoints(ckpt_dir: Path, model_name: str, keep_last: int) -> None:
    pattern = f"{model_name}_step*.pt"
    checkpoints = []
    for path in ckpt_dir.glob(pattern):
        stem = path.stem
        suffix = stem.split("_step")[-1]
        try:
            step = int(suffix)
        except ValueError:
            continue
        checkpoints.append((step, path))
    checkpoints.sort(key=lambda item: item[0])
    if keep_last <= 0:
        for _, path in checkpoints:
            path.unlink(missing_ok=True)
        return
    if len(checkpoints) <= keep_last:
        return
    for _, path in checkpoints[:-keep_last]:
        path.unlink(missing_ok=True)


@torch.no_grad()
def evaluate(
    model,
    data_iter,
    *,
    steps: int,
    device,
    amp_ctx,
    pin_memory: bool,
    compiled_fwd=None,
    cos=None,
    sin=None,
):
    model.eval()
    losses = []
    for _ in range(steps):
        h_ids, x_ids, y_ids, h_len = next(data_iter)
        h_ids = h_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        x_ids = x_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        y_ids = y_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        with amp_ctx:
            if compiled_fwd is None:
                loss = model(x_ids, h_ids, y_ids)
            else:
                s0 = model.build_s0(h_ids)
                loss = compiled_fwd(x_ids, s0, y_ids, cos, sin)
        losses.append(loss.detach())
    model.train()
    return torch.stack(losses).mean().item()


@torch.no_grad()
def evaluate_fixed(
    model,
    fixed_batches,
    *,
    device,
    amp_ctx,
    pin_memory: bool,
    compiled_fwd=None,
    cos=None,
    sin=None,
):
    model.eval()
    losses = []
    for h_ids, x_ids, y_ids, _h_len in fixed_batches:
        h_ids = h_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        x_ids = x_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        y_ids = y_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        with amp_ctx:
            if compiled_fwd is None:
                loss = model(x_ids, h_ids, y_ids)
            else:
                s0 = model.build_s0(h_ids)
                loss = compiled_fwd(x_ids, s0, y_ids, cos, sin)
        losses.append(loss.detach())
    model.train()
    return torch.stack(losses).mean().item()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return data


def load_config_optional(path: Path) -> dict:
    if path is None or not path.is_file():
        return {}
    return load_config(path)


def load_merged_config(run_path: Path) -> tuple[dict, dict]:
    default_path = Path(__file__).with_name("cfg_byos_default.yaml")
    base_cfg = load_config_optional(default_path)
    run_cfg = load_config(run_path)
    merged_cfg = {**base_cfg, **run_cfg}
    return base_cfg, merged_cfg


def main(exp_cfg_path: str) -> None:
    # PyTorch can emit a deprecation warning internally for `SequentialLR` when switching
    # schedulers around warmup milestones (it passes an explicit epoch internally).
    # This filter keeps logs clean; it does not change training behavior.
    warnings.filterwarnings(
        "ignore",
        message=r".*The epoch parameter in `scheduler\.step\(\)` was not necessary.*",
        category=UserWarning,
    )

    cfg_path = Path(exp_cfg_path)
    base_cfg, cfg = load_merged_config(cfg_path)
    run_cfg = dict(cfg)
    resume_checkpoint = None
    resume_path = cfg.get("resume_path")
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_path not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        ckpt_cfg = resume_checkpoint.get("cfg")
        
        if isinstance(ckpt_cfg, dict):
            cfg = {**base_cfg, **ckpt_cfg}
            for k in (
                "ckpt_dir",
                "dataset",
                "dataset_config",
                "dataset_dir",
                "train_split",
                "val_split",
                "text_path",
                "val_ratio",
                "max_train_items",
                "max_val_items",
                "max_train_files",
                "max_val_files",
                "hf_streaming",
                "hf_trust_remote_code",
                "hf_text_field",
                "hf_storage_block_size",
                "hf_streaming_read_max_retries",
                "hf_streaming_read_retry_interval_s",
                "hf_restart_on_stream_error",
                "hf_restart_sleep_s",
                "hf_max_restarts",
                "hf_shuffle_buffer",
                "hf_val_shuffle_buffer",
                "hf_token_buffer_size",
                "hf_prefill_tokens",
                "hf_refresh_tokens",
                "hf_max_doc_tokens",
                "hf_add_eos",
                "hf_max_train_items",
                "hf_max_val_items",
                "save_best",
                "save_freq",
                "save_n_store",
                "eval_every",
                "eval_steps",
                "eval_fixed",
                "non_stop",
                "log_every",
                "comet_experiment_key",
            ):
                if k in run_cfg:
                    cfg[k] = run_cfg[k]
                    
        cfg["resume_path"] = str(resume_path)

    dataset_dir = cfg.get("dataset_dir")
    text_path = cfg.get("text_path")
    dataset = cfg.get("dataset", "wikitext")
    dataset_config = cfg.get("dataset_config", "wikitext-2-raw-v1")
    train_split = cfg.get("train_split", "train")
    val_split = cfg.get("val_split", "validation")
    max_train_items = int(cfg.get("max_train_items", 1000))
    max_val_items = int(cfg.get("max_val_items", 200))
    val_ratio = float(cfg.get("val_ratio", 0.05))
    batch_size = int(cfg.get("batch_size", 4))
    h_len_cfg = parse_h_len_cfg(cfg.get("h_len", 128))
    n_local = int(cfg.get("n_local", 128))
    d_model = int(cfg.get("d_model", 256))
    n_heads = int(cfg.get("n_heads", 4))
    n_layers = int(cfg.get("n_layers", 4))
    n_state = int(cfg.get("n_state", 16))
    steps = int(cfg.get("steps", 200))
    non_stop = bool(cfg.get("non_stop", False))
    eval_every = int(cfg.get("eval_every", 50))
    eval_steps = int(cfg.get("eval_steps", 10))
    eval_fixed = bool(cfg.get("eval_fixed", True))
    log_every = int(cfg.get("log_every", 10))
    lr = float(cfg.get("lr", 1e-4))
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    attn_drop = float(cfg.get("attn_drop", 0.0))
    ffn_drop = float(cfg.get("ffn_drop", 0.0))
    min_lr = float(cfg.get("min_lr", 1e-6))
    warmup_steps = int(cfg.get("warmup_steps", 100))
    tokens_per_step = int(cfg.get("tokens_per_step", 0))
    routing_temp = float(cfg.get("routing_temp", 1.0))
    weight_decay = float(cfg.get("weight_decay", 0.1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    amp_dtype_str = str(cfg.get("amp_dtype", "bf16")).lower()
    compile_model = bool(cfg.get("compile", False))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", True))
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
    save_freq = int(cfg.get("save_freq", 0))
    save_n_store = int(cfg.get("save_n_store", 0))
    resume_path = cfg.get("resume_path")
    save_best = bool(cfg.get("save_best", True))
    token_cache_dir = cfg.get("token_cache_dir")
    max_train_files = int(cfg.get("max_train_files", 0))
    max_val_files = int(cfg.get("max_val_files", 0))
    seed = int(cfg.get("seed", 67))
    device_cfg = cfg.get("device", "auto")
    comet_project_name = str(cfg.get("comet_project_name", "")).strip()
    experiment_name = str(cfg.get("experiment_name", "")).strip()
    if comet_project_name and experiment_name:
        model_name = f"{comet_project_name}_{experiment_name}"
    else:
        model_name = "byos"

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        
        dynamo_config = torch._dynamo.config
        dynamo_config.compiled_autograd = True
        dynamo_config.capture_scalar_outputs = False
        dynamo_config.cache_size_limit = 512

    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(device_cfg))
    if device.type == "cuda":
        pin_memory = True

    if amp_dtype_str == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_str in ("fp16", "float16"):
        amp_dtype = torch.float16
    elif amp_dtype_str in ("none", "off", "false"):
        amp_dtype = None
    else:
        raise ValueError(f"unsupported amp_dtype: {amp_dtype_str}")

    use_amp = device.type == "cuda" and amp_dtype is not None
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else contextlib.nullcontext()
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    tok_cfg = resolve_tokenizer_config(cfg.get("tokenizer"))
    checkpoint_dir = tok_cfg["checkpoint_dir"]
    token_cache_dtype = tok_cfg["token_cache_dtype"]
    tokenizer = Tokenizer(checkpoint_dir)
    print(
        f"INFO: Tokenizer: {tok_cfg['name']} | checkpoint_dir={checkpoint_dir} | cache_dtype={token_cache_dtype}"
    )

    hf_streaming = bool(cfg.get("hf_streaming", False))
    hf_trust_remote_code = bool(cfg.get("hf_trust_remote_code", False))
    hf_text_field = str(cfg.get("hf_text_field", "text"))
    hf_storage_block_size = int(cfg.get("hf_storage_block_size", 0))
    hf_streaming_read_max_retries = int(cfg.get("hf_streaming_read_max_retries", 20))
    hf_streaming_read_retry_interval_s = float(cfg.get("hf_streaming_read_retry_interval_s", 5.0))
    hf_restart_on_stream_error = bool(cfg.get("hf_restart_on_stream_error", True))
    hf_restart_sleep_s = float(cfg.get("hf_restart_sleep_s", 5.0))
    hf_max_restarts = int(cfg.get("hf_max_restarts", 0))
    hf_shuffle_buffer = int(cfg.get("hf_shuffle_buffer", 0))
    hf_val_shuffle_buffer = int(cfg.get("hf_val_shuffle_buffer", 0))
    hf_token_buffer_size = int(cfg.get("hf_token_buffer_size", 2_000_000))
    hf_prefill_tokens = int(cfg.get("hf_prefill_tokens", 500_000))
    hf_refresh_tokens = int(cfg.get("hf_refresh_tokens", 50_000))
    hf_max_doc_tokens = int(cfg.get("hf_max_doc_tokens", 0))
    hf_add_eos = bool(cfg.get("hf_add_eos", True))
    hf_max_train_items = int(cfg.get("hf_max_train_items", 0))
    hf_max_val_items = int(cfg.get("hf_max_val_items", 0))

    if hf_streaming:
        print(
            f"INFO: HF streaming enabled: dataset={dataset} config={dataset_config} split={train_split} text_field={hf_text_field}"
        )
        if hf_storage_block_size == 0:
            print("INFO: HF streaming storage_options: block_size=0 (disable HTTP range requests)")
        print(
            "INFO: HF streaming reader retry: "
            f"max_retries={hf_streaming_read_max_retries} "
            f"retry_interval_s={max(hf_streaming_read_retry_interval_s, 0.0):.1f}"
        )
        if hf_restart_on_stream_error:
            max_tag = "inf" if hf_max_restarts <= 0 else str(hf_max_restarts)
            print(
                f"INFO: HF streaming restart_on_error enabled: sleep_s={max(hf_restart_sleep_s, 0.0):.1f} max_restarts={max_tag}"
            )
        if not hf_trust_remote_code:
            print(
                "INFO: hf_trust_remote_code is false; script-based datasets (e.g. Dolma) may prompt/hang unless you set it true."
            )
        train_loader, val_loader = build_dataloaders_from_hf_streaming(
            dataset=dataset,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            batch_size=batch_size,
            n_local=n_local,
            h_len_cfg=h_len_cfg,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_split=train_split,
            val_split=val_split,
            trust_remote_code=hf_trust_remote_code,
            text_field=hf_text_field,
            storage_block_size=hf_storage_block_size,
            streaming_read_max_retries=hf_streaming_read_max_retries,
            streaming_read_retry_interval_s=hf_streaming_read_retry_interval_s,
            restart_on_stream_error=hf_restart_on_stream_error,
            restart_sleep_s=hf_restart_sleep_s,
            max_restarts=hf_max_restarts,
            shuffle_buffer=hf_shuffle_buffer,
            val_shuffle_buffer=hf_val_shuffle_buffer,
            token_buffer_size=hf_token_buffer_size,
            prefill_tokens=hf_prefill_tokens,
            refresh_tokens=hf_refresh_tokens,
            max_doc_tokens=hf_max_doc_tokens,
            add_eos=hf_add_eos,
            max_train_items=hf_max_train_items,
            max_val_items=hf_max_val_items,
        )
    elif dataset_dir:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
        train_loader, val_loader = build_dataloaders_from_dir(
            dataset_dir=dataset_dir,
            tokenizer=tokenizer,
            batch_size=batch_size,
            n_local=n_local,
            h_len_cfg=h_len_cfg,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_split=train_split,
            val_split=val_split,
            cache_dir=token_cache_dir,
            cache_dtype=token_cache_dtype,
            max_train_files=max_train_files,
            max_val_files=max_val_files,
        )
    else:
        cpu_device = torch.device("cpu")
        if text_path:
            text = load_text(Path(text_path))
            tokens = tokenizer.encode(text, device=cpu_device).to(dtype=torch.long)
            train_tokens, val_tokens = split_tokens(tokens, val_ratio)
        else:
            train_text = load_hf_text(dataset, dataset_config, train_split, max_train_items)
            val_text = load_hf_text(dataset, dataset_config, val_split, max_val_items)
            train_tokens = tokenizer.encode(train_text, device=cpu_device).to(dtype=torch.long)
            val_tokens = tokenizer.encode(val_text, device=cpu_device).to(dtype=torch.long)

        train_loader = make_dataloader(
            train_tokens,
            batch_size=batch_size,
            n_local=n_local,
            h_len_cfg=h_len_cfg,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = make_dataloader(
            val_tokens,
            batch_size=batch_size,
            n_local=n_local,
            h_len_cfg=h_len_cfg,
            seed=seed + 1000,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    print(
        f"INFO: Training setup: batch_size={batch_size} n_local={n_local} h_len={h_len_cfg} steps={steps}"
    )
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    fixed_eval_batches = None
    if eval_fixed and eval_steps > 0:
        fixed_eval_batches = [next(val_iter) for _ in range(eval_steps)]

    tokens_per_batch = batch_size * n_local
    if tokens_per_step > 0:
        accum_steps = max(1, math.ceil(tokens_per_step / tokens_per_batch))
    else:
        accum_steps = 1

    model = BYOSv1(
        predictor=None,
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_state=n_state,
        n_local=n_local,
        label_smoothing=label_smoothing,
        attn_drop=attn_drop,
        ffn_drop=ffn_drop,
    ).to(device)
    model.predictor.routing_temp = routing_temp
    compiled_fwd = None
    rope_positions = None
    rope_cos = None
    rope_sin = None

    if steps <= 0:
        model.train()
        h_ids, x_ids, y_ids, h_len = next(train_iter)
        h_ids = h_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        x_ids = x_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        y_ids = y_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
        with amp_ctx:
            loss = model(x_ids, h_ids, y_ids)
        print(f"DEMO: demo loss: {loss.item():.4f}")
        return

    optimizer = configure_optimizer(
        model, lr=lr, weight_decay=weight_decay, fused=device.type == "cuda"
    )
    scheduler = build_scheduler(
        optimizer,
        base_lr=lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        total_steps=steps,
    )

    best_val = float("inf")
    start_step = 0
    if resume_path:
        resume_path = Path(resume_path)
        if resume_checkpoint is None:
            resume_checkpoint = torch.load(resume_path, map_location="cpu")
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model"], strict=True)
        model.predictor.routing_temp = routing_temp
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state(optimizer, device)
        if checkpoint.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
                print(
                    "DEBUG_SCHED: loaded",
                    type(scheduler).__name__,
                    "last_epoch=",
                    scheduler.last_epoch,
                )
                if hasattr(scheduler, "T_max"):
                    print("DEBUG_SCHED: T_max=", scheduler.T_max, "eta_min=", scheduler.eta_min)
            except Exception as exc:
                print(f"WARNING: failed to load scheduler state: {exc}")
        if scaler.is_enabled() and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint.get("step", -1)) + 1
        best_val = float(checkpoint.get("val_loss", best_val))
        print(f"INFO: Resumed from {resume_path} at step {start_step}")
        if start_step >= steps and not non_stop:
            print(f"WARNING: resume step {start_step} >= total steps {steps}; nothing to do.")
            return

    # Precompute RoPE caches for the local window once. This keeps compile static and avoids
    # recompiles/graph breaks due to the internal rope cache mutation.
    rope_positions = torch.arange(n_local, device=device)
    rope_cos, rope_sin = model.rope_cos_sin(
        rope_positions, device=device, max_pos=n_local
    )

    if compile_model:
        print("INFO: compiling forward_with_s0 (predictor + variable Hlen stays eager)")
        try:
            compiled_fwd = torch.compile(
                model.forward_with_s0,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                mode="max-autotune-no-cudagraphs",
            )
            """# Hard-fail early if compilation can't handle the fixed-shape path.
            dummy_x = torch.zeros((batch_size, n_local), device=device, dtype=torch.long)
            dummy_y = torch.zeros((batch_size, n_local), device=device, dtype=torch.long)
            dummy_s0 = model.tok_emb.weight.new_zeros((batch_size, n_state, d_model))
            with amp_ctx:
                warmup_loss = compiled_fwd(dummy_x, dummy_s0, dummy_y, rope_cos, rope_sin)
            # Also trigger backward compilation early (hard fail) without taking an optimizer step.
            model.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(warmup_loss).backward()
                scaler.unscale_(optimizer)
            else:
                warmup_loss.backward()
            model.zero_grad(set_to_none=True)"""
        except Exception as exc:
            raise RuntimeError(f"torch.compile failed for BYOS forward_with_s0: {exc}") from exc

    print("DEBUG_LR: resume_path =", resume_path)
    print("DEBUG_LR: start_step  =", start_step)
    print("DEBUG_LR: total_steps =", steps)
    print("DEBUG_LR: non_stop    =", non_stop)
    print("DEBUG_LR: curr_lr     =", optimizer.param_groups[0]["lr"])
    print("DEBUG_LR: sched_last_epoch =", scheduler.last_epoch)
    print("DEBUG_LR: eval_fixed  =", eval_fixed)
    print("DEBUG_LR: eval_steps  =", eval_steps)
    print("DEBUG_LR: log_every   =", log_every)

    comet_exp = setup_comet(cfg)
    if comet_exp is not None:
        exp_key = comet_exp.get_key()
        if exp_key:
            cfg["comet_experiment_key"] = exp_key
        comet_exp.log_parameters(
            {
                "batch_size": batch_size,
                "n_local": n_local,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "n_state": n_state,
                "steps": steps,
                "lr": lr,
                "label_smoothing": label_smoothing,
                "min_lr": min_lr,
                "warmup_steps": warmup_steps,
                "tokens_per_step": tokens_per_step,
                "weight_decay": weight_decay,
                "amp_dtype": amp_dtype_str,
                "accum_steps": accum_steps,
            }
        )
    tokens_per_update = tokens_per_batch * accum_steps

    progress = tqdm(
        total=steps,
        initial=min(start_step, steps),
        desc="train",
        dynamic_ncols=True,
    )
    step = start_step
    try:
        while True:
            if step >= steps and not non_stop:
                break
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            if step < start_step + 2:
                print(
                    f"DEBUG_LR: loop step={step} lr={optimizer.param_groups[0]['lr']:.8f} sched_last_epoch={scheduler.last_epoch}"
                )

            for ac_step in range(accum_steps):
                h_ids, x_ids, y_ids, h_len = next(train_iter)
                h_ids = h_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
                x_ids = x_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)
                y_ids = y_ids.to(device=device, non_blocking=pin_memory, dtype=torch.long)

                with amp_ctx:
                    if compiled_fwd is None:
                        loss = model(x_ids, h_ids, y_ids)
                    else:
                        s0 = model.build_s0(h_ids)
                        loss = compiled_fwd(x_ids, s0, y_ids, rope_cos, rope_sin)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

                total_loss += float(loss.detach())
                loss = loss / accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if step < steps:
                scheduler.step()

            step_time = time.time() - step_start
            tokens_per_sec = tokens_per_update / max(step_time, 1e-8)
            avg_loss = total_loss / accum_steps
            current_lr = optimizer.param_groups[0]["lr"]

            if log_every > 0 and (step % log_every == 0 or step == steps - 1):
                progress.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "grad_norm": f"{float(grad_norm):.2f}",
                    }
                )

            if comet_exp is not None:
                mem_gb = None
                if device.type == "cuda":
                    mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                    torch.cuda.reset_peak_memory_stats()
                comet_exp.log_metrics(
                    {
                        "loss": avg_loss,
                        "lr": current_lr,
                        "tokens_per_sec": tokens_per_sec,
                        "grad_norm": float(grad_norm),
                        "gpu_mem_gb": mem_gb,
                    },
                    step=step,
                )

            if eval_every > 0 and step > 0 and (step % eval_every == 0 or step == steps - 1):
                if eval_fixed and fixed_eval_batches is not None:
                    val_loss = evaluate_fixed(
                        model,
                        fixed_eval_batches,
                        device=device,
                        amp_ctx=amp_ctx,
                        pin_memory=pin_memory,
                        compiled_fwd=compiled_fwd,
                        cos=rope_cos,
                        sin=rope_sin,
                    )
                else:
                    val_loss = evaluate(
                        model,
                        val_iter,
                        steps=eval_steps,
                        device=device,
                        amp_ctx=amp_ctx,
                        pin_memory=pin_memory,
                        compiled_fwd=compiled_fwd,
                        cos=rope_cos,
                        sin=rope_sin,
                    )
                if comet_exp is not None:
                    comet_exp.log_metrics({"val_loss": val_loss}, step=step)

                if val_loss < best_val:
                    best_val = val_loss
                    if save_best:
                        best_path = ckpt_dir / f"{model_name}_best.pt"
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            scheduler,
                            scaler if scaler.is_enabled() else None,
                            step,
                            val_loss,
                            cfg,
                        )
                tqdm.write(f"eval step {step:04d} | val={val_loss:.4f}")

            if save_freq > 0 and step > 0 and (step % save_freq == 0):
                latest_path = ckpt_dir / f"{model_name}_latest.pt"
                if latest_path.exists() and save_n_store > 1:
                    prev_step = step - save_freq
                    if prev_step >= 0:
                        prev_path = ckpt_dir / f"{model_name}_step{prev_step:06d}.pt"
                        latest_path.replace(prev_path)
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler if scaler.is_enabled() else None,
                    step,
                    best_val,
                    cfg,
                )
                cleanup_checkpoints(ckpt_dir, model_name, max(0, save_n_store - 1))

            if step < steps:
                progress.update(1)
            step += 1
    finally:
        progress.close()


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    main(cfg_path)
