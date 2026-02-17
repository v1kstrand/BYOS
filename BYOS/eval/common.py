import csv
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

from BYOS.byos_data import build_dataloader_from_hf_streaming_role, parse_h_len_cfg, resolve_tokenizer_config
from BYOS.byos_prototype import BYOSv1
from BYOS.tokenizer import Tokenizer


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return data


def load_checkpoint(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return torch.load(p, map_location="cpu")


def load_cfg_with_checkpoint(cfg_path: str | Path, checkpoint_path: str | Path) -> tuple[dict, dict]:
    cfg = load_yaml(cfg_path)
    ckpt = load_checkpoint(checkpoint_path)
    ckpt_cfg = ckpt.get("cfg")
    if isinstance(ckpt_cfg, dict):
        cfg = ckpt_cfg
    return cfg, ckpt


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_cfg))


def resolve_amp_dtype(cfg: dict, device: torch.device):
    if device.type != "cuda":
        return None
    amp_dtype_str = str(cfg.get("amp_dtype", "bf16")).strip().lower()
    if amp_dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if amp_dtype_str in ("fp16", "float16"):
        return torch.float16
    return None


def build_model_and_tokenizer(cfg: dict, ckpt: dict, device: torch.device) -> tuple[BYOSv1, Tokenizer]:
    tok_cfg = resolve_tokenizer_config(cfg.get("tokenizer"))
    tokenizer = Tokenizer(tok_cfg["checkpoint_dir"])
    model = BYOSv1(
        predictor=None,
        vocab_size=tokenizer.vocab_size,
        d_model=int(cfg.get("d_model", 256)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 4)),
        n_state=int(cfg.get("n_state", 16)),
        n_local=int(cfg.get("n_local", 128)),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.predictor.routing_temp = float(cfg.get("routing_temp", 1.0))
    return model, tokenizer


def build_hf_loader_for_role(cfg: dict, tokenizer: Tokenizer, role: str, seed: int):
    if not bool(cfg.get("hf_streaming", False)):
        raise ValueError("Batch-1 eval currently supports hf_streaming runs only")

    role = str(role).strip().lower()
    train_split = str(cfg.get("train_split", "train"))
    val_split = str(cfg.get("val_split", "validation"))
    test_split = str(cfg.get("test_split", "test"))

    if role == "train":
        split = train_split
    elif role == "val":
        split = val_split
    elif role == "test":
        split = test_split
    else:
        raise ValueError(f"unsupported role: {role}")

    holdout_mode = str(cfg.get("hf_holdout_mode", "none")).strip().lower()
    holdout_role = role if holdout_mode not in ("", "none", "off", "false", "0") else "all"
    shuffle_buffer = int(cfg.get("hf_shuffle_buffer", 0)) if role == "train" else int(cfg.get("hf_val_shuffle_buffer", 0))

    return build_dataloader_from_hf_streaming_role(
        dataset=str(cfg.get("dataset", "allenai/dolma")),
        dataset_config=str(cfg.get("dataset_config", "v1_6")),
        split=split,
        tokenizer=tokenizer,
        batch_size=int(cfg.get("batch_size", 4)),
        n_local=int(cfg.get("n_local", 128)),
        h_len_cfg=parse_h_len_cfg(cfg.get("h_len", 128)),
        seed=int(seed),
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        trust_remote_code=bool(cfg.get("hf_trust_remote_code", False)),
        text_field=str(cfg.get("hf_text_field", "text")),
        storage_block_size=int(cfg.get("hf_storage_block_size", 0)),
        streaming_read_max_retries=int(cfg.get("hf_streaming_read_max_retries", 20)),
        streaming_read_retry_interval_s=float(cfg.get("hf_streaming_read_retry_interval_s", 5.0)),
        holdout_mode=holdout_mode,
        holdout_role=holdout_role,
        holdout_salt=str(cfg.get("hf_holdout_salt", "")),
        holdout_id_field=str(cfg.get("hf_holdout_id_field", "id")),
        val_pct=float(cfg.get("hf_val_pct", 0.0)),
        test_pct=float(cfg.get("hf_test_pct", 0.0)),
        restart_on_stream_error=bool(cfg.get("hf_restart_on_stream_error", True)),
        restart_sleep_s=float(cfg.get("hf_restart_sleep_s", 5.0)),
        max_restarts=int(cfg.get("hf_max_restarts", 0)),
        shuffle_buffer=shuffle_buffer,
        token_buffer_size=int(cfg.get("hf_token_buffer_size", 2_000_000)),
        prefill_tokens=int(cfg.get("hf_prefill_tokens", 500_000)),
        refresh_tokens=int(cfg.get("hf_refresh_tokens", 50_000)),
        max_doc_tokens=int(cfg.get("hf_max_doc_tokens", 0)),
        add_eos=bool(cfg.get("hf_add_eos", True)),
        max_items=0,
    )


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv_rows(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) * (x - mean) for x in values) / n
    return {"mean": float(mean), "std": float(math.sqrt(var)), "count": int(n)}


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    step_logits = logits / max(float(temperature), 1e-8)
    if top_k > 0:
        k = min(int(top_k), step_logits.size(-1))
        vals, idx = torch.topk(step_logits, k=k, dim=-1)
        probs = torch.softmax(vals, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
        return idx.gather(1, sampled[:, None]).squeeze(1)
    if float(temperature) != 1.0:
        probs = torch.softmax(step_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
    return step_logits.argmax(dim=-1)


def fwd_inference_mode(model: BYOSv1, l_ctx_ids: torch.Tensor, s: torch.Tensor, *, kv_cache=None, pop_idx: int = 0, pos_offset: int = 0, mode: str = "normal"):
    mode = str(mode).strip().lower()
    if mode not in ("normal", "state_read_off", "state_write_off", "both_off"):
        raise ValueError(f"unsupported mode: {mode}")
    state_enabled = model.n_state > 0
    read_state = state_enabled and mode in ("normal", "state_write_off")
    write_state = state_enabled and mode in ("normal", "state_read_off")

    n_local = l_ctx_ids.shape[1]
    if pop_idx < 0 or pop_idx >= n_local:
        raise ValueError("pop_idx must be in [0, n_local)")

    pop_token = l_ctx_ids[:, pop_idx]
    device = l_ctx_ids.device

    if kv_cache is None:
        if pop_idx == 0:
            l_ctx_for_model = l_ctx_ids
        else:
            l_ctx_for_model = torch.cat([l_ctx_ids[:, pop_idx:], l_ctx_ids[:, :pop_idx]], dim=1)
        x = model.tok_emb(l_ctx_for_model)
        positions = pos_offset + torch.arange(n_local, device=device)
    else:
        last_idx = (pop_idx - 1) % n_local
        x = model.tok_emb(l_ctx_ids[:, last_idx]).unsqueeze(1)
        positions = torch.tensor([pos_offset + n_local - 1], device=device)

    max_pos = pos_offset + n_local
    cos, sin = model.rope_cos_sin(positions, device=device, max_pos=max_pos)

    if read_state:
        _, x_out, kv_cache = model.run_stack_inference(s, x, kv_cache=kv_cache, cos=cos, sin=sin)
    else:
        x_out, kv_cache = model.run_stack_inference_local(x, kv_cache=kv_cache, cos=cos, sin=sin)

    logits = model.lm_head(x_out)
    next_token = logits[:, -1, :].argmax(dim=-1)

    if write_state:
        v_write = model.tok_emb(pop_token).unsqueeze(1)
        s_next = model.predictor.fwd_inference(v_write, s)
    else:
        s_next = s

    l_ctx_next = l_ctx_ids.clone()
    l_ctx_next[:, pop_idx] = next_token
    pop_idx = (pop_idx + 1) % n_local
    pos_offset += 1
    return l_ctx_next, s_next, next_token, logits, kv_cache, pop_idx, pos_offset


@torch.no_grad()
def generate_with_mode(
    model: BYOSv1,
    prompt_ids: torch.Tensor,
    *,
    n_local: int,
    gen_len: int,
    device: torch.device,
    mode: str,
    temperature: float = 1.0,
    top_k: int = 0,
):
    if prompt_ids.numel() < n_local:
        raise ValueError("prompt_len must be >= n_local")

    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    l_ctx_ids = prompt_ids[:n_local].unsqueeze(0)
    s = model.predictor.s0_template.to(dtype=model.tok_emb.weight.dtype).expand(1, -1, -1)
    kv_cache = None
    pop_idx = 0
    pos_offset = 0

    for idx in range(n_local, prompt_ids.numel()):
        l_ctx_ids, s, _, _, kv_cache, pop_idx, pos_offset = fwd_inference_mode(
            model,
            l_ctx_ids,
            s,
            kv_cache=kv_cache,
            pop_idx=pop_idx,
            pos_offset=pos_offset,
            mode=mode,
        )
        prev_idx = (pop_idx - 1) % n_local
        l_ctx_ids[:, prev_idx] = prompt_ids[idx]

    generated = []
    for _ in range(int(gen_len)):
        l_ctx_ids, s, _, logits, kv_cache, pop_idx, pos_offset = fwd_inference_mode(
            model,
            l_ctx_ids,
            s,
            kv_cache=kv_cache,
            pop_idx=pop_idx,
            pos_offset=pos_offset,
            mode=mode,
        )
        next_token = sample_next_token(logits[:, -1, :], temperature=float(temperature), top_k=int(top_k))
        insert_idx = (pop_idx - 1) % n_local
        l_ctx_ids[:, insert_idx] = next_token
        generated.append(next_token)

    if not generated:
        return torch.empty((1, 0), device=device, dtype=torch.long)
    return torch.stack(generated, dim=1)


@contextmanager
def timed_cuda_metrics(device: torch.device):
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        peak_gb = 0.0
        if device.type == "cuda":
            peak_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        timed_cuda_metrics.last = {"wall_time_s": float(elapsed), "peak_vram_gb": peak_gb}


timed_cuda_metrics.last = {"wall_time_s": 0.0, "peak_vram_gb": 0.0}
