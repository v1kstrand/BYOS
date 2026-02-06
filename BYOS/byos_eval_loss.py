import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from BYOS.byos_data import (
    build_dataloader_from_hf_streaming_role,
    resolve_tokenizer_config,
)
from BYOS.tokenizer import Tokenizer
from BYOS.byos_prototype import BYOSv1
from BYOS.byos_train import load_config as _load_config  # reuse YAML parsing behavior


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--role", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    cfg = _load_config(cfg_path)

    ckpt = load_checkpoint(Path(args.checkpoint))
    ckpt_cfg = ckpt.get("cfg")
    if isinstance(ckpt_cfg, dict):
        # Use checkpoint cfg as baseline so eval matches the training run.
        cfg = ckpt_cfg

    device_cfg = args.device if args.device != "auto" else str(cfg.get("device", "auto"))
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    amp_dtype_str = str(cfg.get("amp_dtype", "bf16")).strip().lower()
    if device.type == "cuda" and amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    elif device.type == "cuda" and amp_dtype_str in ("fp16", "float16"):
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else torch.no_grad()
    )

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

    if not bool(cfg.get("hf_streaming", False)):
        raise ValueError("byos_eval_loss currently supports hf_streaming runs only")

    dataset = str(cfg.get("dataset", "allenai/dolma"))
    dataset_config = str(cfg.get("dataset_config", "v1_6"))
    split = str(cfg.get("train_split", "train"))
    text_field = str(cfg.get("hf_text_field", "text"))
    eval_shuffle_buffer = int(
        cfg.get("hf_shuffle_buffer", 0) if args.role == "train" else cfg.get("hf_val_shuffle_buffer", 0)
    )

    loader = build_dataloader_from_hf_streaming_role(
        dataset=dataset,
        dataset_config=dataset_config,
        split=split,
        tokenizer=tokenizer,
        batch_size=int(cfg.get("batch_size", 4)),
        n_local=int(cfg.get("n_local", 128)),
        h_len_cfg=cfg.get("h_len", 128),
        seed=int(cfg.get("seed", 0)) + (0 if args.role == "train" else 1000),
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        trust_remote_code=bool(cfg.get("hf_trust_remote_code", False)),
        text_field=text_field,
        storage_block_size=int(cfg.get("hf_storage_block_size", 0)),
        streaming_read_max_retries=int(cfg.get("hf_streaming_read_max_retries", 20)),
        streaming_read_retry_interval_s=float(cfg.get("hf_streaming_read_retry_interval_s", 5.0)),
        holdout_mode=str(cfg.get("hf_holdout_mode", "none")),
        holdout_role=args.role if str(cfg.get("hf_holdout_mode", "none")).strip().lower() != "none" else "all",
        holdout_salt=str(cfg.get("hf_holdout_salt", "")),
        holdout_id_field=str(cfg.get("hf_holdout_id_field", "id")),
        val_pct=float(cfg.get("hf_val_pct", 0.0)),
        test_pct=float(cfg.get("hf_test_pct", 0.0)),
        restart_on_stream_error=bool(cfg.get("hf_restart_on_stream_error", True)),
        restart_sleep_s=float(cfg.get("hf_restart_sleep_s", 5.0)),
        max_restarts=int(cfg.get("hf_max_restarts", 0)),
        shuffle_buffer=eval_shuffle_buffer,
        token_buffer_size=int(cfg.get("hf_token_buffer_size", 2_000_000)),
        prefill_tokens=int(cfg.get("hf_prefill_tokens", 500_000)),
        refresh_tokens=int(cfg.get("hf_refresh_tokens", 50_000)),
        max_doc_tokens=int(cfg.get("hf_max_doc_tokens", 0)),
        add_eos=bool(cfg.get("hf_add_eos", True)),
        max_items=0,
    )

    total = 0.0
    n = int(args.steps)
    with torch.inference_mode():
        for step, batch in enumerate(tqdm(loader, total=n, desc=f"eval-{args.role}", dynamic_ncols=True)):
            if step >= n:
                break
            h_ids, x_ids, y_ids, _h_len = batch
            h_ids = h_ids.to(device=device, non_blocking=True, dtype=torch.long)
            x_ids = x_ids.to(device=device, non_blocking=True, dtype=torch.long)
            y_ids = y_ids.to(device=device, non_blocking=True, dtype=torch.long)
            with amp_ctx:
                loss = model(x_ids, h_ids, y_ids)
            total += float(loss)

    mean_loss = total / max(n, 1)
    print(f"eval_role={args.role} steps={n} mean_loss={mean_loss:.6f}")


if __name__ == "__main__":
    main()
