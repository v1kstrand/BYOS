import argparse
import math

import torch
from tqdm.auto import tqdm

from BYOS.eval.common import (
    build_hf_loader_for_role,
    build_model_and_tokenizer,
    load_cfg_with_checkpoint,
    resolve_amp_dtype,
    resolve_device,
    timed_cuda_metrics,
    write_json,
)


def run_loss_eval(cfg_path: str, checkpoint: str, *, role: str, steps: int, seed: int, device_override: str = "auto") -> dict:
    cfg, ckpt = load_cfg_with_checkpoint(cfg_path, checkpoint)
    device_cfg = device_override if device_override != "auto" else str(cfg.get("device", "auto"))
    device = resolve_device(device_cfg)
    model, tokenizer = build_model_and_tokenizer(cfg, ckpt, device)

    loader = build_hf_loader_for_role(cfg, tokenizer, role=role, seed=seed)
    amp_dtype = resolve_amp_dtype(cfg, device)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (device.type == "cuda" and amp_dtype is not None)
        else torch.no_grad()
    )

    total_nll = 0.0
    n_steps = 0
    n_local = int(cfg.get("n_local", 0))
    batch_size = int(cfg.get("batch_size", 0))

    with torch.inference_mode():
        with timed_cuda_metrics(device):
            for step, batch in enumerate(tqdm(loader, total=steps, desc=f"batch1-loss-{role}", dynamic_ncols=True)):
                if step >= steps:
                    break
                h_ids, x_ids, y_ids, _h_len = batch
                h_ids = h_ids.to(device=device, non_blocking=True, dtype=torch.long)
                x_ids = x_ids.to(device=device, non_blocking=True, dtype=torch.long)
                y_ids = y_ids.to(device=device, non_blocking=True, dtype=torch.long)
                with amp_ctx:
                    loss = model(x_ids, h_ids, y_ids)
                total_nll += float(loss)
                n_steps += 1

    mean_nll = total_nll / max(n_steps, 1)
    ppl = float(math.exp(min(mean_nll, 80.0)))
    elapsed = timed_cuda_metrics.last["wall_time_s"]
    tokens = max(n_steps, 0) * max(batch_size, 0) * max(n_local, 0)
    tps = float(tokens / elapsed) if elapsed > 0 else 0.0

    return {
        "module": "loss",
        "role": str(role),
        "seed": int(seed),
        "steps_requested": int(steps),
        "steps_run": int(n_steps),
        "mean_nll": float(mean_nll),
        "perplexity": float(ppl),
        "tokens_processed": int(tokens),
        "tokens_per_s": float(tps),
        "wall_time_s": float(elapsed),
        "peak_vram_gb": float(timed_cuda_metrics.last["peak_vram_gb"]),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--role", default="val", choices=["train", "val", "test"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    result = run_loss_eval(
        args.cfg,
        args.checkpoint,
        role=args.role,
        steps=int(args.steps),
        seed=int(args.seed),
        device_override=str(args.device),
    )

    print(
        f"batch1_loss role={result['role']} steps={result['steps_run']}/{result['steps_requested']} "
        f"nll={result['mean_nll']:.6f} ppl={result['perplexity']:.3f} "
        f"toks/s={result['tokens_per_s']:.1f}"
    )
    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
