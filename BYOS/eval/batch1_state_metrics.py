import argparse

import torch
from tqdm.auto import tqdm

from BYOS.byos_monitor import attn_s_mass, inference_monitor, state_update_stats
from BYOS.eval.common import (
    build_hf_loader_for_role,
    build_model_and_tokenizer,
    load_cfg_with_checkpoint,
    mean_std,
    resolve_device,
    timed_cuda_metrics,
    write_json,
)


def _prompt_from_batch(batch, prompt_len: int, n_local: int) -> torch.Tensor | None:
    h_ids, x_ids, _y_ids, _h_len = batch
    if h_ids.ndim != 2 or x_ids.ndim != 2:
        return None
    prompt = torch.cat([h_ids[0].to(dtype=torch.long), x_ids[0].to(dtype=torch.long)], dim=0)
    if prompt.numel() < n_local:
        return None
    if prompt_len > 0 and prompt.numel() > prompt_len:
        prompt = prompt[-prompt_len:]
    return prompt


def run_state_metrics(
    cfg_path: str,
    checkpoint: str,
    *,
    role: str,
    num_batches: int,
    inference_prompts: int,
    prompt_len: int,
    gen_len: int,
    seed: int,
    device_override: str = "auto",
) -> dict:
    cfg, ckpt = load_cfg_with_checkpoint(cfg_path, checkpoint)
    device_cfg = device_override if device_override != "auto" else str(cfg.get("device", "auto"))
    device = resolve_device(device_cfg)
    model, tokenizer = build_model_and_tokenizer(cfg, ckpt, device)

    loader = build_hf_loader_for_role(cfg, tokenizer, role=role, seed=seed)
    n_local = int(cfg.get("n_local", 128))

    batch_attn_mass = []
    batch_entropy = []
    batch_topk = []
    inf_attn = []
    inf_entropy = []
    inf_topk = []
    inf_delta = []
    prompt_pool = []

    with torch.inference_mode():
        with timed_cuda_metrics(device):
            for idx, batch in enumerate(tqdm(loader, total=num_batches, desc="batch1-state", dynamic_ncols=True)):
                if idx >= num_batches:
                    break
                m_attn = attn_s_mass(model, batch, device=device)
                m_state = state_update_stats(model, batch, device=device, topk=1)
                batch_attn_mass.append(float(m_attn["attn_s_mass_mean"]))
                batch_entropy.append(float(m_state["routing_entropy"]))
                batch_topk.append(float(m_state["routing_topk_mass"]))

                p = _prompt_from_batch(batch, prompt_len=prompt_len, n_local=n_local)
                if p is not None and len(prompt_pool) < inference_prompts:
                    prompt_pool.append(p)

            for prompt in prompt_pool:
                series = inference_monitor(
                    model,
                    prompt_ids=prompt,
                    n_local=n_local,
                    gen_len=gen_len,
                    device=device,
                    topk=1,
                    layers=None,
                )
                if not series:
                    continue
                inf_attn.append(sum(m["attn_s_mass"] for m in series) / len(series))
                inf_entropy.append(sum(m["routing_entropy"] for m in series) / len(series))
                inf_topk.append(sum(m["routing_topk_mass"] for m in series) / len(series))
                inf_delta.append(sum(m["delta_s_norm"] for m in series) / len(series))

    return {
        "module": "state_metrics",
        "role": str(role),
        "seed": int(seed),
        "num_batches": int(num_batches),
        "inference_prompts": int(len(prompt_pool)),
        "batch_metrics": {
            "attn_s_mass": mean_std(batch_attn_mass),
            "routing_entropy": mean_std(batch_entropy),
            "routing_topk_mass": mean_std(batch_topk),
        },
        "inference_metrics": {
            "attn_s_mass": mean_std(inf_attn),
            "routing_entropy": mean_std(inf_entropy),
            "routing_topk_mass": mean_std(inf_topk),
            "delta_s_norm": mean_std(inf_delta),
        },
        "wall_time_s": float(timed_cuda_metrics.last["wall_time_s"]),
        "peak_vram_gb": float(timed_cuda_metrics.last["peak_vram_gb"]),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--role", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--inference-prompts", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    result = run_state_metrics(
        args.cfg,
        args.checkpoint,
        role=args.role,
        num_batches=int(args.num_batches),
        inference_prompts=int(args.inference_prompts),
        prompt_len=int(args.prompt_len),
        gen_len=int(args.gen_len),
        seed=int(args.seed),
        device_override=str(args.device),
    )

    bm = result["batch_metrics"]
    im = result["inference_metrics"]
    print(
        f"batch1_state role={result['role']} batches={result['num_batches']} prompts={result['inference_prompts']} "
        f"batch_attn={bm['attn_s_mass']['mean']:.4f} inf_delta={im['delta_s_norm']['mean']:.4f}"
    )
    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
