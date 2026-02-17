import argparse
import random

import torch

from BYOS.eval.common import (
    build_model_and_tokenizer,
    generate_with_mode,
    load_cfg_with_checkpoint,
    mean_std,
    normalize_text,
    resolve_device,
    timed_cuda_metrics,
    write_json,
)


def build_probe_cases(tokenizer, delays: list[int], samples_per_delay: int, seed: int, n_local: int) -> list[dict]:
    rng = random.Random(seed)
    values = [
        "amber",
        "cobalt",
        "crimson",
        "ivory",
        "onyx",
        "silver",
        "saffron",
        "teal",
        "violet",
        "scarlet",
    ]
    fillers = [
        " The report continues with routine details.",
        " Additional neutral context follows.",
        " This section contains background narrative.",
    ]

    filler_tokens = tokenizer.encode(rng.choice(fillers), device=torch.device("cpu")).to(dtype=torch.long)
    if filler_tokens.numel() == 0:
        filler_tokens = torch.tensor([1], dtype=torch.long)

    cases = []
    for delay in delays:
        for i in range(samples_per_delay):
            key = f"k{i}"
            value = rng.choice(values)
            prefix = f"Remember this mapping. Key {key} maps to value {value}."
            query = f" After a long passage, key {key} maps to value"

            prefix_ids = tokenizer.encode(prefix, device=torch.device("cpu")).to(dtype=torch.long)
            query_ids = tokenizer.encode(query, device=torch.device("cpu")).to(dtype=torch.long)
            target_ids = tokenizer.encode(" " + value, device=torch.device("cpu")).to(dtype=torch.long)
            if target_ids.numel() == 0:
                continue
            target_id = int(target_ids[0].item())

            filler = filler_tokens.repeat((delay // max(filler_tokens.numel(), 1)) + 2)
            filler = filler[:delay]
            prompt_ids = torch.cat([prefix_ids, filler, query_ids], dim=0)

            if prompt_ids.numel() < n_local:
                pad = filler_tokens.repeat((n_local - prompt_ids.numel()) // max(filler_tokens.numel(), 1) + 2)
                prompt_ids = torch.cat([pad[: n_local - prompt_ids.numel()], prompt_ids], dim=0)

            cases.append(
                {
                    "delay": int(delay),
                    "key": key,
                    "value": value,
                    "prompt_ids": prompt_ids,
                    "target_id": target_id,
                }
            )
    return cases


def run_retention_probe(
    cfg_path: str,
    checkpoint: str,
    *,
    delays: list[int],
    samples_per_delay: int,
    seed: int,
    mode: str,
    temperature: float,
    top_k: int,
    device_override: str = "auto",
) -> dict:
    cfg, ckpt = load_cfg_with_checkpoint(cfg_path, checkpoint)
    device_cfg = device_override if device_override != "auto" else str(cfg.get("device", "auto"))
    device = resolve_device(device_cfg)
    model, tokenizer = build_model_and_tokenizer(cfg, ckpt, device)
    n_local = int(cfg.get("n_local", 128))

    cases = build_probe_cases(tokenizer, delays=delays, samples_per_delay=samples_per_delay, seed=seed, n_local=n_local)
    token_hits_by_delay = {int(d): [] for d in delays}
    text_hits_by_delay = {int(d): [] for d in delays}

    with torch.inference_mode():
        with timed_cuda_metrics(device):
            for case in cases:
                gen = generate_with_mode(
                    model,
                    case["prompt_ids"],
                    n_local=n_local,
                    gen_len=1,
                    device=device,
                    mode=mode,
                    temperature=temperature,
                    top_k=top_k,
                )
                if gen.numel() == 0:
                    continue
                pred_id = int(gen[0, 0].item())
                token_hit = 1.0 if pred_id == int(case["target_id"]) else 0.0
                pred_txt = normalize_text(tokenizer.decode(gen[0]))
                gold_txt = normalize_text(case["value"])
                text_hit = 1.0 if pred_txt.startswith(gold_txt) else 0.0
                token_hits_by_delay[int(case["delay"])].append(token_hit)
                text_hits_by_delay[int(case["delay"])].append(text_hit)

    per_delay = {}
    token_acc_values = []
    text_acc_values = []
    for delay in delays:
        d = int(delay)
        tok = token_hits_by_delay.get(d, [])
        txt = text_hits_by_delay.get(d, [])
        tok_m = mean_std(tok)
        txt_m = mean_std(txt)
        per_delay[str(d)] = {
            "token_exact": tok_m,
            "text_prefix": txt_m,
        }
        token_acc_values.extend(tok)
        text_acc_values.extend(txt)

    return {
        "module": "retention_probe",
        "mode": str(mode),
        "seed": int(seed),
        "delays": [int(d) for d in delays],
        "samples_per_delay": int(samples_per_delay),
        "overall": {
            "token_exact": mean_std(token_acc_values),
            "text_prefix": mean_std(text_acc_values),
        },
        "per_delay": per_delay,
        "wall_time_s": float(timed_cuda_metrics.last["wall_time_s"]),
        "peak_vram_gb": float(timed_cuda_metrics.last["peak_vram_gb"]),
        "device": str(device),
    }


def _parse_delays(s: str) -> list[int]:
    vals = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("delays must not be empty")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--delays", type=str, default="256,1024,2048")
    parser.add_argument("--samples-per-delay", type=int, default=200)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "state_read_off", "state_write_off", "both_off"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    result = run_retention_probe(
        args.cfg,
        args.checkpoint,
        delays=_parse_delays(args.delays),
        samples_per_delay=int(args.samples_per_delay),
        seed=int(args.seed),
        mode=str(args.mode),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        device_override=str(args.device),
    )

    print(
        f"batch1_retention mode={result['mode']} token_acc={result['overall']['token_exact']['mean']:.4f} "
        f"text_acc={result['overall']['text_prefix']['mean']:.4f}"
    )
    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
