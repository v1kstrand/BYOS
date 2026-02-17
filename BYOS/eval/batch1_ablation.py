import argparse

from BYOS.eval.batch1_retention_probe import _parse_delays, run_retention_probe
from BYOS.eval.common import mean_std, write_json


def _parse_modes(s: str) -> list[str]:
    vals = [x.strip() for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("modes must not be empty")
    allowed = {"normal", "state_read_off", "state_write_off", "both_off"}
    for v in vals:
        if v not in allowed:
            raise ValueError(f"unsupported mode: {v}")
    return vals


def run_ablation(
    cfg_path: str,
    checkpoint: str,
    *,
    modes: list[str],
    delays: list[int],
    samples_per_delay: int,
    seed: int,
    temperature: float,
    top_k: int,
    device_override: str = "auto",
) -> dict:
    outputs = []
    for mode in modes:
        outputs.append(
            run_retention_probe(
                cfg_path,
                checkpoint,
                delays=delays,
                samples_per_delay=samples_per_delay,
                seed=seed,
                mode=mode,
                temperature=temperature,
                top_k=top_k,
                device_override=device_override,
            )
        )

    by_mode = {o["mode"]: o for o in outputs}
    normal = by_mode.get("normal")
    both_off = by_mode.get("both_off")
    gap = 0.0
    if normal and both_off:
        gap = float(normal["overall"]["token_exact"]["mean"] - both_off["overall"]["token_exact"]["mean"])

    wall_times = [float(o.get("wall_time_s", 0.0)) for o in outputs]
    peak_vram = [float(o.get("peak_vram_gb", 0.0)) for o in outputs]

    return {
        "module": "ablation",
        "seed": int(seed),
        "modes": modes,
        "delays": [int(d) for d in delays],
        "samples_per_delay": int(samples_per_delay),
        "results": outputs,
        "retention_gap_normal_vs_both_off": float(gap),
        "wall_time_s": mean_std(wall_times),
        "peak_vram_gb": mean_std(peak_vram),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modes", type=str, default="normal,state_read_off,state_write_off,both_off")
    parser.add_argument("--delays", type=str, default="256,1024,2048")
    parser.add_argument("--samples-per-delay", type=int, default=200)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    result = run_ablation(
        args.cfg,
        args.checkpoint,
        modes=_parse_modes(args.modes),
        delays=_parse_delays(args.delays),
        samples_per_delay=int(args.samples_per_delay),
        seed=int(args.seed),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        device_override=str(args.device),
    )

    print(
        f"batch1_ablation modes={','.join(result['modes'])} "
        f"gap(normal-both_off)={result['retention_gap_normal_vs_both_off']:.4f}"
    )
    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
