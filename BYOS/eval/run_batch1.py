import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

from BYOS.eval.common import load_yaml, write_csv_rows, write_json


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _rows_from_summary(summary: dict) -> list[dict]:
    rows = []
    run_id = str(summary.get("run_meta", {}).get("run_id", ""))

    for item in summary.get("loss", []):
        rows.append(
            {
                "run_id": run_id,
                "module": "loss",
                "split": item.get("role", ""),
                "metric_name": "mean_nll",
                "metric_value": item.get("mean_nll", 0.0),
                "delay": "",
                "ablation_mode": "",
                "seed": item.get("seed", ""),
            }
        )
        rows.append(
            {
                "run_id": run_id,
                "module": "loss",
                "split": item.get("role", ""),
                "metric_name": "perplexity",
                "metric_value": item.get("perplexity", 0.0),
                "delay": "",
                "ablation_mode": "",
                "seed": item.get("seed", ""),
            }
        )

    sm = summary.get("state_metrics", {})
    im = sm.get("inference_metrics", {})
    for name in ("attn_s_mass", "routing_entropy", "routing_topk_mass", "delta_s_norm"):
        val = float(im.get(name, {}).get("mean", 0.0))
        rows.append(
            {
                "run_id": run_id,
                "module": "state_metrics",
                "split": sm.get("role", ""),
                "metric_name": name,
                "metric_value": val,
                "delay": "",
                "ablation_mode": "normal",
                "seed": sm.get("seed", ""),
            }
        )

    for mode_out in summary.get("ablation", {}).get("results", []):
        mode = mode_out.get("mode", "")
        for delay, rec in mode_out.get("per_delay", {}).items():
            rows.append(
                {
                    "run_id": run_id,
                    "module": "retention",
                    "split": "",
                    "metric_name": "token_exact",
                    "metric_value": float(rec.get("token_exact", {}).get("mean", 0.0)),
                    "delay": delay,
                    "ablation_mode": mode,
                    "seed": mode_out.get("seed", ""),
                }
            )

    eff = summary.get("efficiency", {})
    for entry in eff.get("entries", []):
        rows.append(
            {
                "run_id": run_id,
                "module": "efficiency",
                "split": "",
                "metric_name": "tokens_per_s",
                "metric_value": float(entry.get("tokens_per_s", 0.0)),
                "delay": "",
                "ablation_mode": entry.get("module", ""),
                "seed": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    eval_cfg = load_yaml(args.cfg)
    batch = dict(eval_cfg.get("batch1", {}))
    holdout_mode = str(eval_cfg.get("hf_holdout_mode", "none")).strip().lower()
    if holdout_mode in ("", "none", "off", "false", "0"):
        print(
            "WARNING: hf_holdout_mode=none. train/val roles may not represent disjoint document splits."
        )

    run_id = dt.datetime.utcnow().strftime("batch1_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("eval_runs") / run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    loss_steps = int(batch.get("loss_steps", 100))
    state_batches = int(batch.get("state_num_batches", 50))
    state_prompts = int(batch.get("state_inference_prompts", 8))
    retention_samples = int(batch.get("retention_samples_per_delay", 200))
    roles = [str(x) for x in batch.get("loss_roles", ["train", "val"])]

    if args.quick:
        loss_steps = min(loss_steps, 10)
        state_batches = min(state_batches, 8)
        state_prompts = min(state_prompts, 2)
        retention_samples = min(retention_samples, 16)

    delays = str(batch.get("retention_delays", "256,1024,2048"))
    modes = str(batch.get("ablation_modes", "normal,state_read_off,state_write_off,both_off"))
    prompt_len = int(batch.get("state_prompt_len", 2048))
    state_gen_len = int(batch.get("state_gen_len", 256))
    temperature = float(batch.get("temperature", 1.0))
    top_k = int(batch.get("top_k", 0))

    loss_outputs = []
    for role in roles:
        out_path = raw_dir / f"loss_{role}.json"
        _run(
            [
                sys.executable,
                "-m",
                "BYOS.eval.batch1_loss",
                "--cfg",
                args.cfg,
                "--checkpoint",
                args.checkpoint,
                "--role",
                role,
                "--steps",
                str(loss_steps),
                "--seed",
                str(args.seed),
                "--device",
                str(args.device),
                "--out",
                str(out_path),
            ]
        )
        import json

        loss_outputs.append(json.loads(out_path.read_text(encoding="utf-8")))

    state_out = raw_dir / "state_metrics.json"
    _run(
        [
            sys.executable,
            "-m",
            "BYOS.eval.batch1_state_metrics",
            "--cfg",
            args.cfg,
            "--checkpoint",
            args.checkpoint,
            "--role",
            str(batch.get("state_role", "val")),
            "--num-batches",
            str(state_batches),
            "--inference-prompts",
            str(state_prompts),
            "--prompt-len",
            str(prompt_len),
            "--gen-len",
            str(state_gen_len),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--out",
            str(state_out),
        ]
    )

    retention_out = raw_dir / "retention_normal.json"
    _run(
        [
            sys.executable,
            "-m",
            "BYOS.eval.batch1_retention_probe",
            "--cfg",
            args.cfg,
            "--checkpoint",
            args.checkpoint,
            "--delays",
            delays,
            "--samples-per-delay",
            str(retention_samples),
            "--seed",
            str(args.seed),
            "--mode",
            "normal",
            "--temperature",
            str(temperature),
            "--top-k",
            str(top_k),
            "--device",
            str(args.device),
            "--out",
            str(retention_out),
        ]
    )

    ablation_out = raw_dir / "ablation.json"
    _run(
        [
            sys.executable,
            "-m",
            "BYOS.eval.batch1_ablation",
            "--cfg",
            args.cfg,
            "--checkpoint",
            args.checkpoint,
            "--modes",
            modes,
            "--delays",
            delays,
            "--samples-per-delay",
            str(retention_samples),
            "--seed",
            str(args.seed),
            "--temperature",
            str(temperature),
            "--top-k",
            str(top_k),
            "--device",
            str(args.device),
            "--out",
            str(ablation_out),
        ]
    )

    efficiency_out = raw_dir / "efficiency.json"
    _run(
        [
            sys.executable,
            "-m",
            "BYOS.eval.batch1_efficiency",
            "--raw-dir",
            str(raw_dir),
            "--out",
            str(efficiency_out),
        ]
    )

    import json

    summary = {
        "run_meta": {
            "run_id": run_id,
            "cfg": str(args.cfg),
            "checkpoint": str(args.checkpoint),
            "seed": int(args.seed),
            "device": str(args.device),
            "quick": bool(args.quick),
        },
        "loss": loss_outputs,
        "state_metrics": json.loads(state_out.read_text(encoding="utf-8")),
        "retention": json.loads(retention_out.read_text(encoding="utf-8")),
        "ablation": json.loads(ablation_out.read_text(encoding="utf-8")),
        "efficiency": json.loads(efficiency_out.read_text(encoding="utf-8")),
    }

    normal_acc = float(summary["retention"]["overall"]["token_exact"]["mean"])
    ab_gap = float(summary["ablation"].get("retention_gap_normal_vs_both_off", 0.0))
    summary["derived"] = {
        "retention_token_exact_normal": normal_acc,
        "retention_gap_vs_both_off": ab_gap,
        "state_delta_s_norm_mean": float(
            summary["state_metrics"].get("inference_metrics", {}).get("delta_s_norm", {}).get("mean", 0.0)
        ),
    }

    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    write_json(summary_json, summary)
    write_csv_rows(summary_csv, _rows_from_summary(summary))

    print(f"batch1_run completed: {summary_json}")
    print(f"batch1_table: {summary_csv}")


if __name__ == "__main__":
    main()
