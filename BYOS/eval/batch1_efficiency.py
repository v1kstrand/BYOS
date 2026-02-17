import argparse
from pathlib import Path

from BYOS.eval.common import mean_std, write_json


def run_efficiency(raw_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    modules = []
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    for p in sorted(raw_dir.glob("*.json")):
        try:
            import json

            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        mod = str(obj.get("module", p.stem))
        wall = float(obj.get("wall_time_s", 0.0))
        tps = float(obj.get("tokens_per_s", 0.0))
        vram = float(obj.get("peak_vram_gb", 0.0))
        modules.append(
            {
                "module": mod,
                "wall_time_s": wall,
                "tokens_per_s": tps,
                "peak_vram_gb": vram,
                "source": str(p),
            }
        )

    return {
        "module": "efficiency",
        "num_entries": len(modules),
        "entries": modules,
        "summary": {
            "wall_time_s": mean_std([m["wall_time_s"] for m in modules]),
            "tokens_per_s": mean_std([m["tokens_per_s"] for m in modules if m["tokens_per_s"] > 0]),
            "peak_vram_gb": mean_std([m["peak_vram_gb"] for m in modules]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    result = run_efficiency(args.raw_dir)
    print(
        f"batch1_efficiency entries={result['num_entries']} "
        f"avg_wall={result['summary']['wall_time_s']['mean']:.2f}s"
    )
    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
