import argparse
import random
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

from BYOS.tokenizer import Tokenizer

from BYOS.byos_data import resolve_tokenizer_config
from BYOS.byos_prototype import BYOSv1


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return data


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def select_prompt_tokens(
    test_dir: Path,
    tokenizer: Tokenizer,
    *,
    prompt_len: int,
    gen_len: int,
    seed: int,
    file_path: Path | None = None,
) -> tuple[torch.Tensor, Path, int]:
    rng = random.Random(seed)
    if file_path is not None:
        candidates = [file_path]
    else:
        candidates = sorted(test_dir.glob("*.txt"))
        rng.shuffle(candidates)

    for path in candidates:
        text = path.read_text(encoding="utf-8")
        tokens = tokenizer.encode(text, device=torch.device("cpu")).to(dtype=torch.long)
        total = tokens.numel()
        need = prompt_len + gen_len + 1
        if total < need:
            continue
        start = rng.randint(0, total - need)
        prompt_ids = tokens[start : start + prompt_len]
        return prompt_ids, path, start

    raise RuntimeError("no file has enough tokens for prompt_len + gen_len")


def run_inference(
    model: BYOSv1,
    *,
    prompt_ids: torch.Tensor,
    n_local: int,
    gen_len: int,
    device,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if prompt_ids.numel() < n_local:
        raise ValueError("prompt_len must be >= n_local for teacher-forced prefill")

    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    l_ctx_ids = prompt_ids[:n_local].unsqueeze(0)
    s = model.predictor.s0_template.to(dtype=model.tok_emb.weight.dtype).expand(1, -1, -1)
    kv_cache = None
    pop_idx = 0
    pos_offset = 0

    # teacher-forced prefill for prompt tokens beyond the local window
    for idx in tqdm(range(n_local, prompt_ids.numel()), desc="prefill", dynamic_ncols=True):
        l_ctx_ids, s, _, _, kv_cache, pop_idx, pos_offset = model.fwd_inference(
            l_ctx_ids,
            s,
            kv_cache=kv_cache,
            pop_idx=pop_idx,
            pos_offset=pos_offset,
            use_state=True,
        )
        prev_idx = (pop_idx - 1) % n_local
        l_ctx_ids[:, prev_idx] = prompt_ids[idx]

    # free generation
    generated = []
    for _ in tqdm(range(gen_len), desc="generate", dynamic_ncols=True):
        l_ctx_ids, s, _, logits, kv_cache, pop_idx, pos_offset = model.fwd_inference(
            l_ctx_ids,
            s,
            kv_cache=kv_cache,
            pop_idx=pop_idx,
            pos_offset=pos_offset,
            use_state=True,
        )
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k=top_k, dim=-1)
            probs = torch.softmax(vals, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_token = idx.gather(1, sampled[:, None]).squeeze(1)
        elif temperature != 1.0:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = logits.argmax(dim=-1)
        insert_idx = (pop_idx - 1) % n_local
        l_ctx_ids[:, insert_idx] = next_token
        generated.append(next_token)

    return torch.stack(generated, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-len", type=int, default=0)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--routing-temp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--cfg", type=str, default="")
    args = parser.parse_args()

    cfg_path = Path(args.cfg) if args.cfg else Path(__file__).with_name("cfg_byos.yaml")
    cfg = load_config(cfg_path)
    checkpoint = load_checkpoint(Path(args.checkpoint))
    ckpt_cfg = checkpoint.get("cfg")
    if isinstance(ckpt_cfg, dict):
        cfg = ckpt_cfg

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(device_cfg))

    tok_cfg = resolve_tokenizer_config(cfg.get("tokenizer"))
    tokenizer = Tokenizer(tok_cfg["checkpoint_dir"])

    n_local = int(cfg.get("n_local", 128))
    prompt_len = args.prompt_len if args.prompt_len > 0 else max(n_local * 2, 256)

    dataset_dir = cfg.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("dataset_dir must be set for PG-19 evaluation")
    test_dir = Path(dataset_dir) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test split not found: {test_dir}")

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    file_path = Path(args.file_path) if args.file_path else None
    prompt_ids, source_path, start_idx = select_prompt_tokens(
        test_dir,
        tokenizer,
        prompt_len=prompt_len,
        gen_len=args.gen_len,
        seed=seed,
        file_path=file_path,
    )
    print(f"Source: {source_path}")
    print(f"Seed: {seed}")
    print(f"Start idx: {start_idx}")
    print(
        f"Prompt len: {prompt_ids.numel()} | Gen len: {args.gen_len} | n_local: {n_local} | routing_temp: {args.routing_temp}"
    )

    model = BYOSv1(
        predictor=None,
        vocab_size=tokenizer.vocab_size,
        d_model=int(cfg.get("d_model", 256)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 4)),
        n_state=int(cfg.get("n_state", 16)),
        n_local=n_local,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model.predictor.routing_temp = float(args.routing_temp)

    with torch.no_grad():
        gen_ids = run_inference(
            model,
            prompt_ids=prompt_ids,
            n_local=n_local,
        gen_len=args.gen_len,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    prompt_text = tokenizer.decode(prompt_ids)
    gen_text = tokenizer.decode(gen_ids.squeeze(0))
    print("---- Prompt ----")
    print(prompt_text)
    print("---- Generation ----")
    print(gen_text)


if __name__ == "__main__":
    main()
