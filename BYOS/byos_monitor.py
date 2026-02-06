import math

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from BYOS.rope import apply_rope


def _apply_rope_to_tokens(q, k, cos, sin, state_len):
    if cos is None or sin is None:
        return q, k
    if state_len == 0:
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    q_s = q[:, :, :state_len, :]
    q_x = q[:, :, state_len:, :]
    k_s = k[:, :, :state_len, :]
    k_x = k[:, :, state_len:, :]
    q_x = apply_rope(q_x, cos, sin)
    k_x = apply_rope(k_x, cos, sin)
    q = torch.cat([q_s, q_x], dim=2)
    k = torch.cat([k_s, k_x], dim=2)
    return q, k


def _manual_attention_weights(q, k):
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    t = scores.size(-1)
    mask = torch.triu(torch.ones(t, t, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


def _block_attention_metrics(block, z, cos=None, sin=None):
    state_len = block.state_len
    if state_len == 0:
        return 0.0
    z_ln1 = block.ln1(z)
    qkv = block.attn.qkv(z_ln1)
    q, k, _ = qkv.chunk(3, dim=-1)
    bsz, seq_len, d_model = q.shape
    n_heads = block.attn.n_heads
    head_dim = d_model // n_heads
    q = q.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)

    q, k = _apply_rope_to_tokens(q, k, cos, sin, state_len)
    weights = _manual_attention_weights(q, k)
    token_weights = weights[:, :, state_len:, :]
    mass_s = token_weights[:, :, :, :state_len].sum(dim=-1)
    return float(mass_s.mean().detach())


@torch.no_grad()
def attn_s_mass(model, batch, *, device, layers=None):
    h_ids, x_ids, _, _ = batch
    h_ids = h_ids.to(device=device, dtype=torch.long)
    x_ids = x_ids.to(device=device, dtype=torch.long)
    s0 = model.build_s0(h_ids)
    x_emb = model.tok_emb(x_ids)
    positions = torch.arange(x_emb.shape[1], device=x_emb.device)
    cos, sin = model.rope_cos_sin(
        positions, device=x_emb.device, max_pos=x_emb.shape[1]
    )
    z = torch.cat([s0, x_emb], dim=1)
    if layers is None:
        layers = range(len(model.blocks))
    values = []
    for idx, block in enumerate(model.blocks):
        if idx in layers:
            values.append(_block_attention_metrics(block, z, cos=cos, sin=sin))
        z = block(z, cos=cos, sin=sin)
    return {
        "attn_s_mass_mean": float(sum(values) / len(values)) if values else 0.0,
        "attn_s_mass_layers": values,
    }


@torch.no_grad()
def state_update_stats(model, batch, *, device, topk: int = 1):
    if getattr(model, "n_state", 0) == 0:
        return {
            "routing_entropy": 0.0,
            "routing_topk_mass": 0.0,
        }
    h_ids, x_ids, _, _ = batch
    h_ids = h_ids.to(device=device, dtype=torch.long)
    x_ids = x_ids.to(device=device, dtype=torch.long)
    s0 = model.build_s0(h_ids)
    x_emb = model.tok_emb(x_ids)
    x0 = x_emb[:, :1, :]

    predictor = model.predictor
    bsz, t_q, _ = x0.shape
    p = predictor.prototypes.unsqueeze(0).expand(bsz, -1, -1)
    t_k = p.shape[1]
    q_x = predictor.tok_proj(x0).view(bsz, t_q, predictor.n_heads, predictor.head_dim).transpose(1, 2)
    k_p = p.view(bsz, t_k, predictor.n_heads, predictor.head_dim).transpose(1, 2)
    score = torch.matmul(q_x, k_p.transpose(-1, -2)) / math.sqrt(predictor.head_dim)
    w = torch.softmax(score, dim=-1).mean(dim=1).squeeze(2)

    w_clamped = w.clamp_min(1e-8)
    entropy = -(w_clamped * w_clamped.log()).sum(dim=-1).mean()
    k = min(topk, w.shape[-1])
    topk_mass = torch.topk(w, k=k, dim=-1).values.sum(dim=-1).mean()

    return {
        "routing_entropy": float(entropy.detach()),
        "routing_topk_mass": float(topk_mass.detach()),
    }


def _routing_stats(predictor, x_write, *, topk: int):
    bsz, t_q, _ = x_write.shape
    p = predictor.prototypes.unsqueeze(0).expand(bsz, -1, -1)
    t_k = p.shape[1]
    if t_k == 0:
        return 0.0, 0.0
    q_x = predictor.tok_proj(x_write).view(bsz, t_q, predictor.n_heads, predictor.head_dim).transpose(1, 2)
    k_p = p.view(bsz, t_k, predictor.n_heads, predictor.head_dim).transpose(1, 2)
    score = torch.matmul(q_x, k_p.transpose(-1, -2)) / math.sqrt(predictor.head_dim)
    score = score / max(float(predictor.routing_temp), 1e-8)
    w = torch.softmax(score, dim=-1).mean(dim=1).squeeze(2)

    w_clamped = w.clamp_min(1e-8)
    entropy = -(w_clamped * w_clamped.log()).sum(dim=-1).mean()
    k = min(topk, w.shape[-1])
    topk_mass = torch.topk(w, k=k, dim=-1).values.sum(dim=-1).mean()
    return float(entropy.detach()), float(topk_mass.detach())


def _attn_s_mass_inference(model, s, x_ids, *, device, layers=None, pos_offset: int = 0):
    x_ids = x_ids.to(device=device, dtype=torch.long)
    x_emb = model.tok_emb(x_ids)
    s_proj = model.predictor.out_proj(s)
    positions = pos_offset + torch.arange(x_emb.shape[1], device=x_emb.device)
    cos, sin = model.rope_cos_sin(
        positions, device=x_emb.device, max_pos=int(positions.max().item()) + 1
    )
    z = torch.cat([s_proj, x_emb], dim=1)
    if layers is None:
        layers = range(len(model.blocks))
    values = []
    for idx, block in enumerate(model.blocks):
        if idx in layers:
            values.append(_block_attention_metrics(block, z, cos=cos, sin=sin))
        z = block(z, cos=cos, sin=sin)
    return float(sum(values) / len(values)) if values else 0.0


@torch.no_grad()
def inference_monitor(
    model,
    *,
    prompt_ids: torch.Tensor,
    n_local: int,
    gen_len: int,
    device,
    topk: int = 1,
    layers=None,
):
    if getattr(model, "n_state", 0) == 0:
        raise ValueError("inference_monitor requires n_state > 0 (no routing/state to monitor).")
    if prompt_ids.numel() < n_local:
        raise ValueError("prompt_len must be >= n_local")
    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    l_ctx_ids = prompt_ids[:n_local].unsqueeze(0)
    s = model.predictor.s0_template.to(dtype=model.tok_emb.weight.dtype).expand(1, -1, -1)
    kv_cache = None
    pop_idx = 0
    pos_offset = 0

    for idx in range(n_local, prompt_ids.numel()):
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

    metrics = []
    for _ in range(gen_len):
        pop_token = l_ctx_ids[:, pop_idx]
        x_write = model.tok_emb(pop_token).unsqueeze(1)
        entropy, topk_mass = _routing_stats(model.predictor, x_write, topk=topk)
        l_ctx_ids, s_next, _, _, kv_cache, pop_idx, pos_offset = model.fwd_inference(
            l_ctx_ids,
            s,
            kv_cache=kv_cache,
            pop_idx=pop_idx,
            pos_offset=pos_offset,
            use_state=True,
        )
        delta = (s_next - s).norm(dim=-1).mean()
        base = s.norm(dim=-1).mean().clamp_min(1e-8)
        attn_mass = _attn_s_mass_inference(
            model, s_next, l_ctx_ids, device=device, layers=layers, pos_offset=pos_offset
        )
        metrics.append(
            {
                "attn_s_mass": attn_mass,
                "routing_entropy": entropy,
                "routing_topk_mass": topk_mass,
                "delta_s_norm": float((delta / base).detach()),
            }
        )
        s = s_next
    return metrics


@torch.no_grad()
def inference_aging_monitor(
    model,
    *,
    prompt_ids: torch.Tensor,
    n_local: int,
    gen_len: int,
    device,
    temperature: float = 1.0,
    top_k: int = 0,
    routing_topk: int = 1,
    layers=None,
    window: int | None = None,
    show_progress: bool = True,
    tokenizer=None,
    print_text: bool = False,
):
    if getattr(model, "n_state", 0) == 0:
        raise ValueError("inference_aging_monitor requires n_state > 0 (no routing/state to monitor).")
    if prompt_ids.numel() < n_local:
        raise ValueError("prompt_len must be >= n_local")
    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    l_ctx_ids = prompt_ids[:n_local].unsqueeze(0)
    s = model.predictor.s0_template.to(dtype=model.tok_emb.weight.dtype).expand(1, -1, -1)
    kv_cache = None
    pop_idx = 0
    pos_offset = 0

    prefill_range = range(n_local, prompt_ids.numel())
    if show_progress:
        prefill_range = tqdm(prefill_range, desc="aging prefill", dynamic_ncols=True)
    for idx in prefill_range:
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

    metrics = []
    generated = []
    gen_range = range(gen_len)
    if show_progress:
        gen_range = tqdm(gen_range, desc="aging generate", dynamic_ncols=True)
    for _ in gen_range:
        pop_token = l_ctx_ids[:, pop_idx]
        x_write = model.tok_emb(pop_token).unsqueeze(1)
        entropy, topk_mass = _routing_stats(model.predictor, x_write, topk=routing_topk)
        l_ctx_ids, s_next, _, logits, kv_cache, pop_idx, pos_offset = model.fwd_inference(
            l_ctx_ids,
            s,
            kv_cache=kv_cache,
            pop_idx=pop_idx,
            pos_offset=pos_offset,
            use_state=True,
        )
        step_logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            k = min(top_k, step_logits.size(-1))
            vals, idx = torch.topk(step_logits, k=k, dim=-1)
            probs = torch.softmax(vals, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_token = idx.gather(1, sampled[:, None]).squeeze(1)
        elif temperature != 1.0:
            probs = torch.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = step_logits.argmax(dim=-1)
        insert_idx = (pop_idx - 1) % n_local
        l_ctx_ids[:, insert_idx] = next_token
        generated.append(next_token)

        delta = (s_next - s).norm(dim=-1).mean()
        base = s.norm(dim=-1).mean().clamp_min(1e-8)
        attn_mass = _attn_s_mass_inference(
            model, s_next, l_ctx_ids, device=device, layers=layers, pos_offset=pos_offset
        )
        metrics.append(
            {
                "attn_s_mass": attn_mass,
                "routing_entropy": entropy,
                "routing_topk_mass": topk_mass,
                "delta_s_norm": float((delta / base).detach()),
            }
        )
        s = s_next
    attn_series = [m["attn_s_mass"] for m in metrics]
    window_means = []
    if window is not None and window > 0:
        for start in range(0, len(attn_series), window):
            chunk = attn_series[start : start + window]
            if not chunk:
                continue
            window_means.append(sum(chunk) / len(chunk))
    if print_text and tokenizer is not None:
        prompt_text = tokenizer.decode(prompt_ids)
        gen_text = tokenizer.decode(torch.stack(generated, dim=1).squeeze(0))
        print("---- Prompt ----")
        print(prompt_text)
        print("---- Generation ----")
        print(gen_text)
    return {
        "attn_s_mass_series": attn_series,
        "window_means": window_means,
        "metrics": metrics,
        "generated_ids": generated,
    }


def plot_attn_series(attn_series, window_means=None, *, title: str = "attn_s_mass over time"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(attn_series, label="attn_s_mass")
    if window_means:
        window = max(1, len(attn_series) // len(window_means))
        xs = [window * (idx + 1) - 1 for idx in range(len(window_means))]
        ax.plot(xs, window_means, marker="o", label="window_mean")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("attn_s_mass")
    ax.legend()
    fig.tight_layout()
    return fig, ax
