import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from BYOS.rope import apply_rope, build_rope_cache
from BYOS.tokenizer import Tokenizer

"""
BYOS v0.5 prototype (single-script, minimal baseline)

This is a solo-dev prototype, not production code.
"""


class PredictorCrossAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        d_model: int,
        n_heads: int,
        attn_drop: float = 0.0,
        alpha_state: float = 0.1,
        rope_base: int = 10000,
    ):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_state = int(n_state)
        self.prototypes = nn.Parameter(torch.randn(n_state, d_model))
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.tok_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = attn_drop
        self.routing_temp = 1.0
        self.alpha_state = float(alpha_state)
        self.rope_base = int(rope_base)
        self.register_buffer("s0_template", torch.zeros(1, n_state, d_model))
        self._rope_cache = None
        self._rope_cache_len = 0
        self._rope_cache_device = None

    def rope_cos_sin_for_h(self, h_len, device):
        max_pos = int(h_len) + 1
        if (
            self._rope_cache is None
            or max_pos > self._rope_cache_len
            or self._rope_cache_device != device
        ):
            cos, sin = build_rope_cache(
                max_pos,
                n_elem=self.head_dim,
                device=device,
                base=self.rope_base,
            )
            self._rope_cache = (cos, sin)
            self._rope_cache_len = max_pos
            self._rope_cache_device = device
        cos, sin = self._rope_cache
        positions = torch.arange(h_len, 0, -1, device=device)
        cos = cos.index_select(0, positions).unsqueeze(0)
        sin = -sin.index_select(0, positions).unsqueeze(0)
        return cos, sin

    def forward(self, H):
        # token -> slot routing with unrolled EMA write simulation
        B, T, D = H.shape
        P = self.prototypes
        q = self.tok_proj(H).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = P.view(self.n_state, self.n_heads, self.head_dim).permute(1, 0, 2)
        k = k.unsqueeze(0).expand(B, -1, -1, -1)

        score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        score = score / max(self.routing_temp, 1e-8)
        w = torch.softmax(score, dim=-1)  # [B, H, T, M]

        a = float(self.alpha_state)
        g = 1.0 - a * w
        suffix_inclusive = torch.cumprod(g.flip(2), dim=2).flip(2)
        suffix_exclusive = torch.cat(
            [
                suffix_inclusive[:, :, 1:, :],
                torch.ones(B, self.n_heads, 1, w.size(-1), device=w.device, dtype=w.dtype),
            ],
            dim=2,
        )
        w_eff = a * w * suffix_exclusive

        h_per_head = H.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        s0 = self.s0_template.to(dtype=H.dtype).expand(B, -1, -1)
        s0 = s0.view(B, self.n_state, self.n_heads, self.head_dim).transpose(1, 2)
        residual = suffix_inclusive[:, :, 0, :].unsqueeze(-1)
        s = s0 * residual + torch.einsum("bhtm,bhtd->bhmd", w_eff, h_per_head)
        s = s.transpose(1, 2).reshape(B, self.n_state, D)
        # Return raw state in token-embedding space. Projection happens at the transformer boundary.
        return s
    
    @torch.no_grad()
    def fwd_inference(self, x, S):
        """
        q: [B, 1, D] shallow feature
        k: [B, Tk, D] learned prototypes
        S: [B, Tk, D] curr state (a mix of shallow features)
        """
        B, Tq, D = x.shape # B, 1, D
        P = self.prototypes.unsqueeze(0).expand(B, -1, -1)
        Tk = P.shape[1]
        if S.shape[1] != Tk:
            raise ValueError("S must have the same slot length as prototypes")
        q_x = self.tok_proj(x).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k_P = P.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v_x = x.view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        v_S = S.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        
        score = torch.matmul(q_x, k_P.transpose(-1, -2)) / self.head_dim**0.5
        score = score / max(self.routing_temp, 1e-8)
        w = F.softmax(score, dim=-1) # w: [B, H, 1, Tk] since Tq=1
        w3 = w.squeeze(2).unsqueeze(-1)                # [B, H, Tk, 1]
        s = v_S                               # [B, H, Tk, Hd]
        v_write = v_x.squeeze(2).unsqueeze(2) # [B, H, 1, Hd] (broadcast over Tk)

        a = float(self.alpha_state)
        s = s * (1.0 - a * w3) + (a * w3) * v_write  # -> [B, H, Tk, Hd]
        s = s.transpose(1, 2).reshape(B, Tk, D)
        return s

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_drop: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = attn_drop

    def forward(self, x, cos=None, sin=None, state_len=0):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if cos is not None:
            q_s = q[:, :, :state_len, :]
            q_x = q[:, :, state_len:, :]
            k_s = k[:, :, :state_len, :]
            k_x = k[:, :, state_len:, :]
            q_x = apply_rope(q_x, cos, sin)
            k_x = apply_rope(k_x, cos, sin)
            q = torch.cat([q_s, q_x], dim=2)
            k = torch.cat([k_s, k_x], dim=2)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_drop if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn)

    def forward_inference(
        self,
        x,
        k_cache=None,
        v_cache=None,
        cache_start=0,
        cache_len=None,
        cos=None,
        sin=None,
        state_len=0,
    ):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if cos is not None:
            q_s = q[:, :, :state_len, :]
            q_x = q[:, :, state_len:, :]
            k_s = k[:, :, :state_len, :]
            k_x = k[:, :, state_len:, :]
            q_x = apply_rope(q_x, cos, sin)
            k_x = apply_rope(k_x, cos, sin)
            q = torch.cat([q_s, q_x], dim=2)
            k = torch.cat([k_s, k_x], dim=2)

        if k_cache is None:
            k_all = k
            v_all = v
        else:
            k_all = torch.cat([k_cache, k], dim=2)
            v_all = torch.cat([v_cache, v], dim=2)

        use_causal = k_cache is None and T > 1
        attn = F.scaled_dot_product_attention(
            q, k_all, v_all, is_causal=use_causal, dropout_p=self.attn_drop if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(B, T, D)

        if cache_start < T:
            k_new = k[:, :, cache_start:, :]
            v_new = v[:, :, cache_start:, :]
            if k_cache is None:
                k_cache = k_new
                v_cache = v_new
            else:
                k_cache = torch.cat([k_cache, k_new], dim=2)
                v_cache = torch.cat([v_cache, v_new], dim=2)
            if cache_len is not None and k_cache.shape[2] > cache_len:
                k_cache = k_cache[:, :, -cache_len:, :].contiguous()
                v_cache = v_cache[:, :, -cache_len:, :].contiguous()

        return self.out_proj(attn), k_cache, v_cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, attn_drop: float, ffn_drop: float, state_len: int):
        super().__init__()
        self.state_len = state_len
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, attn_drop=attn_drop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(ffn_drop),
        )

    def forward(self, x, cos=None, sin=None):
        # x: [B, M+N, D]
        attn_out = self.attn(self.ln1(x), cos=cos, sin=sin, state_len=self.state_len)
        if self.state_len > 0:
            attn_out[:, : self.state_len, :] = 0
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_inference(self, s, x, kv_cache=None, cache_len=None, cos=None, sin=None):
        z = torch.cat([s, x], dim=1)
        z_ln1 = self.ln1(z)
        if kv_cache is None:
            k_cache = None
            v_cache = None
        else:
            k_cache = kv_cache.get("k")
            v_cache = kv_cache.get("v")
        attn_out, k_cache, v_cache = self.attn.forward_inference(
            z_ln1,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_start=self.state_len,
            cache_len=cache_len,
            cos=cos,
            sin=sin,
            state_len=self.state_len,
        )
        if self.state_len > 0:
            attn_out[:, : self.state_len, :] = 0
        z = z + attn_out
        z = z + self.mlp(self.ln2(z))
        # This is the transformed *state prefix* for this step. It is not the persistent BYOS memory.
        s_prefix_out = z[:, : self.state_len, :]
        x_out = z[:, self.state_len :, :]
        return s_prefix_out, x_out, {"k": k_cache, "v": v_cache}

    def forward_inference_local(self, x, kv_cache=None, cache_len=None, cos=None, sin=None):
        z_ln1 = self.ln1(x)
        if kv_cache is None:
            k_cache = None
            v_cache = None
        else:
            k_cache = kv_cache.get("k")
            v_cache = kv_cache.get("v")
        attn_out, k_cache, v_cache = self.attn.forward_inference(
            z_ln1,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_start=0,
            cache_len=cache_len,
            cos=cos,
            sin=sin,
            state_len=0,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, {"k": k_cache, "v": v_cache}


class BYOSv1(nn.Module):
    def __init__(
        self,
        predictor: PredictorCrossAttention | None = None,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_state: int,
        n_local: int,
        attn_drop: float = 0.0,
        ffn_drop: float = 0.0,
        mlp_ratio: int = 4,
        label_smoothing: float = 0.0,
        
    ):
        super().__init__()        
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.n_state = int(n_state)
        self.n_local = int(n_local)
        self.label_smoothing = float(label_smoothing)
        self.rope_base = 10000
        self.rope_n_elem = self.d_model // self.n_heads
        self._rope_cache = None
        self._rope_cache_len = 0
        self._rope_cache_device = None

        if predictor is None:
            predictor = PredictorCrossAttention(
                n_state=n_state,
                d_model=d_model,
                n_heads=n_heads,
                attn_drop=attn_drop,
            )
        self.predictor = predictor  # keep it here to include it in the state dict
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    ffn_drop=ffn_drop,
                    state_len=self.n_state,
                )
                for _ in range(self.n_layers)
            ]
        )

    def run_stack(self, s0, x, cos=None, sin=None):
        s0 = self.predictor.out_proj(s0)
        z = torch.cat([s0, x], dim=1)
        for blk in self.blocks:
            z = blk(z, cos=cos, sin=sin)
        s = z[:, : self.n_state, :]
        x_out = z[:, self.n_state :, :]
        return s, x_out

    def run_stack_inference(self, s0, x, kv_cache=None, cos=None, sin=None):
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        s = self.predictor.out_proj(s0)
        x_in = x
        next_cache = []
        for blk, blk_cache in zip(self.blocks, kv_cache):
            s, x_in, blk_cache = blk.forward_inference(
                s,
                x_in,
                kv_cache=blk_cache,
                cache_len=max(self.n_local - 1, 0),
                cos=cos,
                sin=sin,
            )
            next_cache.append(blk_cache)
        return s, x_in, next_cache

    def run_stack_inference_local(self, x, kv_cache=None, cos=None, sin=None):
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        x_in = x
        next_cache = []
        for blk, blk_cache in zip(self.blocks, kv_cache):
            x_in, blk_cache = blk.forward_inference_local(
                x_in,
                kv_cache=blk_cache,
                cache_len=max(self.n_local - 1, 0),
                cos=cos,
                sin=sin,
            )
            next_cache.append(blk_cache)
        return x_in, next_cache

    def rope_cos_sin(self, positions, device, *, max_pos: int | None = None):
        if max_pos is None:
            max_pos = int(positions.max().item()) + 1
        if (
            self._rope_cache is None
            or max_pos > self._rope_cache_len
            or self._rope_cache_device != device
        ):
            cos, sin = build_rope_cache(
                max_pos,
                n_elem=self.rope_n_elem,
                device=device,
                base=self.rope_base,
            )
            self._rope_cache = (cos, sin)
            self._rope_cache_len = max_pos
            self._rope_cache_device = device
        cos, sin = self._rope_cache
        cos = cos.index_select(0, positions).unsqueeze(0)
        sin = sin.index_select(0, positions).unsqueeze(0)
        return cos, sin

    def ntp_loss(self, x_out, y_ids):
        logits = self.lm_head(x_out)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            y_ids.reshape(-1),
            label_smoothing=self.label_smoothing,
        )

    def build_s0(self, h_ids):
        if self.n_state == 0:
            # Vanilla GPT mode: ignore history entirely and skip predictor (it would softmax over 0 slots).
            bsz = int(h_ids.size(0))
            return self.tok_emb.weight.new_zeros((bsz, 0, self.d_model))
        if h_ids.numel() == 0 or h_ids.shape[1] == 0:
            return self.predictor.s0_template.to(dtype=self.tok_emb.weight.dtype).expand(
                h_ids.size(0), -1, -1
            )
        s0 = self.predictor(self.tok_emb(h_ids))
        return s0.to(dtype=self.tok_emb.weight.dtype)

    def forward_with_s0(self, x_ids, s0, y_ids, cos, sin):
        """Training forward that keeps predictor / variable-length history outside torch.compile.

        Callers are expected to compute `s0 = build_s0(h_ids)` eagerly and pass fixed-shape
        RoPE caches for the local window.
        """
        x = self.tok_emb(x_ids)
        _, x_out = self.run_stack(s0, x, cos=cos, sin=sin)
        return self.ntp_loss(x_out, y_ids)

    def forward(self, x_ids, h_ids, y_ids):
        # x_ids: [B, N], h_ids: [B, Hlen], y_ids: [B, N]
        s0 = self.build_s0(h_ids)
        positions = torch.arange(x_ids.shape[1], device=x_ids.device)
        cos, sin = self.rope_cos_sin(positions, device=x_ids.device, max_pos=x_ids.shape[1])
        return self.forward_with_s0(x_ids, s0, y_ids, cos, sin)
    
    #@torch.compile(backend="inductor", fullgraph=True, dynamic=False)
    def fwd_inference(self, l_ctx_ids, s, kv_cache=None, pop_idx=0, pos_offset=0, use_state=True):
        # `use_state` is derived from the model config; callers can pass it, but it won't override `n_state`.
        use_state = self.n_state > 0
        n_local = l_ctx_ids.shape[1]
        if pop_idx < 0 or pop_idx >= n_local:
            raise ValueError("pop_idx must be in [0, n_local)")

        pop_token = l_ctx_ids[:, pop_idx]
        device = l_ctx_ids.device
        if kv_cache is None:
            if pop_idx == 0:
                l_ctx_for_model = l_ctx_ids
            else:
                l_ctx_for_model = torch.cat(
                    [l_ctx_ids[:, pop_idx:], l_ctx_ids[:, :pop_idx]], dim=1
                )
            x = self.tok_emb(l_ctx_for_model)
            positions = pos_offset + torch.arange(n_local, device=device)
        else:
            last_idx = (pop_idx - 1) % n_local
            x = self.tok_emb(l_ctx_ids[:, last_idx]).unsqueeze(1)
            positions = torch.tensor([pos_offset + n_local - 1], device=device)

        max_pos = pos_offset + n_local
        cos, sin = self.rope_cos_sin(positions, device=device, max_pos=max_pos)
        if use_state:
            _, x_out, kv_cache = self.run_stack_inference(
                s, x, kv_cache=kv_cache, cos=cos, sin=sin
            )
        else:
            x_out, kv_cache = self.run_stack_inference_local(
                x, kv_cache=kv_cache, cos=cos, sin=sin
            )

        logits = self.lm_head(x_out)
        next_token = logits[:, -1, :].argmax(dim=-1)

        if use_state:
            v_write = self.tok_emb(pop_token).unsqueeze(1)
            s_next = self.predictor.fwd_inference(v_write, s)
        else:
            s_next = s

        l_ctx_next = l_ctx_ids.clone()
        l_ctx_next[:, pop_idx] = next_token
        pop_idx = (pop_idx + 1) % n_local
        pos_offset += 1
        return l_ctx_next, s_next, next_token, logits, kv_cache, pop_idx, pos_offset


def build_batch(tokens, *, batch_size: int, h_len: int, n_local: int, device):
    tokens = tokens.to(device=device, dtype=torch.long)
    total = tokens.numel()
    span = h_len + n_local + 1
    if total <= span:
        raise ValueError("not enough tokens for the requested h_len and n_local")
    max_start = total - span
    starts = torch.randint(0, max_start, (batch_size,), device=device)

    h_ids = []
    x_ids = []
    y_ids = []
    for s in starts.tolist():
        s = int(s)
        h_ids.append(tokens[s : s + h_len])
        x_ids.append(tokens[s + h_len : s + h_len + n_local])
        y_ids.append(tokens[s + h_len + 1 : s + h_len + 1 + n_local])

    return (
        torch.stack(h_ids, dim=0),
        torch.stack(x_ids, dim=0),
        torch.stack(y_ids, dim=0),
    )


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--text-path", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--h-len", type=int, default=128)
    parser.add_argument("--n-local", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-state", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = Tokenizer(args.checkpoint_dir)
    text = load_text(args.text_path)
    tokens = tokenizer.encode(text, device=device)

    h_ids, x_ids, y_ids = build_batch(
        tokens,
        batch_size=args.batch_size,
        h_len=args.h_len,
        n_local=args.n_local,
        device=device,
    )
    
    model = BYOSv1(
        predictor=None,
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_state=args.n_state,
        n_local=args.n_local,
    ).to(device)
    
    compiled_fwd = None
    positions = torch.arange(args.n_local, device=device)
    cos, sin = model.rope_cos_sin(positions, device=device, max_pos=args.n_local)
    if device.type == "cuda":
        compiled_fwd = torch.compile(
            model.forward_with_s0,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        dummy_x = torch.zeros((args.batch_size, args.n_local), device=device, dtype=torch.long)
        dummy_y = torch.zeros((args.batch_size, args.n_local), device=device, dtype=torch.long)
        dummy_s0 = model.tok_emb.weight.new_zeros((args.batch_size, args.n_state, args.d_model))
        _ = compiled_fwd(dummy_x, dummy_s0, dummy_y, cos, sin)

    if args.steps > 0:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for step in range(args.steps):
            h_ids, x_ids, y_ids = build_batch(
                tokens,
                batch_size=args.batch_size,
                h_len=args.h_len,
                n_local=args.n_local,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            if compiled_fwd is None:
                loss = model(x_ids, h_ids, y_ids)
            else:
                s0 = model.build_s0(h_ids)
                loss = compiled_fwd(x_ids, s0, y_ids, cos, sin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % args.log_every == 0 or step == args.steps - 1:
                print(f"step {step:04d} | loss={loss.item():.4f}")
    else:
        model.eval()
        with torch.no_grad():
            if compiled_fwd is None:
                loss = model(x_ids, h_ids, y_ids)
            else:
                s0 = model.build_s0(h_ids)
                loss = compiled_fwd(x_ids, s0, y_ids, cos, sin)
        print("demo loss:", float(loss))


if __name__ == "__main__":
    main()
