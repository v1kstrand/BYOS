# Copyright Lightning AI. Licensed under the Apache License 2.0.
#
# Minimal RoPE utilities vendored from LitGPT to keep BYOS standalone.

from __future__ import annotations

from typing import Optional, Tuple

import torch


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
    rope_local_base_freq: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        factor = extra_config["factor"]
        if "original_max_seq_len" in extra_config:
            orig_context_len = extra_config["original_max_seq_len"]
            low_freq_factor = extra_config["low_freq_factor"]
            high_freq_factor = extra_config["high_freq_factor"]

            wavelen = 2 * torch.pi / theta
            ratio = orig_context_len / wavelen
            smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

            adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
            theta = adjusted_theta
        else:
            theta = theta / factor

    seq_idx = torch.arange(seq_len, device=device).float() / condense_ratio
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]

    if rope_local_base_freq is not None:
        local_theta = 1.0 / (
            rope_local_base_freq ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem)
        )
        local_idx_theta = torch.outer(seq_idx, local_theta).repeat(1, 2)
        if local_idx_theta.shape[-1] > n_elem > 1:
            local_idx_theta = local_idx_theta[..., :n_elem]
        idx_theta = torch.stack((idx_theta, local_idx_theta), dim=-1)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.dim() != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")
    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]
    x2 = x[..., head_size_half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)

