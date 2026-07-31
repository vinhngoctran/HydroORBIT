# HydroORBIT — architectural layers.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import ModelOutput


# ---------------------------------------------------------------------------
# Layer Norm (T5-style RMS, no bias, no mean subtraction)
# ---------------------------------------------------------------------------

class HydroORBITLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


ALL_LAYERNORM_LAYERS.append(HydroORBITLayerNorm)


# ---------------------------------------------------------------------------
# Patching & Instance Normalisation
# ---------------------------------------------------------------------------

class Patch(nn.Module):
    """Splits a 1-D time series into non-overlapping patches (left-padded with NaN)."""

    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size   = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length % self.patch_size != 0:
            pad_len = self.patch_size - (length % self.patch_size)
            padding = torch.full((*x.shape[:-1], pad_len), float("nan"),
                                 dtype=x.dtype, device=x.device)
            x = torch.cat((padding, x), dim=-1)
        return x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)


class InstanceNorm(nn.Module):
    """
    Per-series z-score normalisation with optional arcsinh transform (NaN-safe).

    Scale computation uses
        nan_to_num(..., nan=1.0, posinf=1.0, neginf=1.0).clamp(min=eps)
    so that near-zero scales are clamped away and Inf values arising from
    extreme outliers in the input are guarded against, rather than only
    handling the exact-zero case.
    """

    def __init__(
        self,
        eps: float = 1e-5,
        use_arcsinh: bool = False,
        binaryaware: bool = False,
        inverse_sinh_clip: float = 20.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh
        self.binaryaware = binaryaware
        self.inverse_sinh_clip = inverse_sinh_clip

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(
                torch.nanmean(x, dim=-1, keepdim=True), nan=0.0
            )
            scale = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(),
                nan=1.0, posinf=1.0, neginf=1.0,   # guard against Inf from outliers
            ).clamp(min=self.eps)                   # handles zero and near-zero
        else:
            loc, scale = loc_scale

        scaled = (x - loc) / scale
        if self.binaryaware:
            valid = ~torch.isnan(x)
            is_binary = ((x == 0.0) | (x == 1.0) | ~valid).all(dim=-1, keepdim=True)
            scaled = torch.where(is_binary, x, scaled)
        if self.use_arcsinh:
            scaled = torch.arcsinh(scaled)
        return scaled.to(orig), (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        orig = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale
        if self.use_arcsinh:
            x = torch.sinh(torch.clamp(x, -self.inverse_sinh_clip, self.inverse_sinh_clip))
        return (x * scale + loc).to(orig)


class CausalPatchInstanceNorm(InstanceNorm):
    """Patch-aware causal scaler.

    Context tokens are normalized with statistics available at the end of each
    patch, which reduces leakage inside long context windows. Forecast outputs
    still use the final context loc/scale so the public API remains unchanged.
    """

    def __init__(
        self,
        patch_size: int,
        min_obs: int = 8,
        eps: float = 1e-5,
        use_arcsinh: bool = False,
        binaryaware: bool = False,
        inverse_sinh_clip: float = 20.0,
    ) -> None:
        super().__init__(
            eps=eps,
            use_arcsinh=use_arcsinh,
            binaryaware=binaryaware,
            inverse_sinh_clip=inverse_sinh_clip,
        )
        self.patch_size = patch_size
        self.min_obs = min_obs

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is not None:
            return super().forward(x, loc_scale)

        orig = x.dtype
        data = x.to(torch.float32)
        mask = ~torch.isnan(data)
        clean = torch.nan_to_num(data, nan=0.0)

        count = mask.cumsum(dim=-1).clamp_min(1)
        causal_loc = (clean * mask).cumsum(dim=-1) / count
        prev_loc = torch.cat([torch.zeros_like(causal_loc[..., :1]), causal_loc[..., :-1]], dim=-1)
        delta = clean - prev_loc
        m2 = (delta * (clean - causal_loc) * mask).cumsum(dim=-1)
        causal_scale = (m2 / (count - 1).clamp_min(1)).sqrt().clamp(min=self.eps)

        if data.shape[-1] % self.patch_size != 0:
            pad_len = self.patch_size - (data.shape[-1] % self.patch_size)
            causal_loc = torch.cat([causal_loc[..., :1].expand(*causal_loc.shape[:-1], pad_len), causal_loc], dim=-1)
            causal_scale = torch.cat([causal_scale[..., :1].expand(*causal_scale.shape[:-1], pad_len), causal_scale], dim=-1)
            data = torch.cat([torch.full((*data.shape[:-1], pad_len), float("nan"), dtype=data.dtype, device=data.device), data], dim=-1)
            mask = torch.cat([torch.zeros((*mask.shape[:-1], pad_len), dtype=mask.dtype, device=mask.device), mask], dim=-1)

        loc = rearrange(causal_loc, "... (n p) -> ... n p", p=self.patch_size)[..., -1]
        scale = rearrange(causal_scale, "... (n p) -> ... n p", p=self.patch_size)[..., -1]
        loc = loc.repeat_interleave(self.patch_size, dim=-1)[..., -data.shape[-1]:]
        scale = scale.repeat_interleave(self.patch_size, dim=-1)[..., -data.shape[-1]:]

        if self.min_obs > 0:
            observed = mask.cumsum(dim=-1)
            enough = observed >= self.min_obs
            donor_mask = (observed <= self.min_obs) & mask
            denom = donor_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            donor_loc = (torch.nan_to_num(data, nan=0.0) * donor_mask).sum(dim=-1, keepdim=True) / denom
            donor_scale = (((torch.nan_to_num(data, nan=0.0) - donor_loc) * donor_mask).square().sum(dim=-1, keepdim=True) / (denom - 1).clamp_min(1)).sqrt().clamp(min=self.eps)
            loc = torch.where(enough, loc, donor_loc)
            scale = torch.where(enough, scale, donor_scale)

        if self.binaryaware:
            valid = ~torch.isnan(data)
            is_binary = ((data == 0.0) | (data == 1.0) | ~valid).all(dim=-1, keepdim=True)
            loc = torch.where(is_binary, torch.zeros_like(loc), loc)
            scale = torch.where(is_binary, torch.ones_like(scale), scale)

        scaled = (data - loc) / scale
        if self.use_arcsinh:
            scaled = torch.arcsinh(scaled)

        final_loc = causal_loc[..., -1:]
        final_scale = causal_scale[..., -1:]
        scaled = scaled[..., -x.shape[-1]:]
        return scaled.to(orig), (final_loc.to(orig), final_scale.to(orig))


# ---------------------------------------------------------------------------
# Residual Block (input / output patch embeddings)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dropout        = nn.Dropout(dropout_p)
        self.hidden_layer   = nn.Linear(in_dim, h_dim)
        self.act            = ACT2FN[act_fn_name]
        self.output_layer   = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = HydroORBITLayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.output_layer(self.act(self.hidden_layer(x))))
        out = out + self.residual_layer(x)
        return self.layer_norm(out) if self.use_layer_norm else out


# ---------------------------------------------------------------------------
# MLP (T5-style: no norm, no residual — used inside FeedForward)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Linear → activation → dropout → Linear.  GeGLU when is_gated=True."""

    def __init__(self, d_model: int, d_ff: int, act_fn: str, dropout: float,
                 is_gated: bool = False):
        super().__init__()
        self.is_gated = is_gated
        self.wi   = nn.Linear(d_model, d_ff, bias=False)
        if is_gated:
            self.wi_1 = nn.Linear(d_model, d_ff, bias=False)   # gate projection
        self.wo   = nn.Linear(d_ff, d_model, bias=False)
        self.act  = ACT2FN[act_fn]
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.wi(x))
        if self.is_gated:
            h = h * self.wi_1(x)   # GeGLU gate
        return self.wo(self.drop(h))


# ---------------------------------------------------------------------------
# Feed-Forward sublayer (pre-norm + MLP + residual)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, act_fn: str,
                 dropout: float, layer_norm_eps: float, is_gated: bool = False):
        super().__init__()
        self.mlp  = MLP(d_model, d_ff, act_fn, dropout, is_gated=is_gated)
        self.norm = HydroORBITLayerNorm(d_model, eps=layer_norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.drop(self.mlp(self.norm(h)))


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RoPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim  = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        self.inv_freq.to(x.device)
        inv_freq_exp = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_ids_exp  = position_ids[:, None, :].float()
        # MPS does not support float32 autocast; compute on CPU.
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_exp.float() @ pos_ids_exp.float()).transpose(1, 2)
            emb   = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return (q * cos) + (RoPE.rotate_half(q) * sin), (k * cos) + (RoPE.rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

@dataclass
class AttentionOutput(ModelOutput):
    hidden_states: Optional[torch.Tensor] = None
    attn_weights:  Optional[torch.Tensor] = None


class MHA(nn.Module):
    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout: float,
                 rope_theta: float, attn_impl: str, use_rope: bool = True,
                 use_qk_norm: bool = False, use_xpos: bool = False,
                 xpos_scale_base: float = 512.0):
        super().__init__()
        self.d_model     = d_model
        self.kv_proj_dim = d_kv
        self.n_heads     = num_heads
        self.dropout     = dropout
        self.inner_dim   = num_heads * d_kv
        self.attn_impl   = attn_impl
        self.use_qk_norm = use_qk_norm
        self.use_xpos    = use_xpos
        self.xpos_scale_base = xpos_scale_base

        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE(dim=d_kv, base=rope_theta)

    def _eager(self, q, k, v, mask):
        scores  = torch.matmul(q, k.transpose(-2, -1)) + mask
        weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v), weights

    def _sdpa(self, q, k, v, mask):
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,   # no 1/√d_k scaling — matches the eager implementation above
        )
        return out, None

    @staticmethod
    def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype)

    def _apply_xpos(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        pos = position_ids.to(dtype=q.dtype, device=q.device)
        center = pos.mean(dim=-1, keepdim=True)
        scale = torch.exp((pos - center) / max(self.xpos_scale_base, 1.0))
        scale = scale[:, None, :, None].clamp(min=1e-4, max=1e4)
        return q * scale, k / scale

    def forward(
        self,
        hidden_states:   torch.Tensor,
        mask:            torch.Tensor,
        encoder_states:  Optional[torch.Tensor] = None,
        position_ids:    Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        def shape(t):
            return rearrange(t, "b s (h d) -> b h s d", h=self.n_heads, d=self.kv_proj_dim)

        def unshape(t):
            return rearrange(t, "b h s d -> b s (h d)", h=self.n_heads, d=self.kv_proj_dim)

        q   = shape(self.q(hidden_states))
        src = encoder_states if encoder_states is not None else hidden_states
        k, v = shape(self.k(src)), shape(self.v(src))

        if self.use_qk_norm:
            q = self._rms_norm(q)
            k = self._rms_norm(k)

        if self.use_rope and position_ids is not None:
            cos, sin = self.rope_embed(v, position_ids)
            q, k = RoPE.apply_rotary_pos_emb(q, k, cos, sin)
            if self.use_xpos:
                q, k = self._apply_xpos(q, k, position_ids)

        impl = "eager" if output_attentions else self.attn_impl
        if impl == "sdpa":
            out, attn_w = self._sdpa(q, k, v, mask)
        else:
            out, attn_w = self._eager(q, k, v, mask)

        return AttentionOutput(
            hidden_states=self.o(unshape(out)),
            attn_weights=attn_w if output_attentions else None,
        )


# ---------------------------------------------------------------------------
# Attention sublayers
# ---------------------------------------------------------------------------

class TimeSelfAttention(nn.Module):
    """Pre-norm self-attention along the time (patch) axis with RoPE."""

    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout: float,
                 layer_norm_eps: float, rope_theta: float, attn_impl: str,
                 use_qk_norm: bool = False, use_xpos: bool = False,
                 xpos_scale_base: float = 512.0):
        super().__init__()
        self.attn = MHA(d_model, d_kv, num_heads, dropout, rope_theta, attn_impl,
                        use_rope=True, use_qk_norm=use_qk_norm,
                        use_xpos=use_xpos, xpos_scale_base=xpos_scale_base)
        self.norm = HydroORBITLayerNorm(d_model, eps=layer_norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor,
                pos: torch.Tensor, output_attentions: bool = False) -> AttentionOutput:
        out = self.attn(self.norm(h), mask=mask, position_ids=pos,
                        output_attentions=output_attentions)
        return AttentionOutput(hidden_states=h + self.drop(out.hidden_states),
                               attn_weights=out.attn_weights)


class GroupSelfAttention(nn.Module):
    """Pre-norm self-attention along the batch axis — mixes series in the same group."""

    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout: float,
                 layer_norm_eps: float, rope_theta: float, attn_impl: str,
                 use_qk_norm: bool = False):
        super().__init__()
        self.attn = MHA(d_model, d_kv, num_heads, dropout, rope_theta, attn_impl,
                        use_rope=False, use_qk_norm=use_qk_norm)
        self.norm = HydroORBITLayerNorm(d_model, eps=layer_norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor,
                output_attentions: bool = False) -> AttentionOutput:
        # Swap batch ↔ time so MHA operates over the batch (series) dimension.
        h = rearrange(h, "b t d -> t b d")
        out = self.attn(self.norm(h), mask=mask, output_attentions=output_attentions)
        h = h + self.drop(out.hidden_states)
        return AttentionOutput(hidden_states=rearrange(h, "t b d -> b t d"),
                               attn_weights=out.attn_weights)


class VariateMixingMLP(nn.Module):
    """Lightweight variate mixer over same-group series."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, layer_norm_eps: float,
                 act_fn: str):
        super().__init__()
        self.norm = HydroORBITLayerNorm(d_model, eps=layer_norm_eps)
        self.mix = ResidualBlock(d_model, d_ff, d_model, act_fn, dropout)
        self.gate = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        mixed = torch.zeros_like(h)
        for gid in torch.unique(group_ids):
            idx = torch.nonzero(group_ids == gid, as_tuple=True)[0]
            group_h = h.index_select(0, idx)
            pooled = group_h.mean(dim=0, keepdim=True).expand_as(group_h)
            mixed.index_copy_(0, idx, pooled)
        z = self.norm(h + mixed)
        gate = torch.sigmoid(self.gate(z))
        return h + self.drop(gate * self.mix(z))


# ---------------------------------------------------------------------------
# Encoder block & Encoder
# ---------------------------------------------------------------------------

@dataclass
class EncoderBlockOutput(ModelOutput):
    hidden_states:      Optional[torch.Tensor] = None
    time_attn_weights:  Optional[torch.Tensor] = None
    group_attn_weights: Optional[torch.Tensor] = None


class HydroORBITEncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_kv: int, d_ff: int, num_heads: int,
                 dropout: float, layer_norm_eps: float, rope_theta: float,
                 act_fn: str, attn_impl: str, is_gated: bool = False,
                 use_qk_norm: bool = False, use_xpos: bool = False,
                 xpos_scale_base: float = 512.0, use_variate_mixer: bool = False):
        super().__init__()
        self.time_attn  = TimeSelfAttention(d_model, d_kv, num_heads, dropout,
                                            layer_norm_eps, rope_theta, attn_impl,
                                            use_qk_norm=use_qk_norm,
                                            use_xpos=use_xpos,
                                            xpos_scale_base=xpos_scale_base)
        self.group_attn = GroupSelfAttention(d_model, d_kv, num_heads, dropout,
                                             layer_norm_eps, rope_theta, attn_impl,
                                             use_qk_norm=use_qk_norm)
        self.variate_mixer = (
            VariateMixingMLP(d_model, d_ff, dropout, layer_norm_eps, act_fn)
            if use_variate_mixer else None
        )
        self.ff         = FeedForward(d_model, d_ff, act_fn, dropout, layer_norm_eps,
                                      is_gated=is_gated)

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor,
                attn_mask: torch.Tensor, group_time_mask: torch.Tensor,
                group_ids: torch.Tensor,
                output_attentions: bool = False) -> EncoderBlockOutput:
        t_out = self.time_attn(h, attn_mask, position_ids, output_attentions)
        g_out = self.group_attn(t_out.hidden_states, group_time_mask, output_attentions)
        h = g_out.hidden_states
        if self.variate_mixer is not None:
            h = self.variate_mixer(h, group_ids)
        h = self.ff(h)
        return EncoderBlockOutput(hidden_states=h,
                                  time_attn_weights=t_out.attn_weights,
                                  group_attn_weights=g_out.attn_weights)


@dataclass
class EncoderOutput(ModelOutput):
    last_hidden_state:      Optional[torch.Tensor] = None
    all_time_attn_weights:  Optional[tuple]        = None
    all_group_attn_weights: Optional[tuple]        = None


class HydroORBITEncoder(nn.Module):
    def __init__(self, d_model: int, d_kv: int, d_ff: int, num_layers: int,
                 num_heads: int, dropout: float, layer_norm_eps: float,
                 rope_theta: float, act_fn: str, attn_impl: str, is_gated: bool = False,
                 use_qk_norm: bool = False, use_xpos: bool = False,
                 xpos_scale_base: float = 512.0, use_variate_mixer: bool = False,
                 variate_mixer_interval: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            mix_here = use_variate_mixer and variate_mixer_interval > 0 and (
                (layer_idx + 1) % variate_mixer_interval == 0
            )
            self.blocks.append(
                HydroORBITEncoderBlock(d_model, d_kv, d_ff, num_heads, dropout,
                                       layer_norm_eps, rope_theta, act_fn, attn_impl,
                                       is_gated=is_gated,
                                       use_qk_norm=use_qk_norm,
                                       use_xpos=use_xpos,
                                       xpos_scale_base=xpos_scale_base,
                                       use_variate_mixer=mix_here)
            )
        self.final_norm = HydroORBITLayerNorm(d_model, eps=layer_norm_eps)
        self.dropout    = nn.Dropout(dropout)

    @staticmethod
    def _expand_invert_time_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        mask = mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    @staticmethod
    def _construct_group_time_mask(
        group_ids: torch.Tensor, attn_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        group_mask = group_ids[:, None] == group_ids[None, :]
        gt_mask    = torch.einsum("qb,bt->qbt", group_mask.float(), attn_mask.float())
        gt_mask    = rearrange(gt_mask, "q b t -> t 1 q b")
        return (1.0 - gt_mask) * torch.finfo(dtype).min

    def forward(
        self,
        inputs_embeds:     torch.Tensor,
        group_ids:         torch.Tensor,
        attention_mask:    Optional[torch.Tensor] = None,
        position_ids:      Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> EncoderOutput:
        B, S, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(S, dtype=torch.long,
                                        device=inputs_embeds.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(B, S, device=inputs_embeds.device,
                                        dtype=inputs_embeds.dtype)

        ext_mask = self._expand_invert_time_mask(attention_mask, inputs_embeds.dtype)
        grp_mask = self._construct_group_time_mask(group_ids, attention_mask,
                                                   inputs_embeds.dtype)

        h = self.dropout(inputs_embeds)
        all_t, all_g = (), ()

        for block in self.blocks:
            out: EncoderBlockOutput = block(h, position_ids, ext_mask, grp_mask,
                                            group_ids,
                                            output_attentions)
            h = out.hidden_states
            if output_attentions:
                all_t = (*all_t, out.time_attn_weights)
                all_g = (*all_g, out.group_attn_weights)

        h = self.final_norm(h)
        h = self.dropout(h)
        return EncoderOutput(
            last_hidden_state=h,
            all_time_attn_weights=all_t or None,
            all_group_attn_weights=all_g or None,
        )
