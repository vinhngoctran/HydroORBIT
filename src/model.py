import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .config import HydroORBITCoreConfig, HydroORBITForecastingConfig
from .layers import (
    CausalPatchInstanceNorm,
    MLP,
    MHA,
    InstanceNorm,
    HydroORBITEncoder,
    HydroORBITLayerNorm,
    Patch,
    ResidualBlock,
)


@dataclass
class HydroORBITOutput(ModelOutput):
    loss:                  Optional[torch.Tensor] = None
    quantile_preds:        Optional[torch.Tensor] = None   # (B, n_q, H) — original scale
    loc:                   Optional[torch.Tensor] = None   # (B, 1)
    scale:                 Optional[torch.Tensor] = None   # (B, 1)
    encoder_hidden_states: Optional[torch.Tensor] = None   # (B, S, d_model)


class HydroORBITModel(PreTrainedModel):
    """
    HydroORBIT — resolution-agnostic hydrological foundation model.

    Key design points:
    ─────────────────────────────────────────────────────────────────────
    1. Separate ctx_patch_embedding / fut_patch_embedding.
       Context tokens ("what happened") and future tokens ("what forcing
       is known") use independent projection networks, allowing the
       encoder to learn specialised representations for each role.

    2. Separate Patch modules for input_patch_size vs output_patch_size.
       Enables configurations where context is encoded at finer granularity
       than the prediction (e.g. in_ps=8, out_ps=16).

    3. use_reg_token flag.
       Can be disabled for ablations or when future tokens are sufficient
       to separate context from prediction in the attention patterns.

    4. encode() method.
       Returns encoder hidden states for context-only tasks (basin embedding,
       similarity search, transfer learning).

    5. Resolution-agnostic time encoding: positions normalised to [-1, 0]
       for context and [0, H/context_length] for future.
    6. arcsinh normalization: far better than z-score alone for skewed
       streamflow distributions (zero-inflation, heavy upper tail).
    7. 21-quantile output for fine-grained uncertainty characterisation.
    8. GeGLU activation (gelu_new gate): better gradient flow than ReLU.
    9. GroupSelfAttention: multivariate conditioning from forcing channels.
    10. Mesh-TF weight initialisation.

    Architecture: ~127M params at default_config()
        d=768, L=8, heads=12, d_kv=64, d_ff=3072, GeGLU
    """

    config_class = HydroORBITCoreConfig
    # MHA (layers.py) already implements a correct, scale-matched SDPA branch
    # (F.scaled_dot_product_attention, scale=1.0 to match eager). Without this
    # flag transformers' PreTrainedModel._supports_sdpa defaults to False and
    # refuses attn_implementation="sdpa" outright, forcing the slower unfused
    # eager path (manual matmul/softmax/matmul) for every attention call.
    _supports_sdpa = True

    def __init__(self, config: HydroORBITCoreConfig):
        super().__init__(config)
        self.fc = HydroORBITForecastingConfig(**config.hydroorbit_config)

        d     = config.d_model
        fc    = self.fc
        in_ps = fc.input_patch_size
        out_ps = fc.output_patch_size

        scaler_cls = CausalPatchInstanceNorm if fc.use_causal_patch_scaler else InstanceNorm
        if fc.use_causal_patch_scaler:
            self.instance_norm = scaler_cls(
                patch_size=in_ps,
                min_obs=fc.scaler_fallback_min_obs,
                use_arcsinh=fc.use_arcsinh,
                binaryaware=fc.binaryaware_scaler,
                inverse_sinh_clip=fc.inverse_sinh_clip,
            )
        else:
            self.instance_norm = scaler_cls(
                use_arcsinh=fc.use_arcsinh,
                binaryaware=fc.binaryaware_scaler,
                inverse_sinh_clip=fc.inverse_sinh_clip,
            )

        # Separate patches for context and output windows
        self.ctx_patch = Patch(in_ps, in_ps)
        self.out_patch = Patch(out_ps, out_ps)

        # Separate embeddings: context encodes [time, past_value, mask];
        # future encodes [time, covariate_value, covariate_mask].
        # Both produce d_model-dim tokens but learn distinct projections.
        self.ctx_patch_embedding = ResidualBlock(
            in_ps * 3, config.d_ff, d,
            config.dense_act_fn, config.dropout_rate,
        )
        self.fut_patch_embedding = ResidualBlock(
            out_ps * 3, config.d_ff, d,
            config.dense_act_fn, config.dropout_rate,
        )

        self.reg_token = nn.Embedding(1, d)

        self.encoder = HydroORBITEncoder(
            d_model=d,
            d_kv=config.d_kv,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            layer_norm_eps=config.layer_norm_epsilon,
            rope_theta=config.rope_theta,
            act_fn=config.dense_act_fn,
            attn_impl=config._attn_implementation,
            is_gated=config.is_gated_act,
            use_qk_norm=config.use_qk_norm,
            use_xpos=config.use_xpos,
            xpos_scale_base=config.xpos_scale_base,
            use_variate_mixer=config.use_variate_mixer,
            variate_mixer_interval=config.variate_mixer_interval,
        )

        n_q = len(fc.quantiles)
        self.output_patch_embedding = ResidualBlock(
            d, config.d_ff, n_q * out_ps,
            config.dense_act_fn, config.dropout_rate,
        )

        self.register_buffer(
            "quantile_levels",
            torch.tensor(fc.quantiles, dtype=torch.float32),
        )

        self.post_init()

    # ------------------------------------------------------------------
    # Weight initialisation — Mesh-TensorFlow style
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        factor  = self.config.initializer_factor
        d_model = self.config.d_model
        d_ff    = self.config.d_ff
        d_kv    = self.config.d_kv
        n_heads = self.config.num_heads

        if isinstance(module, MLP):
            module.wi.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            if hasattr(module, "wi_1"):
                module.wi_1.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * (d_ff ** -0.5))
            for proj in [module.wi, module.wo] + ([module.wi_1] if hasattr(module, "wi_1") else []):
                if proj.bias is not None:
                    proj.bias.data.zero_()

        elif isinstance(module, MHA):
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))

        elif isinstance(module, ResidualBlock):
            in_h = module.hidden_layer.weight.size(-1)
            in_o = module.output_layer.weight.size(-1)
            in_r = module.residual_layer.weight.size(-1)
            module.hidden_layer.weight.data.normal_( mean=0.0, std=factor * (in_h ** -0.5))
            module.output_layer.weight.data.normal_( mean=0.0, std=factor * (in_o ** -0.5))
            module.residual_layer.weight.data.normal_(mean=0.0, std=factor * (in_r ** -0.5))
            for layer in [module.hidden_layer, module.output_layer, module.residual_layer]:
                if layer.bias is not None:
                    layer.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)

        elif isinstance(module, HydroORBITLayerNorm):
            module.weight.data.fill_(1.0)

    # ------------------------------------------------------------------
    # Context: truncate → normalise → patch → time-encode
    # ------------------------------------------------------------------

    def _prepare_patched_context(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        fc      = self.fc
        in_ps   = fc.input_patch_size
        ctx_len = fc.context_length

        if context.shape[-1] > ctx_len:
            context = context[..., -ctx_len:]

        B = context.shape[0]

        ctx_mask  = (~torch.isnan(context)).to(torch.float32)
        ctx_norm, loc_scale = self.instance_norm(context)
        ctx_clean = torch.nan_to_num(ctx_norm, nan=0.0)

        patched_vals = self.ctx_patch(ctx_clean)
        patched_mask = torch.nan_to_num(self.ctx_patch(ctx_mask), nan=0.0)
        patched_vals = torch.where(patched_mask > 0.0,
                                   patched_vals, torch.zeros_like(patched_vals))

        n_ctx     = patched_vals.shape[1]
        final_len = n_ctx * in_ps

        time_enc = (
            torch.arange(-final_len, 0, dtype=torch.float32, device=context.device)
            .div(ctx_len)
            .to(patched_vals.dtype)
            .view(1, n_ctx, in_ps)
            .expand(B, -1, -1)
        )

        patch_feats = torch.cat([time_enc, patched_vals, patched_mask], dim=-1)
        attn_mask   = (patched_mask.sum(dim=-1) > 0).to(patched_vals.dtype)

        return patch_feats, attn_mask, loc_scale

    # ------------------------------------------------------------------
    # Future: normalise → patch → time-encode
    # ------------------------------------------------------------------

    def _prepare_patched_future(
        self,
        future_covariates:  torch.Tensor,
        loc_scale:          tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        B:                  int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fc         = self.fc
        out_ps     = fc.output_patch_size
        ctx_len    = fc.context_length
        H          = future_covariates.shape[-1]
        padded_len = num_output_patches * out_ps
        device     = future_covariates.device

        fut_mask = (~torch.isnan(future_covariates)).to(torch.float32)
        fut_norm, _ = self.instance_norm(future_covariates, loc_scale)
        fut_clean   = torch.nan_to_num(fut_norm, nan=0.0)

        if H < padded_len:
            pad = padded_len - H
            fut_clean = torch.cat(
                [fut_clean, torch.zeros(B, pad, device=device, dtype=fut_clean.dtype)], dim=-1
            )
            fut_mask = torch.cat(
                [fut_mask, torch.zeros(B, pad, device=device, dtype=fut_mask.dtype)], dim=-1
            )
        elif H > padded_len:
            fut_clean = fut_clean[:, :padded_len]
            fut_mask  = fut_mask[:, :padded_len]

        patched_fut  = rearrange(fut_clean, "b (n p) -> b n p", n=num_output_patches, p=out_ps)
        patched_mask = rearrange(fut_mask,  "b (n p) -> b n p", n=num_output_patches, p=out_ps)

        time_enc = (
            torch.arange(0, padded_len, dtype=torch.float32, device=device)
            .div(ctx_len)
            .to(patched_fut.dtype)
            .view(1, num_output_patches, out_ps)
            .expand(B, -1, -1)
        )

        patch_feats = torch.cat([time_enc, patched_fut, patched_mask], dim=-1)
        return patch_feats, patched_mask

    # ------------------------------------------------------------------
    # Pinball loss in normalised arcsinh space
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        qpreds_norm:   torch.Tensor,
        future_target: torch.Tensor,
        patched_mask:  torch.Tensor,
        loc_scale:     tuple[torch.Tensor, torch.Tensor],
        H:             int,
    ) -> torch.Tensor:
        ft_norm, _ = self.instance_norm(future_target, loc_scale)
        ft_mask    = (~torch.isnan(future_target)).to(ft_norm.dtype)

        # cov_mask: positions where future_covariates are known (not target)
        out_ps   = self.fc.output_patch_size
        cov_mask = rearrange(patched_mask, "b n p -> b (n p)")[:, :H]
        loss_mask = ft_mask * (1.0 - cov_mask)

        ft_norm = torch.nan_to_num(ft_norm, nan=0.0)

        qp = qpreds_norm[:, :, :H]
        ft = ft_norm.unsqueeze(1)
        ql = self.quantile_levels[None, :, None]

        quantile_loss = 2.0 * torch.abs((ft - qp) * ((ft <= qp).float() - ql))
        masked_loss   = quantile_loss * loss_mask.unsqueeze(1)

        return masked_loss.mean(dim=-1).sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # Encode-only (embedding extraction)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        context:   torch.Tensor,   # (B, Tc) — NaN=missing
        group_ids: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode context sequences and return hidden states + normalisation params.

        Returns
        -------
        hidden_states : (B, S, d_model) — S = n_ctx_patches [+ 1 REG]
        loc_scale     : (loc (B,1), scale (B,1)) for un-normalising predictions
        """
        B = context.shape[0]
        ctx_feats, ctx_attn_mask, loc_scale = self._prepare_patched_context(context)
        ctx_h = self.ctx_patch_embedding(ctx_feats)

        if self.fc.use_reg_token:
            reg_ids  = torch.zeros(B, 1, dtype=torch.long, device=context.device)
            reg_h    = self.reg_token(reg_ids)
            reg_mask = torch.ones(B, 1, dtype=ctx_feats.dtype, device=context.device)
            h         = torch.cat([ctx_h, reg_h], dim=1)
            attn_mask = torch.cat([ctx_attn_mask, reg_mask], dim=1)
        else:
            h         = ctx_h
            attn_mask = ctx_attn_mask

        enc_out = self.encoder(
            inputs_embeds=h,
            group_ids=group_ids,
            attention_mask=attn_mask,
        )
        return enc_out.last_hidden_state, loc_scale

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        context:            torch.Tensor,                   # (B, Tc) — NaN=missing
        future_covariates:  torch.Tensor,                   # (B, H)  — NaN=predict
        group_ids:          torch.Tensor,                   # (B,)
        num_output_patches: int,
        future_target:      Optional[torch.Tensor] = None,  # (B, H) for training
        output_attentions:  bool = False,
    ) -> HydroORBITOutput:
        B  = context.shape[0]
        H  = future_covariates.shape[-1]

        ctx_feats, ctx_attn_mask, loc_scale = self._prepare_patched_context(context)
        ctx_h = self.ctx_patch_embedding(ctx_feats)         # (B, n_ctx, d)

        fut_feats, patched_fut_mask = self._prepare_patched_future(
            future_covariates, loc_scale, num_output_patches, B
        )
        fut_h    = self.fut_patch_embedding(fut_feats)      # (B, n_out, d)
        fut_mask = torch.ones(B, num_output_patches, dtype=ctx_feats.dtype,
                              device=context.device)

        if self.fc.use_reg_token:
            reg_ids  = torch.zeros(B, 1, dtype=torch.long, device=context.device)
            reg_h    = self.reg_token(reg_ids)
            reg_mask = torch.ones(B, 1, dtype=ctx_feats.dtype, device=context.device)
            h         = torch.cat([ctx_h, reg_h, fut_h], dim=1)
            attn_mask = torch.cat([ctx_attn_mask, reg_mask, fut_mask], dim=1)
        else:
            h         = torch.cat([ctx_h, fut_h], dim=1)
            attn_mask = torch.cat([ctx_attn_mask, fut_mask], dim=1)

        enc_out = self.encoder(
            inputs_embeds=h,
            group_ids=group_ids,
            attention_mask=attn_mask,
            output_attentions=output_attentions,
        )

        # Last num_output_patches tokens are always the future tokens
        out_h = enc_out.last_hidden_state[:, -num_output_patches:]

        n_q  = self.quantile_levels.shape[0]
        out_ps = self.fc.output_patch_size
        qpreds_flat = self.output_patch_embedding(out_h)
        qpreds_norm = (
            qpreds_flat
            .view(B, num_output_patches, n_q, out_ps)
            .permute(0, 2, 1, 3)
            .reshape(B, n_q, num_output_patches * out_ps)
        )
        if self.fc.sort_quantiles:
            qpreds_norm = torch.sort(qpreds_norm, dim=1).values

        loss = None
        if future_target is not None:
            loss = self._compute_loss(
                qpreds_norm, future_target, patched_fut_mask, loc_scale, H
            )

        loc_exp   = loc_scale[0].unsqueeze(-1)
        scale_exp = loc_scale[1].unsqueeze(-1)
        qpreds    = self.instance_norm.inverse(
            qpreds_norm[:, :, :H], (loc_exp, scale_exp)
        )

        return HydroORBITOutput(
            loss=loss,
            quantile_preds=qpreds,
            loc=loc_scale[0],
            scale=loc_scale[1],
            encoder_hidden_states=enc_out.last_hidden_state if output_attentions else None,
        )
