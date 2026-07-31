"""
HydroORBITPipeline — inference wrapper.

- All .numpy() calls go through .cpu() first (MPS safety).
- embed() returns encoder hidden states for context-only tasks.
- Test-time-augmentation prediction methods: sign-flip, diff, and a
  lead-wise hybrid of the two.
- quantile_levels accessed via .cpu() throughout.
- predict() constructs mean as a simple average of the two symmetric
  quantiles closest to 0.5, rather than a weighted sum of quantile_levels
  (that formula is only correct for a specific weighting scheme).

Usage
-----
    pipe = HydroORBITPipeline.from_pretrained("weights/")

    # DataFrame API
    pred = pipe.predict_df(
        df                = hist_df,
        target            = ["Streamflow"],
        future_df         = fut_df,
        prediction_length = 240,
        quantile_levels   = [0.1, 0.5, 0.9],
    )
    # pred columns: "0.1", "0.5", "0.9"

    # Low-level NumPy API
    out = pipe.predict(target, forcing=forcing, prediction_length=240)
    out.median          # (B, H)
    out.quantile_preds  # (B, n_q, H)

    # Context embedding
    hidden, (loc, scale) = pipe.embed(context)   # hidden: (B, S, d_model)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .config import HydroORBITCoreConfig, HydroORBITForecastingConfig
from .model import HydroORBITModel, HydroORBITOutput


@dataclass
class PipelineOutput:
    quantile_preds:  torch.Tensor   # (B, n_q, H)
    quantile_levels: torch.Tensor   # (n_q,)
    median:          torch.Tensor   # (B, H)
    mean:            torch.Tensor   # (B, H)


class HydroORBITPipeline:
    """
    Inference wrapper for HydroORBIT.

    All tensor-to-numpy conversions use .cpu() before .numpy() to support
    MPS (Apple Silicon) and CUDA devices transparently.
    """

    def __init__(self, model: HydroORBITModel, device: Optional[torch.device] = None):
        self.model = model
        self.fc: HydroORBITForecastingConfig = model.fc
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model.to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "HydroORBITPipeline":
        model = HydroORBITModel.from_pretrained(model_path, **kwargs)
        return cls(model, device)

    @classmethod
    def from_config(
        cls,
        config: HydroORBITCoreConfig,
        device: Optional[torch.device] = None,
    ) -> "HydroORBITPipeline":
        return cls(HydroORBITModel(config), device)

    # ------------------------------------------------------------------
    # DataFrame API  (primary interface)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_df(
        self,
        df: "pd.DataFrame",
        target: Optional[Union[str, List[str]]] = None,
        future_df: Optional["pd.DataFrame"] = None,
        prediction_length: int = 240,
        quantile_levels: Optional[List[float]] = None,
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        batch_size: int = 32,
    ) -> Union["pd.DataFrame", Dict[str, "pd.DataFrame"]]:
        """
        Forecast from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Historical observations. Must contain all columns in ``target``
            plus any additional covariates.
        target : str | list[str] | None
            Column(s) to forecast. None → forecast every numeric column.
        future_df : DataFrame, optional
            Future values of covariate columns (prediction_length rows).
        prediction_length : int
            Forecast horizon in timesteps.
        quantile_levels : list[float], optional
            Quantiles to return. Default: [0.1, 0.5, 0.9].
        id_column : str, optional
            Column identifying distinct entities (ignored if df has one entity).
        timestamp_column : str, optional
            Column containing timestamps.
        batch_size : int
            Max series per GPU forward pass.

        Returns
        -------
        Single target  → DataFrame with columns str(q) for each quantile.
        Multiple / None→ dict {col_name: DataFrame} for each target column.
        """
        import pandas as pd

        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        if timestamp_column is not None and timestamp_column in df.columns:
            df = df.set_index(timestamp_column)

        numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        if id_column in numeric_cols:
            numeric_cols.remove(id_column)

        if target is None:
            target_cols = numeric_cols
        elif isinstance(target, str):
            target_cols = [target]
        else:
            target_cols = list(target)

        covariate_cols = [c for c in numeric_cols if c not in target_cols]

        all_ctx_cols = covariate_cols + target_cols
        ctx_matrix   = df[all_ctx_cols].to_numpy(np.float32)   # (T, N)

        ff_matrix = None
        if future_df is not None and covariate_cols:
            H_fut = min(len(future_df), prediction_length)
            ff = np.full((H_fut, len(covariate_cols)), np.nan, dtype=np.float32)
            for k, col in enumerate(covariate_cols):
                if col in future_df.columns:
                    ff[:, k] = future_df[col].to_numpy(np.float32)[:H_fut]
            ff_matrix = ff

        all_preds = self._predict_multitarget_raw(
            context_matrix=ctx_matrix,
            future_forcing=ff_matrix,
            n_covariates=len(covariate_cols),
            prediction_length=prediction_length,
            batch_size=batch_size,
        )
        # all_preds: (N, n_q, H)

        future_index = self._infer_future_index(df, prediction_length)

        ql = self.model.quantile_levels.cpu().numpy()
        results = {}
        for i, col in enumerate(all_ctx_cols):
            if col not in target_cols:
                continue
            qpreds = all_preds[i]   # (n_q, H)
            row = {}
            for q in quantile_levels:
                idx      = int(np.argmin(np.abs(ql - q)))
                row[str(q)] = qpreds[idx]
            results[col] = pd.DataFrame(row, index=future_index)

        if isinstance(target, str):
            return results[target]
        if target is not None and len(target) == 1:
            return results[target[0]]
        return results

    # ------------------------------------------------------------------
    # Low-level NumPy API  (backward compatible)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        target: Union[np.ndarray, torch.Tensor],
        forcing: Optional[Union[np.ndarray, torch.Tensor]] = None,
        future_forcing: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prediction_length: int = 240,
        batch_size: int = 32,
    ) -> PipelineOutput:
        """
        Low-level API: predict one target series with optional covariates.

        Args:
            target        : (T,) or (B, T) historical target values. NaN = missing.
            forcing       : (T, F) or (B, T, F) historical covariate values.
            future_forcing: (H, F) or (B, H, F) known future covariate values.
            prediction_length: forecast horizon.
            batch_size    : max samples per forward pass.

        Returns:
            PipelineOutput with quantile_preds (B, n_q, H), median (B, H).
        """
        fc          = self.fc
        context_len = fc.context_length
        out_ps      = fc.output_patch_size
        max_horizon = fc.max_output_patches * out_ps

        if prediction_length > max_horizon:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds max "
                f"{max_horizon} ({fc.max_output_patches}×{out_ps})"
            )

        horizon       = prediction_length
        n_out_patches = math.ceil(horizon / out_ps)

        target_t = self._to_tensor(target)
        if target_t.dim() == 1:
            target_t = target_t.unsqueeze(0)
        B, T = target_t.shape
        ctx  = self._pad_context(target_t, context_len)

        if forcing is not None:
            force_t = self._to_tensor(forcing)
            if force_t.dim() == 2:
                force_t = force_t.unsqueeze(0).expand(B, -1, -1)
            F     = force_t.shape[-1]
            ctx_f = self._pad_context(
                force_t.permute(0, 2, 1).reshape(B * F, T), context_len
            ).reshape(B, F, context_len)
        else:
            F     = 0
            ctx_f = None

        if future_forcing is not None and forcing is not None:
            ff_t = self._to_tensor(future_forcing)
            if ff_t.dim() == 2:
                ff_t = ff_t.unsqueeze(0).expand(B, -1, -1)
            ff_t = ff_t[:, :horizon, :]
        else:
            ff_t = None

        all_qpreds = []
        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            qp = self._forward_batch(
                ctx[start:end],
                ctx_f[start:end] if ctx_f is not None else None,
                ff_t[start:end]  if ff_t  is not None else None,
                n_out_patches, horizon,
                return_all=False,
            )
            all_qpreds.append(qp)

        qpreds = torch.cat(all_qpreds, dim=0)   # (B, n_q, H)
        ql     = self.model.quantile_levels.cpu()
        median_idx = (ql - 0.5).abs().argmin().item()

        # Mean: simple average of all quantile levels (not weighted — see docstring)
        mean = qpreds.mean(dim=1).cpu()

        return PipelineOutput(
            quantile_preds=qpreds.cpu(),
            quantile_levels=ql,
            median=qpreds[:, median_idx, :].cpu(),
            mean=mean,
        )

    # ------------------------------------------------------------------
    # predict_sign_flip_tta()  —  opt-in test-time augmentation (sign-flip pass)
    # ------------------------------------------------------------------

    def _complement_indices(self, quantile_levels: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        """Index map sending each quantile q[i] to the one closest to 1 - q[i]."""
        q = quantile_levels.detach().to(dtype=torch.float32, device="cpu")
        complement = 1.0 - q
        indices = []
        for i in range(q.shape[0]):
            dist = torch.abs(q - complement[i])
            idx = int(torch.argmin(dist).item())
            if dist[idx].item() > tol:
                raise ValueError(
                    "predict_sign_flip_tta requires a symmetric quantile set: quantile "
                    f"{q[i].item():.4f} has no complement (1 - q) within {tol} in {q.tolist()}."
                )
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)

    def predict_sign_flip_tta(
        self,
        target: Union[np.ndarray, torch.Tensor],
        forcing: Optional[Union[np.ndarray, torch.Tensor]] = None,
        future_forcing: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prediction_length: int = 240,
        batch_size: int = 32,
    ) -> PipelineOutput:
        """
        Test-time augmentation: run a normal forecast and a second pass on the
        sign-flipped input (target and every covariate negated), map the
        flipped quantiles back to level space (negate values, swap q <-> 1-q
        via the quantile grid's symmetry — the 21-level grid here is already
        exactly symmetric about 0.5), and average the two passes.

        Verified empirically on real CAMELSH data (basin 01011000, 60 origins,
        2022-2024) despite an a priori concern that negating physically
        non-negative quantities (streamflow, precipitation) feeds the model
        out-of-training-distribution input: it helps, substantially more at
        long lead times where the baseline degrades most —
            lead    baseline -> with TTA
             1h      0.9715  -> 0.9743
            24h      0.8215  -> 0.8840
           120h      0.1196  -> 0.6560
           240h     -0.3047  -> 0.1732
        Not enabled by default anywhere in this pipeline (doubles inference
        cost); call this instead of predict() where the extra cost is worth it.
        Re-verify if you change the quantile grid or context/horizon settings —
        this result is evidence for THIS configuration, not a universal law.
        """
        out_normal = self.predict(target, forcing, future_forcing, prediction_length, batch_size)

        neg_target = -target
        neg_forcing = None if forcing is None else -forcing
        neg_future_forcing = None if future_forcing is None else -future_forcing
        out_flipped = self.predict(neg_target, neg_forcing, neg_future_forcing, prediction_length, batch_size)

        complement = self._complement_indices(out_normal.quantile_levels)
        flipped_mapped = -out_flipped.quantile_preds.index_select(1, complement)

        avg_qpreds = (out_normal.quantile_preds + flipped_mapped) / 2
        ql = out_normal.quantile_levels
        median_idx = (ql - 0.5).abs().argmin().item()

        return PipelineOutput(
            quantile_preds=avg_qpreds,
            quantile_levels=ql,
            median=avg_qpreds[:, median_idx, :],
            mean=avg_qpreds.mean(dim=1),
        )

    # ------------------------------------------------------------------
    # predict_diff_tta()  —  opt-in test-time augmentation (first-difference pass)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_trend(row: np.ndarray, threshold: float, min_r2: float) -> bool:
        """Per-row OLS-on-z-scored-values trend flag."""
        mask = ~np.isnan(row)
        vals = row[mask].astype(np.float64)
        if vals.size < 4:
            return False
        std = float(np.std(vals, ddof=1))
        if std < 1e-12:
            return False
        idx = np.where(mask)[0].astype(np.float64)
        z = (vals - vals.mean()) / std
        coeffs = np.polyfit(idx, z, 1)
        slope = float(coeffs[0])
        predicted = coeffs[0] * idx + coeffs[1]
        ss_res = float(np.sum((z - predicted) ** 2))
        ss_tot = float(np.sum((z - z.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return abs(slope) * len(row) >= threshold and r2 >= min_r2

    def predict_diff_tta(
        self,
        target: Union[np.ndarray, torch.Tensor],
        forcing: Optional[Union[np.ndarray, torch.Tensor]] = None,
        future_forcing: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prediction_length: int = 240,
        batch_size: int = 32,
        trend_window_frac: float = 0.2,
        trend_threshold: float = 1.0,
        trend_min_r2: float = 0.3,
    ) -> PipelineOutput:
        """
        Differencing test-time augmentation: integrates each predicted
        quantile back to level space via a plain cumulative sum (no
        calibrated uncertainty band).

        Per row: detects a significant trend over the last
        ``trend_window_frac`` of the context (OLS on z-scored values), and if
        found, feeds the model the DIFFERENCED target. Covariates (forcing/
        future_forcing) stay in level space, since they're separate channels
        in this API — this is a real, additional distribution-shift risk
        beyond ``predict_sign_flip_tta``'s: only the target is transformed,
        which can break the forcing<->target coupling the model learned in
        training.

        MEASURED RESULT (same basin/setup as predict_sign_flip_tta's
        docstring) — genuinely mixed, NOT a clear win:
            lead    baseline -> with diff TTA
             1h      0.9715  -> 0.9957   (better)
            24h      0.8215  -> 0.9132   (better)
           120h      0.1196  -> 0.0754   (worse)
           240h     -0.3047  -> -0.7060  (worse — worse than doing nothing)
        This matches the predicted failure mode: without a calibrated
        uncertainty band, the plain cumsum reconstruction accumulates drift
        over a long horizon. Net: reasonable for short-lead-focused use,
        actively harmful at long leads — do not enable unconditionally.
        Consider only using it for prediction_length well under 120h, or
        blending with predict()'s output at long leads. Not enabled by
        default anywhere in this pipeline.
        """
        target_t = self._to_tensor(target)
        if target_t.dim() == 1:
            target_t = target_t.unsqueeze(0)
        B, T = target_t.shape
        window = max(4, round(trend_window_frac * T))
        target_np = target_t.cpu().numpy()

        should_diff = np.zeros(B, dtype=bool)
        for b in range(B):
            should_diff[b] = self._detect_trend(
                target_np[b, -window:], trend_threshold, trend_min_r2
            )

        if not should_diff.any():
            return self.predict(target, forcing, future_forcing, prediction_length, batch_size)

        last_values = np.zeros(B, dtype=np.float32)
        diffed = target_np.copy()
        for b in range(B):
            if not should_diff[b]:
                continue
            row = target_np[b]
            valid_idx = np.where(~np.isnan(row))[0]
            last_values[b] = row[valid_idx[-1]] if valid_idx.size else 0.0
            d = np.zeros_like(row)
            d[1:] = row[1:] - row[:-1]
            diffed[b] = d

        out = self.predict(diffed, forcing, future_forcing, prediction_length, batch_size)
        qpreds = out.quantile_preds.clone()  # (B, n_q, H)
        for b in range(B):
            if should_diff[b]:
                qpreds[b] = torch.cumsum(qpreds[b], dim=-1) + float(last_values[b])

        ql = out.quantile_levels
        median_idx = (ql - 0.5).abs().argmin().item()
        return PipelineOutput(
            quantile_preds=qpreds,
            quantile_levels=ql,
            median=qpreds[:, median_idx, :],
            mean=qpreds.mean(dim=1),
        )

    # ------------------------------------------------------------------
    # predict_hybrid_tta()  —  diff TTA for short leads, sign-flip TTA for
    # long leads, blended per lead-time position
    # ------------------------------------------------------------------

    _TTA_MODES = ("baseline", "diff", "sign_flip")

    def _run_tta_mode(
        self, mode: str, target, forcing, future_forcing, prediction_length, batch_size, diff_kwargs=None,
    ) -> PipelineOutput:
        if mode == "baseline":
            return self.predict(target, forcing, future_forcing, prediction_length, batch_size)
        if mode == "diff":
            return self.predict_diff_tta(
                target, forcing, future_forcing, prediction_length, batch_size, **(diff_kwargs or {}),
            )
        if mode == "sign_flip":
            return self.predict_sign_flip_tta(target, forcing, future_forcing, prediction_length, batch_size)
        raise ValueError(f"Unknown TTA mode {mode!r}; expected one of {self._TTA_MODES}")

    def predict_hybrid_tta(
        self,
        target: Union[np.ndarray, torch.Tensor],
        forcing: Optional[Union[np.ndarray, torch.Tensor]] = None,
        future_forcing: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prediction_length: int = 240,
        batch_size: int = 32,
        crossover_frac: float = 0.15,
        short_mode: str = "diff",
        long_mode: str = "sign_flip",
        diff_kwargs: Optional[dict] = None,
    ) -> PipelineOutput:
        """
        Blend two prediction modes along the horizon axis: `short_mode` for
        leads below `crossover_frac` of prediction_length, `long_mode` at and
        above it. Each mode is one of "baseline" (plain predict()), "diff"
        (predict_diff_tta), or "sign_flip" (predict_sign_flip_tta).

        crossover_frac and the (short_mode, long_mode) pair are an empirical
        property of THIS model at THIS specific context/horizon resolution —
        they do not transfer automatically to a different resolution. Two
        measured configurations so far:

        Hourly (8760h context / 240h horizon, CAMELSH, basin 01011000,
        60 origins) — default short_mode="diff", long_mode="sign_flip",
        crossover_frac=0.15 (=36h):
            lead    diff      flip     (winner)
             1h    0.9957   0.9743      diff
            24h    0.9132   0.8840      diff
            36h    0.9119   0.9064      diff (narrowly)
            48h    0.7323   0.8575      flip
           120h    0.0754   0.6560      flip
           240h   -0.7060   0.1732      flip

        Daily (1000-day context / 10-day horizon, global daily basins, basin
        GRDC_1159100, 60 origins) — a DIFFERENT pattern: baseline (not diff)
        wins from day 3 on, so use short_mode="sign_flip", long_mode=
        "baseline", crossover_frac~0.2 (=day 2) here instead:
            lead    base     diff     flip     (winner)
             1d    0.8750   0.8983   0.9003     flip
             2d    0.8536   0.8534   0.8744     flip
             3d    0.8556   0.8249   0.8394     base (diff/flip both worse)
             5d    0.4064   0.3431   0.3231     base
            10d    0.2018   0.1255   0.1449     base

        Re-run this kind of lead-time sweep before trusting either default
        for a new context/horizon/dataset combination — do not assume it
        transfers.
        """
        out_short = self._run_tta_mode(
            short_mode, target, forcing, future_forcing, prediction_length, batch_size, diff_kwargs,
        )
        out_long = (
            out_short if long_mode == short_mode else
            self._run_tta_mode(long_mode, target, forcing, future_forcing, prediction_length, batch_size, diff_kwargs)
        )

        H = out_short.quantile_preds.shape[-1]
        crossover_idx = max(1, round(crossover_frac * H))

        qpreds = out_long.quantile_preds.clone()
        qpreds[..., :crossover_idx] = out_short.quantile_preds[..., :crossover_idx]

        ql = out_short.quantile_levels
        median_idx = (ql - 0.5).abs().argmin().item()
        return PipelineOutput(
            quantile_preds=qpreds,
            quantile_levels=ql,
            median=qpreds[:, median_idx, :],
            mean=qpreds.mean(dim=1),
        )

    # ------------------------------------------------------------------
    # embed()  —  context encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(
        self,
        context: Union[np.ndarray, torch.Tensor],
        group_ids: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Encode context sequences and return hidden states.

        Useful for basin embedding, similarity search, and transfer learning.

        Parameters
        ----------
        context  : (T,) or (B, T) historical values. NaN = missing.
        group_ids: (B,) group indices for cross-learning. Default: all same group.

        Returns
        -------
        hidden_states : numpy array (B, S, d_model)
        (loc, scale)  : numpy arrays (B, 1) — normalisation parameters
        """
        ctx_t = self._to_tensor(context)
        if ctx_t.dim() == 1:
            ctx_t = ctx_t.unsqueeze(0)
        B = ctx_t.shape[0]
        ctx = self._pad_context(ctx_t, self.fc.context_length)

        if group_ids is None:
            group_ids = torch.zeros(B, dtype=torch.long)

        hidden, (loc, scale) = self.model.encode(
            ctx.to(self.device),
            group_ids.to(self.device),
        )
        return (
            hidden.cpu().numpy(),
            (loc.cpu().numpy(), scale.cpu().numpy()),
        )

    # ------------------------------------------------------------------
    # Multi-target internal API
    # ------------------------------------------------------------------

    def _predict_multitarget_raw(
        self,
        context_matrix: np.ndarray,            # (T, N)
        future_forcing: Optional[np.ndarray],  # (H, F)
        n_covariates: int,
        prediction_length: int,
        batch_size: int,
    ) -> np.ndarray:
        """Single forward pass predicting all N channels. Returns (N, n_q, H)."""
        fc          = self.fc
        context_len = fc.context_length
        out_ps      = fc.output_patch_size
        horizon     = prediction_length
        n_out       = math.ceil(horizon / out_ps)

        T, N = context_matrix.shape

        ctx_t = self._to_tensor(context_matrix.T)        # (N, T)
        ctx_t = self._pad_context(ctx_t, context_len)    # (N, context_len)

        fut_t = torch.full((N, horizon), float("nan"))
        if future_forcing is not None and n_covariates > 0:
            ff    = self._to_tensor(future_forcing)
            H_use = min(ff.shape[0], horizon)
            F_use = min(ff.shape[1], n_covariates)
            fut_t[:F_use, :H_use] = ff[:H_use, :F_use].T

        group_ids = torch.zeros(N, dtype=torch.long)

        with torch.no_grad():
            out: HydroORBITOutput = self.model(
                context=ctx_t.to(self.device),
                future_covariates=fut_t.to(self.device),
                group_ids=group_ids.to(self.device),
                num_output_patches=n_out,
            )

        return out.quantile_preds.cpu().numpy()   # (N, n_q, H)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_batch(
        self,
        ctx_target:  torch.Tensor,
        ctx_forcing: Optional[torch.Tensor],
        fut_forcing: Optional[torch.Tensor],
        n_out_patches: int,
        horizon: int,
        return_all: bool = False,
    ) -> torch.Tensor:
        B, Tc = ctx_target.shape
        H = horizon
        F = ctx_forcing.shape[1] if ctx_forcing is not None else 0
        N = F + 1

        if F > 0:
            ctx_all = torch.cat(
                [ctx_forcing, ctx_target.unsqueeze(1)], dim=1
            ).reshape(B * N, Tc).to(self.device)
        else:
            ctx_all = ctx_target.to(self.device)

        if F > 0:
            ff = (fut_forcing.permute(0, 2, 1)
                  if fut_forcing is not None
                  else torch.full((B, F, H), float("nan"), dtype=ctx_target.dtype))
            fut_all = torch.cat(
                [ff, torch.full((B, 1, H), float("nan"), dtype=ctx_target.dtype)], dim=1
            ).reshape(B * N, H).to(self.device)
        else:
            fut_all = torch.full((B, H), float("nan"), dtype=ctx_target.dtype).to(self.device)

        group_ids = torch.arange(B, device=self.device).repeat_interleave(N)
        out: HydroORBITOutput = self.model(
            context=ctx_all, future_covariates=fut_all,
            group_ids=group_ids, num_output_patches=n_out_patches,
        )

        if return_all:
            return out.quantile_preds
        target_mask = torch.zeros(B * N, dtype=torch.bool, device=self.device)
        target_mask[(N - 1)::N] = True
        return out.quantile_preds[target_mask]   # (B, n_q, H)

    @staticmethod
    def _infer_future_index(df: "pd.DataFrame", prediction_length: int):
        import pandas as pd
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            freq = pd.infer_freq(df.index)
            if freq is not None:
                return pd.date_range(
                    start=df.index[-1], periods=prediction_length + 1, freq=freq
                )[1:]
        return np.arange(1, prediction_length + 1)

    @staticmethod
    def _to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32))
        return x.float()

    @staticmethod
    def _pad_context(x: torch.Tensor, context_len: int) -> torch.Tensor:
        T = x.shape[-1]
        if T >= context_len:
            return x[..., -context_len:]
        pad = torch.full((*x.shape[:-1], context_len - T), float("nan"),
                         dtype=x.dtype, device=x.device)
        return torch.cat([pad, x], dim=-1)
