# HydroORBIT

**HydroORBIT** is a resolution-agnostic hydrological foundation model for streamflow and related
hydrological variable forecasting, described in *"An Adaptable Foundation Model for Operational River
Forecasting."*

The included `weights/` are exported from the final training checkpoint via the model's
HuggingFace-style `save_pretrained()`. The pretrained model can be downloaded here: https://huggingface.co/vinhtn/HydroORBIT.

## Design

- Resolution-agnostic context/future patching, with separate context and future patch embeddings.
- Future covariate masking for direct multi-step forecasting.
- Group-aware attention across series that share `group_ids`, enabling multivariate conditioning
  from forcing channels within a basin.
- Arcsinh normalization and a 21-quantile pinball loss, suited to zero-inflated, heavy-tailed
  hydrological variables such as streamflow.
- A patch-aware causal context scaler (`use_causal_patch_scaler`), with early-window fallback
  statistics (`scaler_fallback_min_obs`) and binary-aware scaling for 0/1 covariates
  (`binaryaware_scaler`).
- Clipped inverse `sinh` to avoid overflow on extreme hydrologic events.
- Q/K RMS normalization (`use_qk_norm`) and length-extrapolatable rotary position embeddings
  (`use_xpos`) for stable attention over long context windows.
- Sorted output quantiles (`sort_quantiles`) to reduce quantile crossing.
- A lightweight grouped variate mixer inserted every `variate_mixer_interval` encoder blocks: it
  pools same-group series at each token position, gates the pooled signal, and feeds it through a
  residual MLP before the normal feed-forward network — a cheap additional multivariate mixing path
  alongside the main cross-variate attention.
- A HuggingFace-style `save_pretrained` / `from_pretrained` workflow.
- Optional test-time-augmentation passes at inference (sign-flip, first-difference, and a
  lead-wise hybrid of the two) — see `src/pipeline.py`.

## Files

```text
HydroORBIT/
├── README.md
├── requirements.txt
├── train.py
├── train_ddp.py
├── weights/
│   ├── config.json
│   └── model.safetensors
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── layers.py
│   ├── model.py
│   └── pipeline.py
└── Example/
    ├── data/
    └── *.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start (Inference)

```python
from src.pipeline import HydroORBITPipeline

pipe = HydroORBITPipeline.from_pretrained("weights/")

# DataFrame API
pred = pipe.predict_df(
    df                = hist_df,             # historical context, one row per time step
    target            = ["Streamflow"],
    future_df         = fut_df,               # known future covariates, optional
    prediction_length = 240,
    quantile_levels    = [0.1, 0.5, 0.9],
)
# pred columns: "0.1", "0.5", "0.9"

# Low-level NumPy/Tensor API
out = pipe.predict(target, forcing=forcing, prediction_length=240)
out.median          # (B, H)
out.quantile_preds  # (B, n_q, H)
```

See `Example/` for five runnable notebooks, one per experiment type, each with a small sample
dataset included.

