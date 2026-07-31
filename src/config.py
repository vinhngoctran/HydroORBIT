import math
from typing import List, Literal, Optional

from transformers import PretrainedConfig


class HydroORBITForecastingConfig:
    """
    Resolution-agnostic forecasting config.

    - Separate input_patch_size / output_patch_size for context vs. horizon.
    - use_reg_token flag toggles a learned register token.
    - Accepts a legacy 'patch_size' key as an alias for input_patch_size.
    """

    def __init__(
        self,
        input_patch_size:  int            = 16,
        output_patch_size: int            = 16,
        patch_size:        Optional[int]  = None,   # legacy alias for input_patch_size
        context_length:    int            = 8760,
        max_output_patches: int           = 64,
        quantiles:         Optional[List[float]] = None,
        use_arcsinh:       bool           = True,
        use_reg_token:     bool           = True,
        use_causal_patch_scaler: bool     = True,
        scaler_fallback_min_obs: int      = 8,
        binaryaware_scaler: bool          = False,
        inverse_sinh_clip: float          = 20.0,
        sort_quantiles: bool              = True,
        **_kwargs,   # absorb unknown keys from older config.json files silently
    ):
        if patch_size is not None:
            input_patch_size = patch_size

        self.input_patch_size  = input_patch_size
        self.output_patch_size = output_patch_size
        self.context_length    = context_length
        self.max_output_patches = max_output_patches
        self.quantiles = quantiles or [
            0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
            0.5,  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
        ]
        self.use_arcsinh   = use_arcsinh
        self.use_reg_token = use_reg_token
        self.use_causal_patch_scaler = use_causal_patch_scaler
        self.scaler_fallback_min_obs = scaler_fallback_min_obs
        self.binaryaware_scaler = binaryaware_scaler
        self.inverse_sinh_clip = inverse_sinh_clip
        self.sort_quantiles = sort_quantiles

    @property
    def patch_size(self) -> int:
        """Legacy alias for input_patch_size."""
        return self.input_patch_size

    def num_output_patches(self, horizon: int) -> int:
        return math.ceil(horizon / self.output_patch_size)


class HydroORBITCoreConfig(PretrainedConfig):
    """
    HuggingFace-compatible model configuration.

    Default values match the released ~127M-param trained weights
    (d=768, L=8, heads=12, d_ff=3072, GeGLU).
    """

    model_type = "hydroorbit"

    def __init__(
        self,
        d_model:            int                           = 768,
        d_kv:               int                           = 64,
        d_ff:               int                           = 3072,
        num_layers:         int                           = 8,
        num_heads:          int                           = 12,
        dropout_rate:       float                         = 0.1,
        layer_norm_epsilon: float                         = 1e-6,
        initializer_factor: float                         = 0.05,
        dense_act_fn:       str                           = "gelu_new",
        rope_theta:         float                         = 10000.0,
        is_decoder:         bool                          = False,
        is_gated_act:       bool                          = True,
        use_qk_norm:        bool                          = True,
        use_xpos:           bool                          = True,
        xpos_scale_base:    float                         = 512.0,
        use_variate_mixer:  bool                          = True,
        variate_mixer_interval: int                       = 2,
        attn_implementation: Optional[Literal["eager", "sdpa"]] = None,
        pad_token_id:       int                           = 0,
        hydroorbit_config:  Optional[dict]                = None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.d_model              = d_model
        self.d_kv                 = d_kv
        self.d_ff                 = d_ff
        self.num_layers           = num_layers
        self.num_heads            = num_heads
        self.dropout_rate         = dropout_rate
        self.layer_norm_epsilon   = layer_norm_epsilon
        self.initializer_factor   = initializer_factor
        self.dense_act_fn         = dense_act_fn
        self.rope_theta           = rope_theta
        self.is_decoder           = is_decoder
        self.is_gated_act         = is_gated_act
        self.use_qk_norm          = use_qk_norm
        self.use_xpos              = use_xpos
        self.xpos_scale_base      = xpos_scale_base
        self.use_variate_mixer    = use_variate_mixer
        self.variate_mixer_interval = variate_mixer_interval
        self._attn_implementation = attn_implementation or "eager"
        self.hydroorbit_config    = hydroorbit_config or {}


def _fc_dict(fc: HydroORBITForecastingConfig) -> dict:
    return {
        "input_patch_size":  fc.input_patch_size,
        "output_patch_size": fc.output_patch_size,
        "context_length":    fc.context_length,
        "max_output_patches": fc.max_output_patches,
        "quantiles":         fc.quantiles,
        "use_arcsinh":       fc.use_arcsinh,
        "use_reg_token":     fc.use_reg_token,
        "use_causal_patch_scaler": fc.use_causal_patch_scaler,
        "scaler_fallback_min_obs": fc.scaler_fallback_min_obs,
        "binaryaware_scaler": fc.binaryaware_scaler,
        "inverse_sinh_clip": fc.inverse_sinh_clip,
        "sort_quantiles": fc.sort_quantiles,
    }


def small_config() -> HydroORBITCoreConfig:
    """~7M params — smoke tests and ablations."""
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=256, d_kv=32, d_ff=1024, num_layers=6, num_heads=8,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def tiny3m_config() -> HydroORBITCoreConfig:
    """~3M params — very small scaling-law ablation."""
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=128, d_kv=16, d_ff=512, num_layers=6, num_heads=8,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def tiny10m_config() -> HydroORBITCoreConfig:
    """~10M params — compact scaling-law ablation."""
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=288, d_kv=36, d_ff=1152, num_layers=4, num_heads=8,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def small20m_config() -> HydroORBITCoreConfig:
    """~20M params — small scaling-law ablation."""
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=288, d_kv=36, d_ff=1152, num_layers=9, num_heads=8,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def base50m_config() -> HydroORBITCoreConfig:
    """~50M params — mid-size scaling-law ablation."""
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=480, d_kv=60, d_ff=1920, num_layers=8, num_heads=8,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def medium_config() -> HydroORBITCoreConfig:
    """~39M params — fast iteration."""
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=512, d_kv=64, d_ff=2048, num_layers=8, num_heads=8,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def default_config() -> HydroORBITCoreConfig:
    """~127M params — default configuration (d=768, L=8, heads=12, GeGLU)."""
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=768, d_kv=64, d_ff=3072, num_layers=8, num_heads=12,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def large_config() -> HydroORBITCoreConfig:
    """~150M params — high-capacity variant.
    d=768, L=12, heads=12, d_ff=3072, GeGLU
    """
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=768, d_kv=64, d_ff=3072, num_layers=12, num_heads=12,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def xlarge_config() -> HydroORBITCoreConfig:
    """~308M params — 300M scale.
    d=1024, L=14, heads=16, d_ff=4096, GeGLU

    Each encoder block has 2 MHA modules (TimeSelfAttention + GroupSelfAttention)
    and a GeGLU FFN, giving ~20.97M params/layer at this width.

    Recommended training command (from this directory):
        python -u train.py \
            --hourly_dir ../data/prepared_hourly/train \
            --daily_dir  ../data/prepared_daily/train \
            --config     xlarge \
            --max_iters  600000 \
            --batch      1 \
            --grad_accum 8 \
            --lr         3e-5
    """
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=1024, d_kv=64, d_ff=4096, num_layers=14, num_heads=16,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def huge_config() -> HydroORBITCoreConfig:
    """~514M params — 500M scale.
    d=1280, L=15, heads=20, d_ff=5120, GeGLU

    Each encoder block has 2 MHA modules (TimeSelfAttention + GroupSelfAttention)
    and a GeGLU FFN, giving ~32.77M params/layer at this width.

    Recommended training command (from this directory):
        python -u train.py \\
            --hourly_dir ../data/prepared_hourly/train \\
            --daily_dir  ../data/prepared_daily/train \\
            --config     huge \\
            --max_iters  800000 \\
            --batch      1 \\
            --grad_accum 16 \\
            --lr         2e-5
    """
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=1280, d_kv=64, d_ff=5120, num_layers=15, num_heads=20,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )


def billion_config() -> HydroORBITCoreConfig:
    """~1.02B params — 1B scale.
    d=1536, L=21, heads=24, d_ff=6144, GeGLU

    Each encoder block has 2 MHA modules (TimeSelfAttention + GroupSelfAttention)
    and a GeGLU FFN, giving ~47.19M params/layer at this width.

    Recommended training command (from this directory):
        python -u train.py \
            --hourly_dir ../data/prepared_hourly/train \
            --daily_dir  ../data/prepared_daily/train \
            --config     billion \
            --max_iters  1000000 \
            --batch      1 \
            --grad_accum 32 \
            --lr         1e-5
    """
    fc = HydroORBITForecastingConfig()
    return HydroORBITCoreConfig(
        d_model=1536, d_kv=64, d_ff=6144, num_layers=21, num_heads=24,
        dropout_rate=0.1, dense_act_fn="gelu_new", is_gated_act=True,
        hydroorbit_config=_fc_dict(fc),
    )
