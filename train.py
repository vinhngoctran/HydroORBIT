#!/usr/bin/env python
"""
HydroORBIT — pre-training script.

Training quality:
  --augment         Gaussian noise (σ=0.02) + channel dropout (p=0.10).
                    Noise adds robustness to sensor errors; channel dropout
                    teaches the model to forecast with missing covariates.

Checkpoint management:
  Keep last 3       Older checkpoints deleted automatically; saves disk.

Weight warm-start:
  --warm_start_dir PATH   Load compatible weights from another weights/
                    directory (e.g. an earlier or differently-sized run).
                    Layers are matched by name and shape; any layer whose
                    name or shape does not match remains freshly initialised.

LR warm-restart:
  --restart_lr      When combined with --warm_start_dir or --resume, reset
                    the scheduler so LR ramps from 0 again. Useful when
                    switching to a new LR or architecture variant.

Training philosophy:
  Random target/forcing split — one channel is chosen per step as the
  unknown target, the rest are known covariates.
  Multi-resolution: 1× hourly step + 2× daily steps per optimizer iter.
  AdamW, cosine decay, gradient clipping.

Run from this directory:
    # Both resolutions (recommended):
    python -u train.py \\
        --hourly_dir ../Data/prepared_hourly/train \\
        --daily_dir  ../Data/prepared_daily/train

    # Warm-start from another weights directory:
    python -u train.py \\
        --hourly_dir ../Data/prepared_hourly/train \\
        --daily_dir  ../Data/prepared_daily/train \\
        --warm_start_dir path/to/other/weights \\
        --restart_lr

    # Resume training:
    python -u train.py [same args] --resume

Run from project root:
    cd /path/to/project
    python -u HydroORBIT/train.py \\
        --hourly_dir Data/prepared_hourly/train \\
        --daily_dir  Data/prepared_daily/train \\
        --out_dir    HydroORBIT/weights \\
        --ckpt_dir   HydroORBIT/checkpoints \\
        --max_iters  300000 \\
        --batch      1 \\
        --grad_accum 4 \\
        --lr         8e-5 \\
        --augment \\
        --resume
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")
from src.config import (base50m_config, billion_config, default_config,
                        huge_config, large_config, medium_config,
                        small20m_config, small_config, tiny10m_config,
                        tiny3m_config, xlarge_config)
from src.model import HydroORBITModel

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--hourly_dir",    default=None,
                    help="Directory of hourly parquet files (train split)")
parser.add_argument("--daily_dir",     default=None,
                    help="Directory of daily parquet files (train split)")
parser.add_argument("--ctx",           type=int, default=8760,
                    help="Hourly context length (timesteps)")
parser.add_argument("--horizon",       type=int, default=240,
                    help="Hourly forecast horizon (timesteps)")
parser.add_argument("--daily_ctx",     type=int, default=730,
                    help="Daily context length (timesteps)")
parser.add_argument("--daily_horizon", type=int, default=90,
                    help="Daily forecast horizon (timesteps)")
parser.add_argument("--out_dir",       default="weights",
                    help="Directory to save final HuggingFace weights")
parser.add_argument("--ckpt_dir",      default="checkpoints",
                    help="Directory for resumable checkpoints")
parser.add_argument("--config",        default="default",
                    choices=["tiny3m", "tiny10m", "small20m", "base50m",
                             "small", "medium", "default", "large",
                             "xlarge", "huge", "billion"],
                    help="Model size preset  (tiny3m~3M, tiny10m~10M, "
                         "small20m~20M, base50m~50M, small~7M, medium~39M, "
                         "default~103M, large~150M, xlarge~308M, huge~514M, "
                         "billion~1.02B)")
parser.add_argument("--max_iters",     type=int, default=300_000,
                    help="Total optimizer iterations")
parser.add_argument("--batch",         type=int, default=1,
                    help="Batch size per gradient step")
parser.add_argument("--grad_accum",    type=int, default=4,
                    help="Gradient accumulation steps")
parser.add_argument("--lr",            type=float, default=8e-5,
                    help="Peak learning rate")
parser.add_argument("--seed",          type=int, default=42)
parser.add_argument("--augment",       action="store_true",
                    help="Enable data augmentation (Gaussian noise + channel dropout)")
parser.add_argument("--noise_sigma",   type=float, default=0.02,
                    help="Gaussian noise std for --augment")
parser.add_argument("--chan_drop",     type=float, default=0.10,
                    help="Per-channel dropout probability for --augment")
parser.add_argument("--resume",    dest="resume", action="store_true",  default=True,
                    help="Resume from latest checkpoint in --ckpt_dir (default: on)")
parser.add_argument("--no_resume", dest="resume", action="store_false",
                    help="Ignore checkpoints and start fresh")
parser.add_argument("--resume_iter", type=int, default=None,
                    help="Resume from a specific iteration, e.g. --resume_iter 70000 "
                         "loads iter_00070000.pt; implies --resume")
parser.add_argument("--warm_start_dir", default=None,
                    help="Load a compatible weights directory (partial, see docstring)")
parser.add_argument("--restart_lr",    action="store_true",
                    help="Reset LR schedule when using --warm_start_dir or --resume")
parser.add_argument("--keep_ckpts",    type=int, default=3,
                    help="Number of recent checkpoints to keep")
args = parser.parse_args()

if not args.hourly_dir and not args.daily_dir:
    parser.error("Provide at least one of --hourly_dir or --daily_dir")

OUT_DIR  = Path(args.out_dir);   OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path(args.ckpt_dir);  CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR  = Path("logs");         LOG_DIR.mkdir(exist_ok=True)

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
SEED         = args.seed
MAX_ITERS    = args.max_iters
BATCH        = args.batch
GRAD_ACCUM   = args.grad_accum
LR           = args.lr
MIN_LR_R     = 0.05          # floor: final LR = peak × MIN_LR_R
GRAD_CLIP    = 1.0
WEIGHT_DECAY = 0.01
WARMUP_ITERS = max(1, MAX_ITERS // 100)

H_CTX = args.ctx
H_HOR = args.horizon
D_CTX = args.daily_ctx
D_HOR = args.daily_horizon

PRINT_EVERY = 200
CKPT_EVERY  = 10_000

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)
print(f"Device : {DEVICE}", flush=True)


# ─── Column auto-detection ────────────────────────────────────────────────────
def detect_columns(files: list, n_scan: int = 5) -> list:
    col_sets = []
    for f in files[:n_scan]:
        try:
            df = pd.read_parquet(f)
            numerics = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerics:
                col_sets.append(set(numerics))
        except Exception:
            pass
    if not col_sets:
        raise ValueError(f"Could not detect columns from {files[:n_scan]}.")
    common = col_sets[0]
    for s in col_sets[1:]:
        common &= s
    return sorted(common)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    """
    Random-window sampler. Each item: (ctx, tgt) for ALL N channels.
    Target/forcing split happens inside training_step, not here.

    Window acceptance: randomly-chosen target channel (selected in training_step)
    must have at least one non-NaN value in the forecast horizon, so the step
    always produces a meaningful gradient. Checked here by requiring that at
    least half the channels have valid horizon data (conservative proxy).
    """

    def __init__(self, files: list, ctx: int, hor: int, columns: list,
                 windows_per_file: int = 8):
        self.files   = files
        self.ctx     = ctx
        self.wl      = ctx + hor
        self.columns = columns
        self.N       = len(columns)
        self.wpf     = windows_per_file
        self.bad_files = set()

    def __len__(self):
        return len(self.files) * self.wpf

    def _load(self, path: Path) -> np.ndarray:
        df  = pd.read_parquet(path)
        T   = len(df)
        arr = np.full((T, self.N), np.nan, dtype=np.float32)
        for i, col in enumerate(self.columns):
            if col in df.columns:
                arr[:, i] = df[col].to_numpy(np.float32)
        return arr

    def __getitem__(self, idx):
        for attempt in range(100):
            fp = self.files[idx % len(self.files)] if attempt == 0 else random.choice(self.files)
            try:
                arr = self._load(fp)
            except Exception as e:
                if fp not in self.bad_files:
                    print(f"  [data skip] cannot read {fp}: {e}", flush=True)
                    self.bad_files.add(fp)
                if len(self.bad_files) >= len(self.files):
                    raise RuntimeError("All parquet files failed to load.") from e
                continue
            T = arr.shape[0]
            if T < self.wl:
                continue
            s = random.randint(0, T - self.wl)
            w = arr[s : s + self.wl]
            hor = w[self.ctx:]
            valid_cols = (~np.isnan(hor)).any(axis=0).sum()
            if valid_cols < max(1, self.N // 2):
                continue
            return {
                "ctx": torch.tensor(w[: self.ctx]),   # (Tc, N)
                "tgt": torch.tensor(w[self.ctx:]),    # (H,  N)
            }
        return self.__getitem__(random.randrange(len(self)))


def collate_pad(batch):
    mc = max(x["ctx"].shape[0] for x in batch)
    ctxs, tgts = [], []
    for it in batch:
        p = mc - it["ctx"].shape[0]
        N = it["ctx"].shape[-1]
        ctxs.append(
            torch.cat([torch.full((p, N), float("nan")), it["ctx"]]) if p else it["ctx"]
        )
        tgts.append(it["tgt"])
    return {"ctx": torch.stack(ctxs), "tgt": torch.stack(tgts)}


# ─── Data augmentation ────────────────────────────────────────────────────────
def augment_batch(ctx: torch.Tensor, tgt: torch.Tensor,
                  noise_sigma: float, chan_drop_p: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian noise + per-channel dropout applied to raw context/target.

    Gaussian noise (σ per-value):
        Makes the model robust to sensor quantisation and measurement errors.
        Applied only to non-NaN values; NaN positions stay NaN.

    Channel dropout (entire channels zeroed → NaN):
        Teaches the model to forecast when some covariates are unavailable.
        Each channel independently dropped with probability chan_drop_p.
        The target channel selected in training_step is never dropped here
        (it becomes NaN in the future window anyway via the target/forcing split).
    """
    # Gaussian noise on non-NaN values
    if noise_sigma > 0:
        noise = torch.randn_like(ctx) * noise_sigma
        ctx = torch.where(torch.isnan(ctx), ctx, ctx + noise)
        noise = torch.randn_like(tgt) * noise_sigma
        tgt = torch.where(torch.isnan(tgt), tgt, tgt + noise)

    # Per-channel dropout: shape (1, 1, N) broadcast over (B, T, N)
    if chan_drop_p > 0 and ctx.shape[-1] > 1:
        N    = ctx.shape[-1]
        keep = torch.bernoulli(
            torch.full((1, 1, N), 1.0 - chan_drop_p, device=ctx.device)
        )
        nan_val = torch.full_like(ctx, float("nan"))
        ctx = torch.where(keep == 1, ctx, nan_val)
        tgt = torch.where(keep == 1, tgt, torch.full_like(tgt, float("nan")))

    return ctx, tgt


# ─── Training step ────────────────────────────────────────────────────────────
def training_step(batch, nout: int, model, device, grad_accum: int,
                  augment: bool = False, noise_sigma: float = 0.02,
                  chan_drop_p: float = 0.10) -> torch.Tensor:
    """
    Randomly selects exactly 1 channel as the unknown target:
      - target channel:   future = NaN → model predicts; loss computed here
      - forcing channels: future = actual → model uses as known NWP-like input
    """
    ctx = batch["ctx"].to(device)   # (B, Tc, N)
    tgt = batch["tgt"].to(device)   # (B, H,  N)

    if augment:
        ctx, tgt = augment_batch(ctx, tgt, noise_sigma, chan_drop_p)

    B, Tc, N = ctx.shape
    H        = tgt.shape[1]

    ctx_all = ctx.permute(0, 2, 1).reshape(B * N, Tc)
    tgt_BNH = tgt.permute(0, 2, 1)   # (B, N, H)

    valid_mask = ~tgt_BNH.isnan().all(dim=0).all(dim=1)   # (N,) bool
    valid_idxs = valid_mask.nonzero(as_tuple=False).squeeze(1).tolist()
    target_j   = random.choice(valid_idxs) if valid_idxs else random.randrange(N)

    fut_BNH = tgt_BNH.clone()
    fut_BNH[:, target_j, :] = float("nan")

    loss_BNH = torch.full_like(tgt_BNH, float("nan"))
    loss_BNH[:, target_j, :] = tgt_BNH[:, target_j, :]

    fut_all = fut_BNH.reshape(B * N, H)
    tgt_all = loss_BNH.reshape(B * N, H)

    gids = torch.arange(B, device=device).repeat_interleave(N)

    model.train()
    out = model(
        context=ctx_all, future_covariates=fut_all,
        group_ids=gids, num_output_patches=nout,
        future_target=tgt_all,
    )
    return out.loss / grad_accum


# ─── Data loading ─────────────────────────────────────────────────────────────
def make_loader(dir_path, ctx, hor, batch_size):
    files = sorted(Path(dir_path).glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {dir_path}")
    cols = detect_columns(files)
    print(f"  columns ({len(cols)}): {cols[:6]}{'…' if len(cols)>6 else ''}", flush=True)
    ds = WindowDataset(files, ctx, hor, cols)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collate_pad, num_workers=0, pin_memory=False), \
           len(files), len(cols)


from src.config import HydroORBITForecastingConfig
FC     = HydroORBITForecastingConfig()
OUT_PS = FC.output_patch_size
H_NOUT = math.ceil(H_HOR / OUT_PS)
D_NOUT = math.ceil(D_HOR / OUT_PS)

hl = dl = None
has_hourly = has_daily = False

if args.hourly_dir:
    print(f"Hourly data: {args.hourly_dir}", flush=True)
    hl, n_h, n_cols_h = make_loader(args.hourly_dir, H_CTX, H_HOR, BATCH)
    has_hourly = True
    print(f"  {n_h} files  {n_cols_h} channels/file", flush=True)

if args.daily_dir:
    print(f"Daily data:  {args.daily_dir}", flush=True)
    dl, n_d, n_cols_d = make_loader(args.daily_dir, D_CTX, D_HOR, BATCH)
    has_daily = True
    print(f"  {n_d} files  {n_cols_d} channels/file", flush=True)

steps_per_iter = (1 if has_hourly else 0) + (2 if has_daily else 0)

# ─── Model ────────────────────────────────────────────────────────────────────
cfg_map = {
    "tiny3m":  tiny3m_config,
    "tiny10m": tiny10m_config,
    "small20m": small20m_config,
    "base50m": base50m_config,
    "small":   small_config,
    "medium":  medium_config,
    "default": default_config,
    "large":   large_config,
    "xlarge":  xlarge_config,
    "huge":    huge_config,
    "billion": billion_config,
}
cfg   = cfg_map[args.config]()
model = HydroORBITModel(cfg).to(DEVICE)
n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nConfig : {args.config}  d={cfg.d_model}  L={cfg.num_layers}  "
      f"heads={cfg.num_heads}  gated={cfg.is_gated_act}", flush=True)
print(f"Params : {n_p/1e6:.1f}M", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR,
                        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))


def lr_schedule(opt_step: int) -> float:
    if opt_step < WARMUP_ITERS:
        return opt_step / max(1, WARMUP_ITERS)
    t = (opt_step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    return MIN_LR_R + (1.0 - MIN_LR_R) * 0.5 * (1.0 + math.cos(math.pi * t))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

# ─── Weight initialisation from another compatible checkpoint ───────────────
def load_warm_start_weights(model: HydroORBITModel, warm_start_dir: str) -> HydroORBITModel:
    """
    Partially load a compatible weights directory into this model.

    Name mapping (older weights → current model):
        input_patch_embedding.*  →  ctx_patch_embedding.*   (renamed)
    Layers with no matching name or shape in the source weights are left at
    their fresh initialisation (e.g. fut_patch_embedding.*, or any layer
    introduced since the source weights were saved).
    All other layers: loaded when shapes match.
    """
    from safetensors.torch import load_file
    import os

    sf_path = os.path.join(warm_start_dir, "model.safetensors")
    if not os.path.exists(sf_path):
        raise FileNotFoundError(f"No model.safetensors in {warm_start_dir}")

    src_state = load_file(sf_path)
    dst_state = model.state_dict()

    loaded = fresh = shape_mismatch = 0
    for k_dst in list(dst_state.keys()):
        # ctx_patch_embedding was input_patch_embedding in older checkpoints
        k_src = k_dst.replace("ctx_patch_embedding.", "input_patch_embedding.")
        if k_src in src_state and dst_state[k_dst].shape == src_state[k_src].shape:
            dst_state[k_dst] = src_state[k_src]
            loaded += 1
        elif k_dst in src_state and dst_state[k_dst].shape == src_state[k_dst].shape:
            dst_state[k_dst] = src_state[k_dst]
            loaded += 1
        elif k_dst in src_state:
            shape_mismatch += 1   # shape changed — keep random init
        else:
            fresh += 1            # new or shape-changed — keep random init

    model.load_state_dict(dst_state)
    print(f"  warm-start: {loaded} tensors loaded, "
          f"{fresh} fresh, {shape_mismatch} shape-mismatched (kept random)", flush=True)
    return model


# ─── Resume / warm-start load ────────────────────────────────────────────────
start_iter = 0

if args.warm_start_dir:
    print(f"Loading warm-start weights from: {args.warm_start_dir}", flush=True)
    model = load_warm_start_weights(model, args.warm_start_dir)
    if not args.restart_lr:
        print("  Hint: consider --restart_lr to ramp LR from 0 with new weights",
              flush=True)

if args.resume_iter is not None:
    args.resume = True   # --resume_iter implies --resume

if args.resume:
    if args.resume_iter is not None:
        ck_path = CKPT_DIR / f"iter_{args.resume_iter:08d}.pt"
        if not ck_path.exists():
            available = [p.name for p in sorted(CKPT_DIR.glob("iter_*.pt"))]
            raise FileNotFoundError(
                f"Checkpoint {ck_path.name} not found in {CKPT_DIR}.\n"
                f"Available: {available}"
            )
        ckpts_to_load = [ck_path]
    else:
        ckpts_to_load = sorted(CKPT_DIR.glob("iter_*.pt"))

    if ckpts_to_load:
        ck_path = ckpts_to_load[-1]
        ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if not args.restart_lr:
            sched.load_state_dict(ck["sched"])
            start_iter = ck["iteration"]
        print(f"Resumed from {ck_path.name}  (iter {ck['iteration']})  "
              f"→ continuing from iter {start_iter + 1}", flush=True)
        if args.restart_lr:
            print("  LR schedule reset to 0 (--restart_lr)", flush=True)
    else:
        print("No checkpoint found — starting fresh.", flush=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def next_batch(loader, it):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    all_ckpts = sorted(ckpt_dir.glob("iter_*.pt"))
    for old in all_ckpts[:-keep]:
        old.unlink(missing_ok=True)


# ─── Training loop ────────────────────────────────────────────────────────────
hit = iter(hl) if has_hourly else None
dit = iter(dl) if has_daily  else None

it          = start_iter
accum_count = 0
lh_sum = ld_sum = 0.0
nh = nd = 0
t0 = time.time()

aug_str = f"noise={args.noise_sigma} chan_drop={args.chan_drop}" if args.augment else "off"
print(
    f"\nTraining {MAX_ITERS:,} iters  "
    f"BATCH={BATCH}  GRAD_ACCUM={GRAD_ACCUM}  "
    f"steps_per_iter={steps_per_iter}  LR={LR:.0e}  WARMUP={WARMUP_ITERS}",
    flush=True,
)
print(f"Augmentation: {aug_str}", flush=True)
header = (
    f"{'Iter':>7}  {'H-loss':>8}  {'D-loss':>8}  "
    f"{'LR':>9}  {'s/it':>7}  {'ETA':>8}"
)
print(header, flush=True)

opt.zero_grad()

while it < MAX_ITERS:

    # ── hourly step ───────────────────────────────────────────────────
    if has_hourly:
        bh, hit = next_batch(hl, hit)
        lh = training_step(bh, H_NOUT, model, DEVICE, GRAD_ACCUM,
                           augment=args.augment,
                           noise_sigma=args.noise_sigma,
                           chan_drop_p=args.chan_drop)
        lh.backward()
        lh_sum += lh.item() * GRAD_ACCUM;  nh += 1;  accum_count += 1

    # ── daily steps (×2 to balance the larger daily corpus) ──────────
    if has_daily:
        for _ in range(2):
            bd, dit = next_batch(dl, dit)
            ld = training_step(bd, D_NOUT, model, DEVICE, GRAD_ACCUM,
                               augment=args.augment,
                               noise_sigma=args.noise_sigma,
                               chan_drop_p=args.chan_drop)
            ld.backward()
            ld_sum += ld.item() * GRAD_ACCUM;  nd += 1;  accum_count += 1

    # ── optimizer step ────────────────────────────────────────────────
    if accum_count >= GRAD_ACCUM * steps_per_iter:
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()
        accum_count = 0

    it += 1

    # ── logging ───────────────────────────────────────────────────────
    if it % PRINT_EVERY == 0 or it == 1:
        elapsed = time.time() - t0
        spi     = elapsed / max(1, it - start_iter)
        eta     = (MAX_ITERS - it) * spi
        lr_now  = sched.get_last_lr()[0] * LR
        h_loss  = f"{lh_sum/max(1,nh):.4f}" if has_hourly else "   n/a"
        d_loss  = f"{ld_sum/max(1,nd):.4f}" if has_daily  else "   n/a"
        print(
            f"{it:>7}  {h_loss:>8}  {d_loss:>8}  "
            f"{lr_now:>9.3e}  {spi:>7.3f}  {eta/60:>7.1f}m",
            flush=True,
        )
        lh_sum = ld_sum = 0.0;  nh = nd = 0

    # ── checkpoint ────────────────────────────────────────────────────
    if it % CKPT_EVERY == 0:
        ck_path = CKPT_DIR / f"iter_{it:08d}.pt"
        torch.save({
            "iteration": it,
            "model":  model.state_dict(),
            "opt":    opt.state_dict(),
            "sched":  sched.state_dict(),
            "config": cfg.to_dict(),
        }, ck_path)
        print(f"  Checkpoint saved: {ck_path.name}", flush=True)
        prune_checkpoints(CKPT_DIR, keep=args.keep_ckpts)


# ─── Save final weights ───────────────────────────────────────────────────────
model.save_pretrained(str(OUT_DIR))
elapsed = time.time() - t0
print(
    f"\nTraining complete — {it:,} iters  "
    f"{elapsed/60:.1f} min  ({elapsed/max(1,it-start_iter):.3f} s/iter)",
    flush=True,
)
print(f"Weights saved to: {OUT_DIR}", flush=True)
