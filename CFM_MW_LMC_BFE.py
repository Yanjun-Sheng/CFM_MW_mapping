import torch
print(torch.__version__)
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.func import vjp

import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

from dataclasses import dataclass
from astropy.io import fits
# we will read in data with pandas frame
import math

from scipy.stats import qmc
from scipy.optimize import minimize

import time
from datetime import datetime

from typing import List, Dict, Any, Optional
from torchdiffeq import odeint


def set_seed(seed: int = 42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)


if torch.cuda.is_available():
  print("ran on GPU")

idx_sim = np.genfromtxt('/g/data/y89/ys9983/LHC_fit_final_precise.txt')[:,0]
param_sim = np.genfromtxt('/g/data/y89/ys9983/LHC_fit_final_precise.txt')[:,-4:]

np.random.seed(42)    # <--- fix randomness here

S = len(idx_sim)
perm = np.random.permutation(S)

train_size = int(0.9 * S)
train_idx = perm[:train_size]
valid_idx = perm[train_size:]

# split θ and suite ids
idx_train = idx_sim[train_idx]
idx_valid = idx_sim[valid_idx]

param_train = param_sim[train_idx]
param_valid = param_sim[valid_idx]

print("Train suites:", len(idx_train))
print("Valid suites:", len(idx_valid))


# ============================================================
# 0. CONFIG
# ============================================================

@dataclass
class Config:
    # data
    data_dir: str = "/g/data/y89/ys9983/Condition_FM/isotropic_MW_halo"
    n_per_suite: int = 100_000              # each file already contains 100k rows
    batch_size: int = 512
    # model
    dim: int = 6
    hidden_dim: int = 1024
    n_layers: int = 7
    time_fourier_dim: int = 16
    theta_dim: int = 4                      # (M_MW, M_LMC, c, q) for example
    theta_emb_dim: int = 256
    sigma_min: float = 0.0
    # training
    total_steps: int = 1500_000
    warmup_steps: int = 50_000
    decay_start: int = 25_000
    lr_max: float = 1e-4
    weight_decay: float = 1e-4
    print_every: int = 1000
    eval_every: int = 2000
    snapshot_every: int = 20_000
    # solver / sampling
    ode_nfe: int = 128
    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------- CURRICULUM HYPERPARAMETERS -------------
    use_curriculum_theta: bool = True
    theta_curr_end: int = 300_000      # steps over which θ-curriculum fades out

    # theta-space curriculum (in normalized θ space)
    theta_sigma_early      = 0.75      # strongly emphasize very central θ
    theta_sigma_late       = 2.5       # cover most of the prior by the end
    theta_min_weight       = 0.3       # don't zero out extreme θ

cfg = Config()
device = torch.device(cfg.device)


# ============================================================
# 1. NORMALIZATION STATS (positions/velocities + parameters)
# ============================================================

@dataclass
class NormStats:
    center_pos: torch.Tensor  # [3]
    center_vel: torch.Tensor  # [3]
    scale_pos: torch.Tensor   # scalar tensor
    scale_vel: torch.Tensor   # scalar tensor

    def normalize(self, x6: torch.Tensor) -> torch.Tensor:
        # x6: [N,6]
        device = x6.device
        center_pos = self.center_pos.to(device)
        center_vel = self.center_vel.to(device)
        scale_pos  = self.scale_pos.to(device)
        scale_vel  = self.scale_vel.to(device)

        xp, xv = x6[:, :3], x6[:, 3:]
        xp_n = (xp - center_pos) / (scale_pos + 1e-8)
        xv_n = (xv - center_vel) / (scale_vel + 1e-8)
        return torch.cat([xp_n, xv_n], dim=1)


    def denormalize(self, xn6: torch.Tensor) -> torch.Tensor:
        # xn6: [N,6]
        device = xn6.device
        center_pos = self.center_pos.to(device)
        center_vel = self.center_vel.to(device)
        scale_pos  = self.scale_pos.to(device)
        scale_vel  = self.scale_vel.to(device)

        xp_n, xv_n = xn6[:, :3], xn6[:, 3:]
        xp = xp_n * (scale_pos + 1e-8) + center_pos
        xv = xv_n * (scale_vel + 1e-8) + center_vel
        return torch.cat([xp, xv], dim=1)

    def to(self, device):
        self.center_pos = self.center_pos.to(device)
        self.center_vel = self.center_vel.to(device)
        self.scale_pos  = self.scale_pos.to(device)
        self.scale_vel  = self.scale_vel.to(device)
        return self

@dataclass
class ThetaStats:
    mean: torch.Tensor   # [theta_dim]
    std:  torch.Tensor   # [theta_dim]
    def normalize(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.mean) / (self.std + 1e-8)
    def denormalize(self, thetan: torch.Tensor) -> torch.Tensor:
        return thetan * (self.std + 1e-8) + self.mean
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
        return self

def compute_stats6d_from_files(idx_sim, data_dir, subsample_per_file=5000):
    """Compute robust isotropic position/velocity scales from a small subsample across files."""
    pos_list, vel_list = [], []
    rng = np.random.default_rng(42)
    for idx in idx_sim[:100]:
        path = os.path.join(data_dir, f"sample_{int(idx)}.txt")
        # Load columns 1..6 (0-based): use np.loadtxt or genfromtxt; here assume whitespace-separated
        arr = np.loadtxt(path, usecols=range(1,7))  # shape (~100k,6)
        m = arr.shape[0]
        take = min(subsample_per_file, m)
        sel = rng.choice(m, size=take, replace=False)
        pos_list.append(arr[sel, :3])
        vel_list.append(arr[sel, 3:6])
    pos = torch.from_numpy(np.concatenate(pos_list, axis=0)).float()
    vel = torch.from_numpy(np.concatenate(vel_list, axis=0)).float()
    center_pos = torch.zeros(3)   # assume Galactocentric
    center_vel = torch.zeros(3)
    r_pos = torch.linalg.norm(pos - center_pos, dim=1)
    r_vel = torch.linalg.norm(vel - center_vel, dim=1)
    scale_pos = torch.quantile(r_pos, 0.95)
    scale_vel = torch.quantile(r_vel, 0.95)
    return NormStats(center_pos, center_vel, scale_pos, scale_vel)

def compute_theta_stats(param_sim_np):
    thetas = torch.from_numpy(param_sim_np).float()
    mean = thetas.mean(dim=0)
    std  = thetas.std(dim=0, unbiased=False)
    # guard small std to avoid exploding normalization
    std = torch.clamp(std, min=1e-6)
    return ThetaStats(mean, std)

# You already have these numpy arrays in your environment:
# idx_sim: np.ndarray shape (S,)
# param_sim: np.ndarray shape (S, cfg.theta_dim)
# stats6d can also reuse your prior one if you prefer.
stats6d = compute_stats6d_from_files(idx_sim, cfg.data_dir, subsample_per_file=5000).to(cfg.device)
theta_stats = compute_theta_stats(param_sim).to(cfg.device)


# ============================================================
# 3. MODEL: CONDITIONAL VELOCITY FIELD u(t, x; θ)
# ============================================================

class TimeFourier(nn.Module):
    def __init__(self, fourier_dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.arange(fourier_dim).float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2: t = t.squeeze(-1)
        angles = math.pi * (2.0 ** self.freqs)[None, :] * t[:, None]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class ThetaEmbed(nn.Module):
    def __init__(self, in_dim=4, hidden=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim), nn.GELU()
        )
    def forward(self, theta):  # (B,P)
        return self.net(theta)

class FlowCond(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers, time_fourier_dim, theta_dim, theta_emb_dim=128, use_radius=True):
        super().__init__()
        self.time_embed = TimeFourier(time_fourier_dim)
        self.theta_embed = ThetaEmbed(theta_dim, hidden=128, out_dim=theta_emb_dim)
        self.use_radius = use_radius
        extra_in = 2 if use_radius else 0
        in_dim = dim + extra_in + 2*time_fourier_dim + theta_emb_dim
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(in_dim if i==0 else hidden_dim, hidden_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, dim)
    def forward(self, x: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        tfeat = self.time_embed(t)
        the = self.theta_embed(theta)
        if self.use_radius:
            r_pos = torch.linalg.norm(x[:, :3], dim=1, keepdim=True)
            s_vel = torch.linalg.norm(x[:, 3:], dim=1, keepdim=True)
            h = torch.cat([x, r_pos, s_vel, tfeat, the], dim=-1)
        else:
            h = torch.cat([x, tfeat, the], dim=-1)
        return self.out(self.mlp(h))

model = FlowCond(
    dim=cfg.dim,
    hidden_dim=cfg.hidden_dim,
    n_layers=cfg.n_layers,
    time_fourier_dim=cfg.time_fourier_dim,
    theta_dim=cfg.theta_dim,
    theta_emb_dim=cfg.theta_emb_dim,
    use_radius=True
).to(device)


# ============================================================
# 4. CONDITIONAL CFM TARGETS (same scheduler; now with θ)
# ============================================================

def make_cfm_batch(x1_phys: torch.Tensor, theta_phys: torch.Tensor, stats6d: NormStats, theta_stats: ThetaStats, sigma_min=0.0):
    """
    x1_phys: (B,6) in physical units
    theta_phys: (B,P) original parameter values
    Returns t (B,), x_t (B,6), target_u (B,6), plus normalized x1n and thetan
    """
    x1n = stats6d.normalize(x1_phys)
    thetan = theta_stats.normalize(theta_phys)

    B, D = x1n.shape
    x0 = torch.randn(B, D, device=x1n.device)

    eps = 1e-3
    t = eps + (1.0 - 2*eps) * torch.rand(B, 1, device=x1n.device)

    sigma_t = 1.0 - (1.0 - sigma_min) * t
    x_t = sigma_t * x0 + t * x1n
    target_u = x1n - (1.0 - sigma_min) * x0  # matches your earlier correction

    return t.squeeze(-1), x_t, target_u, x1n, thetan


# ============================================================
# 5. TRAINING LOOP (by steps, not epochs)
# ============================================================

def lr_schedule(step: int):
    s = float(step)
    if s < cfg.warmup_steps:
        return cfg.lr_max * s / max(1.0, float(cfg.warmup_steps))
    if s < cfg.decay_start:
        return cfg.lr_max
    progress = (s - cfg.decay_start) / max(1.0, float(cfg.total_steps - cfg.decay_start))
    progress = max(0.0, min(1.0, progress))
    return 0.5 * cfg.lr_max * (1.0 + math.cos(math.pi * progress))

opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)

# simplest restore
model.load_state_dict(torch.load("/g/data/y89/ys9983/CFM_MW_LMC/CFM_6d_cur_2.pt", map_location=device, weights_only=False))
model.eval()

# ============================================================
# 6. CONDITIONAL SAMPLING + STRINGENT LOG-LIKELIHOOD
# ============================================================

def vf_raw_cond(t, x, theta_in):
    tau = 0.5 * (1.0 - torch.cos(torch.pi * t))
    dtaudt = 0.5 * torch.pi * torch.sin(torch.pi * t)
    tb = tau.expand(x.size(0))
    return model(x, tb, theta_in) * dtaudt

def divergence_vjp_cond(t_scalar, x_in, theta_in, nH=4, generator=None):
    x_req = x_in.detach().requires_grad_(True)
    def f(x): 
        return vf_raw_cond(t_scalar, x, theta_in)
    y, vjp_fn = vjp(f, x_req)
    div_terms = []
    for _ in range(nH):
        if generator is None:
            eps = torch.randn_like(x_req)
        else:
            eps = torch.randn(x_req.shape, device=x_req.device, dtype=x_req.dtype, generator=generator)
        (Jt_eps,) = vjp_fn(eps)
        div_terms.append((Jt_eps * eps).sum(dim=1))

    div = torch.stack(div_terms, 0).mean(0)
    return y.detach(), div.detach()


@torch.no_grad()
def sample_cond(model, theta_phys, n_samples=50_000, nfe=128, seed=None):
    model.eval()
    device = torch.device(cfg.device)

    # Set seed if provided for reproducibility
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None

    thetan = theta_stats.normalize(torch.as_tensor(theta_phys, dtype=torch.float32, device=device))
    if thetan.dim()==1: thetan = thetan.unsqueeze(0).repeat(n_samples,1)
    
    if generator is not None:
        x0 = torch.randn(n_samples, cfg.dim, device=device, generator=generator)
    else:
        x0 = torch.randn(n_samples, cfg.dim, device=device)
    
    t_grid = torch.linspace(0.0, 1.0, nfe+1, device=device)

    def vf(t, x): return vf_raw_cond(t, x, thetan)
    x_traj = odeint(vf, x0, t_grid, method="rk4", atol=1e-5, rtol=1e-5)
    x1n = x_traj[-1]#.detach()
    x1_phys = stats6d.denormalize(x1n).cpu()
    return x1_phys


def save_samples_to_fits(samples, output_path, theta_phys=None):
    """
    Save generated samples to a FITS file.
    
    Parameters:
    -----------
    samples : torch.Tensor or np.ndarray
        Shape (N, 6) with columns [X, Y, Z, VX, VY, VZ] in physical units
    output_path : str
        Path to output FITS file
    theta_phys : np.ndarray, optional
        The parameter vector used to generate these samples
    """
    # Convert to numpy if needed
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    
    # Create a structured array with named columns
    n_samples = samples.shape[0]
    dtype = [('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), 
             ('VX', 'f8'), ('VY', 'f8'), ('VZ', 'f8')]
    
    data = np.zeros(n_samples, dtype=dtype)
    data['X'] = samples[:, 0]
    data['Y'] = samples[:, 1]
    data['Z'] = samples[:, 2]
    data['VX'] = samples[:, 3]
    data['VY'] = samples[:, 4]
    data['VZ'] = samples[:, 5]
    
    # Create HDU
    primary_hdu = fits.PrimaryHDU()
    table_hdu = fits.BinTableHDU(data=data)
    
    # Add metadata to header
    table_hdu.header['N_SAMPLE'] = (n_samples, 'Number of samples')
    table_hdu.header['COORDSYS'] = ('Galactocentric', 'Coordinate system')
    table_hdu.header['UNIT_POS'] = ('kpc', 'Position units')
    table_hdu.header['UNIT_VEL'] = ('km/s', 'Velocity units')
    
    if theta_phys is not None:
        param_names = ['M_MW', 'M_LMC', 'c', 'q']
        for i, (name, val) in enumerate(zip(param_names, theta_phys)):
            table_hdu.header[f'THETA{i}'] = (float(val), f'Parameter {name}')
    
    # Create HDU list and save
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(output_path, overwrite=True)
    print(f"Saved {n_samples} samples to {output_path}")
    
    return output_path


def loglik_of_sample_cond(model, x1_phys, theta_phys, nfe=128, nH=4, generator=None):
    """
    Evaluate stringent log p(x | theta) for arbitrary physical samples.
    """
    model.eval()
    # x1_phys might already be a torch tensor on device
    if torch.is_tensor(x1_phys):
        x1 = x1_phys.to(device=device, dtype=torch.float32)
    else:
        x1 = torch.as_tensor(x1_phys, dtype=torch.float32, device=device)

    x1n = stats6d.normalize(x1)
    D = x1n.shape[1]

    thetan_single = theta_stats.normalize(torch.as_tensor(theta_phys, dtype=torch.float32, device=device))
    if thetan_single.dim()==1:
        thetan = thetan_single.unsqueeze(0).repeat(x1.shape[0], 1)
    else:
        thetan = thetan_single

    t_grid = torch.linspace(1.0, 0.0, nfe+1, device=device)

    def vf_aug_rev(t, state):
        x, logp = state
        # reuse divergence_vjp_cond (no minus sign on the vector field)
        v, div = divergence_vjp_cond(t, x, thetan, nH=nH, generator=generator)
        # same ODE as forward: d logp_corr / dt = - div(u_t)
        return (v, -div)

    x0_traj, logp_traj = odeint(vf_aug_rev, (x1n, torch.zeros(x1n.size(0), device=device)),
                                t_grid, method="rk4", atol=1e-5, rtol=1e-5)
    x0 = x0_traj[-1]
    logp_corr = logp_traj[-1]  # = ∫_1^0 -div dt = ∫_0^1 div dt

    logp0 = -0.5*(x0.pow(2).sum(1) + D*math.log(2.0*math.pi))
    logp1_norm = logp0 - logp_corr

    log_abs_det = 3.0*torch.log(stats6d.scale_pos.to(device)+1e-8) + 3.0*torch.log(stats6d.scale_vel.to(device)+1e-8)
    logp1_phys = logp1_norm - log_abs_det
    finite_mask = torch.isfinite(logp1_phys)
    if not finite_mask.all():
        # You can print here occasionally to monitor
        logp1_phys = torch.where(
            finite_mask,
            logp1_phys,
            torch.full_like(logp1_phys, -1e10),  # effectively -∞
        )

    return logp1_phys


def batched_loglik_of_sample_cond(
    model,
    x1_phys_all,      # (N,6) torch tensor (CPU or GPU) or numpy
    theta_phys,
    nfe=128,
    nH=4,
    batch_size=512,
    generator=None,
):

    if not torch.is_tensor(x1_phys_all):
        x1_phys_all = torch.as_tensor(x1_phys_all, dtype=torch.float32)

    # Put on device ONCE (critical speedup)
    if x1_phys_all.device != device:
        x1_phys_all = x1_phys_all.to(device=device, dtype=torch.float32)
    else:
        x1_phys_all = x1_phys_all.to(dtype=torch.float32)

    N = x1_phys_all.shape[0]
    logp_chunks = []

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        x_batch = x1_phys_all[start:end]  # already on device

        # call your existing function on this batch
        logp_batch = loglik_of_sample_cond(
            model,
            x_batch,
            theta_phys,
            nfe=nfe,
            nH=nH,
            generator=generator,
        )  # returns (B,) on device

        logp_chunks.append(logp_batch)

    return torch.cat(logp_chunks, dim=0)  # (N,) on device


# ============================================================
# 7. SETUP OUTPUT DIRECTORY AND CONFIGURATION
# ============================================================

theta_true = np.array([0.7, 15.0, 9.415, 1.0], dtype=np.float32)

outdir = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs"   # adjust if needed
os.makedirs(outdir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
chain_tag = f"{timestamp}"

# ---- user-facing config ----
#SCAN_OUTDIR = os.path.join(outdir, f"BFE_{chain_tag}")
#os.makedirs(SCAN_OUTDIR, exist_ok=True)

# profile settings (even cleaner than refinement if you want)
PROF_NFE = 128
PROF_NH  = 8
PROF_BS  = 5000
PROF_SEED = 2015

# Set seed for reproducible sampling
torch.manual_seed(PROF_SEED)
torch.cuda.manual_seed_all(PROF_SEED)
np.random.seed(PROF_SEED)

# Output directory for generated samples
samples_outdir = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k"
os.makedirs(samples_outdir, exist_ok=True)

params         = ["M_mw", "M_lmc", "c", "q"]
step_size_abs  = {"M_mw": 0.05, "M_lmc": 0.5, "c": 0.3, "q": 0.05}



'''  # ===== COMMENTED OUT: Test theta suites (already generated) =====
# ============================================================
# TEST THETA SUITES (10 parameter combinations, non-overlap with LHC grid)
# ============================================================
# Goal:
#   Build 10 test suites for transfer validation of both linear and neural
#   compressors away from the fiducial theta.
#
# For each suite (one theta from param_sim):
#   - N_cov   = 2000 fiducial repeats  (suite-specific "fiducial" pool)
#   - N_deriv = 100 repeats per perturbation for each of 4 params × 2 dirs
#             = 800 perturbed files total
#   - total files per suite = 2800
#
# Seed indexing rule follows BFE_Fisher_neural.ipynb style partitions:
#   suite_base_seed = TEST_BASE_SEED + suite_idx * (N_cov + N_deriv)
#   fiducial seeds  : [suite_base_seed, suite_base_seed + N_cov - 1]
#   deriv seeds     : [suite_base_seed + N_cov, suite_base_seed + N_cov + N_deriv - 1]
#     (same deriv-seed block reused across all param +/- perturbation files)
# ============================================================

TEST_SUITES_OUTDIR = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k/test_theta_suites"
TEST_LHC_THETA_FILE = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k/lhc_theta_grid.npy"
TEST_N_SUITES = 10
TEST_N_COV = 2000
TEST_N_DERIV = 100
TEST_SAMPLES_PER_FILE = 5000
TEST_BASE_SEED = 5855   # starts immediately after LHC block 5215..5854
TEST_SELECT_SEED = 20260224
TEST_SUITE_START_IDX = int(os.getenv("TEST_SUITE_START_IDX", "0"))
TEST_SUITE_STOP_IDX = int(os.getenv("TEST_SUITE_STOP_IDX", str(TEST_N_SUITES)))  # exclusive
TEST_THETA_SUITES_VERBOSE = False


def ts_print(*args, **kwargs):
    if TEST_THETA_SUITES_VERBOSE:
        print(*args, **kwargs)


def _theta_key(theta_row: np.ndarray, decimals: int = 8):
    return tuple(np.round(np.asarray(theta_row, dtype=np.float64), decimals=decimals).tolist())


def select_test_thetas_no_lhc_overlap(
    all_theta: np.ndarray,
    lhc_theta_file: str,
    n_select: int,
    rng_seed: int,
):
    """
    Select n_select rows from all_theta with no overlap against rows in lhc_theta_file.
    """
    if os.path.exists(lhc_theta_file):
        lhc_theta = np.load(lhc_theta_file)
        lhc_set = {_theta_key(row) for row in lhc_theta}
    else:
        print(f"WARNING: LHC theta grid not found at {lhc_theta_file}; overlap exclusion disabled.")
        lhc_theta = np.empty((0, all_theta.shape[1]), dtype=np.float64)
        lhc_set = set()

    keep_mask = np.array([_theta_key(row) not in lhc_set for row in all_theta], dtype=bool)
    candidates = all_theta[keep_mask]

    if len(candidates) < n_select:
        raise RuntimeError(
            f"Only {len(candidates)} non-LHC candidates available, need {n_select}."
        )

    rng = np.random.default_rng(rng_seed)
    chosen_idx = rng.choice(len(candidates), size=n_select, replace=False)
    chosen_idx = np.sort(chosen_idx)
    selected = candidates[chosen_idx].astype(np.float32)
    return selected, candidates, lhc_theta


def suite_is_complete(suite_dir: str, expected_fits: int) -> bool:
    done_marker = os.path.join(suite_dir, "suite_done.txt")
    if os.path.exists(done_marker):
        return True

    if not os.path.isdir(suite_dir):
        return False

    suite_meta_path = os.path.join(suite_dir, "suite_meta.npz")
    if not os.path.exists(suite_meta_path):
        return False

    n_fits = sum(1 for name in os.listdir(suite_dir) if name.endswith(".fits"))
    if n_fits >= expected_fits:
        with open(done_marker, "w") as f:
            f.write(f"completed_at={datetime.now().isoformat()}\n")
            f.write(f"n_fits={n_fits}\n")
            f.write(f"expected_fits={expected_fits}\n")
        return True

    return False


os.makedirs(TEST_SUITES_OUTDIR, exist_ok=True)

theta_ref_fid = np.array([0.7, 15.0, 9.415, 1.0], dtype=np.float32)
test_thetas, theta_candidates, lhc_theta_used = select_test_thetas_no_lhc_overlap(
    all_theta=param_sim,
    lhc_theta_file=TEST_LHC_THETA_FILE,
    n_select=TEST_N_SUITES,
    rng_seed=TEST_SELECT_SEED,
)

manifest_path = os.path.join(TEST_SUITES_OUTDIR, "test_theta_suite_manifest.npz")
if not os.path.exists(manifest_path):
    np.savez(
        manifest_path,
        selected_thetas=test_thetas,
        fiducial_theta=theta_ref_fid,
        all_candidates_non_lhc=theta_candidates,
        lhc_theta_grid=lhc_theta_used,
        params=np.array(params),
        step_size_abs=np.array([step_size_abs[p] for p in params], dtype=np.float32),
        n_suites=np.int64(TEST_N_SUITES),
        n_cov=np.int64(TEST_N_COV),
        n_deriv=np.int64(TEST_N_DERIV),
        test_base_seed=np.int64(TEST_BASE_SEED),
        selection_seed=np.int64(TEST_SELECT_SEED),
    )
    ts_print(f"Manifest written : {manifest_path}")
else:
    ts_print(f"Manifest already exists (not overwritten): {manifest_path}")

ts_print("\n" + "="*70)
ts_print("GENERATING TEST THETA SUITES (NON-OVERLAP WITH LHC GRID)")
ts_print("="*70)
ts_print(f"Suites               : {TEST_N_SUITES}")
ts_print(f"Per-suite files      : {TEST_N_COV} fiducial + {2*len(params)*TEST_N_DERIV} perturbed = {TEST_N_COV + 2*len(params)*TEST_N_DERIV}")
ts_print(f"Base seed            : {TEST_BASE_SEED}")
ts_print(f"Suite seed stride    : {TEST_N_COV + TEST_N_DERIV}")
ts_print(f"Selection seed       : {TEST_SELECT_SEED}")
ts_print(f"Manifest             : {manifest_path}")

if not (0 <= TEST_SUITE_START_IDX <= TEST_N_SUITES):
    raise ValueError(f"TEST_SUITE_START_IDX={TEST_SUITE_START_IDX} out of valid range [0, {TEST_N_SUITES}]")
if not (0 <= TEST_SUITE_STOP_IDX <= TEST_N_SUITES):
    raise ValueError(f"TEST_SUITE_STOP_IDX={TEST_SUITE_STOP_IDX} out of valid range [0, {TEST_N_SUITES}]")
if TEST_SUITE_STOP_IDX < TEST_SUITE_START_IDX:
    raise ValueError(
        f"Invalid suite range: start={TEST_SUITE_START_IDX} > stop={TEST_SUITE_STOP_IDX}. "
        "TEST_SUITE_STOP_IDX must be >= TEST_SUITE_START_IDX."
    )

ts_print(f"Suite range (0-based): [{TEST_SUITE_START_IDX}, {TEST_SUITE_STOP_IDX})")

expected_suite_fits = TEST_N_COV + 2 * len(params) * TEST_N_DERIV

for sidx in range(TEST_SUITE_START_IDX, TEST_SUITE_STOP_IDX):
    theta_suite = test_thetas[sidx]
    suite_dir = os.path.join(TEST_SUITES_OUTDIR, f"suite_{sidx:03d}")
    os.makedirs(suite_dir, exist_ok=True)

    if suite_is_complete(suite_dir, expected_suite_fits):
        ts_print(f"\nSuite {sidx:02d} already complete -> skipping")
        continue

    suite_t0 = time.time()
    suite_base_seed = TEST_BASE_SEED + sidx * (TEST_N_COV + TEST_N_DERIV)
    fid_seed_start = suite_base_seed
    fid_seed_end = suite_base_seed + TEST_N_COV - 1
    deriv_seed_start = suite_base_seed + TEST_N_COV
    deriv_seed_end = deriv_seed_start + TEST_N_DERIV - 1

    rel = (theta_suite - theta_ref_fid) / np.maximum(theta_ref_fid, 1e-8)
    ts_print("\n" + "-"*70)
    ts_print(f"Suite {sidx:02d}  theta={theta_suite.tolist()}")
    ts_print(f"  delta/fid (%)   : {(100.0*rel).round(3).tolist()}")
    ts_print(f"  fid seeds       : {fid_seed_start:05d}..{fid_seed_end:05d}  (N={TEST_N_COV})")
    ts_print(f"  deriv seeds     : {deriv_seed_start:05d}..{deriv_seed_end:05d}  (N={TEST_N_DERIV})")

    suite_meta_path = os.path.join(suite_dir, "suite_meta.npz")
    if not os.path.exists(suite_meta_path):
        np.savez(
            suite_meta_path,
            theta_suite=theta_suite,
            theta_fiducial=theta_ref_fid,
            params=np.array(params),
            step_size_abs=np.array([step_size_abs[p] for p in params], dtype=np.float32),
            n_cov=np.int64(TEST_N_COV),
            n_deriv=np.int64(TEST_N_DERIV),
            suite_base_seed=np.int64(suite_base_seed),
            fid_seed_start=np.int64(fid_seed_start),
            fid_seed_end=np.int64(fid_seed_end),
            deriv_seed_start=np.int64(deriv_seed_start),
            deriv_seed_end=np.int64(deriv_seed_end),
        )

    # (A) suite-specific fiducial samples at theta_suite
    n_written_fid, n_skipped_fid = 0, 0
    for seed in range(fid_seed_start, fid_seed_end + 1):
        out_fid = os.path.join(suite_dir, f"fiducial_samples_seed{seed:05d}.fits")
        if os.path.exists(out_fid):
            n_skipped_fid += 1
            continue
        x = sample_cond(model, theta_suite, n_samples=TEST_SAMPLES_PER_FILE,
                        nfe=cfg.ode_nfe, seed=seed)
        save_samples_to_fits(x, out_fid, theta_phys=theta_suite)
        n_written_fid += 1
        if n_written_fid % 50 == 0:
            ts_print(f"    fiducial progress: written={n_written_fid} skipped={n_skipped_fid}")

    ts_print(f"  ✓ fiducial done: written={n_written_fid}, skipped={n_skipped_fid}")

    # (B) perturbed samples (4 params × 2 dirs × N_deriv)
    n_written_pert, n_skipped_pert = 0, 0
    theta_suite_dict = dict(zip(params, theta_suite.tolist()))

    for pidx, pname in enumerate(params):
        dp = step_size_abs[pname]  # absolute step, NOT fractional
        for direction, sign in [("plus", +1.0), ("minus", -1.0)]:
            theta_pert = theta_suite.copy()
            theta_pert[pidx] += sign * dp

            for j in range(TEST_N_DERIV):
                seed = deriv_seed_start + j
                out_pert = os.path.join(suite_dir, f"{pname}_{direction}_seed{seed:05d}.fits")
                if os.path.exists(out_pert):
                    n_skipped_pert += 1
                    continue
                x = sample_cond(model, theta_pert, n_samples=TEST_SAMPLES_PER_FILE,
                                nfe=cfg.ode_nfe, seed=seed)
                save_samples_to_fits(x, out_pert, theta_phys=theta_pert)
                n_written_pert += 1

            ts_print(f"    ✓ {pname}_{direction}: done over seeds {deriv_seed_start:05d}..{deriv_seed_end:05d}")

    expected_pert = 2 * len(params) * TEST_N_DERIV
    suite_elapsed = time.time() - suite_t0
    ts_print(f"  ✓ perturbed done: written={n_written_pert}, skipped={n_skipped_pert}, expected_total={expected_pert}")
    ts_print(f"  ✓ Suite {sidx:02d} wall time: {suite_elapsed:.1f}s ({suite_elapsed/60:.1f} min)")

    done_marker = os.path.join(suite_dir, "suite_done.txt")
    with open(done_marker, "w") as f:
        f.write(f"completed_at={datetime.now().isoformat()}\n")
        f.write(f"suite_idx={sidx}\n")
        f.write(f"theta={theta_suite.tolist()}\n")
        f.write(f"elapsed_seconds={suite_elapsed:.1f}\n")
        f.write(f"expected_fits={expected_suite_fits}\n")
        f.write(f"fiducial_written={n_written_fid}\n")
        f.write(f"fiducial_skipped={n_skipped_fid}\n")
        f.write(f"perturbed_written={n_written_pert}\n")
        f.write(f"perturbed_skipped={n_skipped_pert}\n")

ts_print("\n" + "="*70)
ts_print("✓ Test-theta suite generation complete")
ts_print(f"Output root: {TEST_SUITES_OUTDIR}")
ts_print("Each suite contains 2000 fiducial + 800 perturbed files.")
ts_print("="*70)
'''  # ===== END COMMENTED OUT: Test theta suites =====


# ============================================================
# LHC EXTRA SAMPLES  (256 new parameter grids)
# ============================================================
# 1. Load lhc_theta_grid.npy (128 points, rows 0..127).
# 2. If it already has 384 rows the previous run already appended
#    the extra 256 points — use rows 128..383 as lhc_theta_extra.
# 3. Otherwise: select 256 new rows from LHC_fit_final_precise.txt
#    that are NOT already in the grid (exact-value match), append
#    them in-place to lhc_theta_grid.npy  →  shape becomes (384, 4).
# 4. Generate 5 realizations per new point, continuing the existing
#    index/seed convention:
#      lhc_idx  = 128, 129, ..., 383
#      seed     = 5215 + lhc_idx * 5 + r,   r ∈ {0,1,2,3,4}
#      filename = lhc_theta{lhc_idx:04d}_seed{seed:05d}.fits
#    Output → same lhc_samples directory; skip files that already exist.
# ============================================================

LHC_THETA_FILE        = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k/lhc_theta_grid.npy"
LHC_DIR_EXTRA         = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k/lhc_samples"
LHC_BASE_SEED_EXTRA   = 5215          # same formula as original block
LHC_N_REALIZ_EXTRA    = 5
LHC_FIRST_NEW_IDX     = 128           # original block used 0..127
LHC_N_NEW_POINTS      = 256
LHC_TOTAL_POINTS      = LHC_FIRST_NEW_IDX + LHC_N_NEW_POINTS   # 384
N_SAMPLES_LHC_EXTRA   = 5000
LHC_EXTRA_SELECT_SEED = 20260304      # reproducible selection

os.makedirs(LHC_DIR_EXTRA, exist_ok=True)

# ---- Step 1: Load existing grid ----------------------------------------
if not os.path.exists(LHC_THETA_FILE):
    raise FileNotFoundError(
        f"LHC theta grid not found: {LHC_THETA_FILE}\n"
        "Run the original LHC block first."
    )
lhc_theta_grid_full = np.load(LHC_THETA_FILE)
print(f"Loaded LHC theta grid: {LHC_THETA_FILE}  shape={lhc_theta_grid_full.shape}")

# Validate uniqueness in existing grid before any append/generation.
grid_keys = [
    tuple(np.round(np.asarray(row, dtype=np.float64), decimals=8).tolist())
    for row in lhc_theta_grid_full
]
if len(set(grid_keys)) != len(grid_keys):
    raise RuntimeError(
        f"Existing grid contains duplicate rows: {len(grid_keys) - len(set(grid_keys))} duplicates. "
        "Please de-duplicate lhc_theta_grid.npy before continuing."
    )

if not (LHC_FIRST_NEW_IDX <= lhc_theta_grid_full.shape[0] <= LHC_TOTAL_POINTS):
    raise RuntimeError(
        f"Unexpected grid size {lhc_theta_grid_full.shape[0]}; "
        f"expected between {LHC_FIRST_NEW_IDX} and {LHC_TOTAL_POINTS}."
    )

# ---- Step 2 / 3: Append 256 new rows if not already done ---------------
if lhc_theta_grid_full.shape[0] == LHC_TOTAL_POINTS:
    print(f"Grid already extended to {LHC_TOTAL_POINTS} rows — using rows "
          f"{LHC_FIRST_NEW_IDX}..{LHC_TOTAL_POINTS - 1} as the extra set.")
    lhc_theta_extra = lhc_theta_grid_full[LHC_FIRST_NEW_IDX:]
else:
    # lhc_theta_grid_full has 128..383 rows; append only the missing tail.
    n_existing = int(lhc_theta_grid_full.shape[0])
    n_needed = LHC_TOTAL_POINTS - n_existing

    all_lhc_params = np.genfromtxt('/g/data/y89/ys9983/LHC_fit_final_precise.txt')[:, -4:]
    print(f"Total rows in LHC_fit_final_precise.txt: {len(all_lhc_params)}")

    existing_grid = np.asarray(lhc_theta_grid_full, dtype=np.float64)
    available_mask = np.array([
        not np.any(np.all(np.isclose(existing_grid, row, rtol=0.0, atol=1e-6), axis=1))
        for row in np.asarray(all_lhc_params, dtype=np.float64)
    ], dtype=bool)
    available_params = all_lhc_params[available_mask]
    print(f"Available (not in existing grid): {len(available_params)} rows")

    if len(available_params) < n_needed:
        raise RuntimeError(
            f"Only {len(available_params)} candidate rows available after excluding "
            f"{lhc_theta_grid_full.shape[0]} existing grid points; need {n_needed}."
        )

    rng_extra = np.random.default_rng(seed=LHC_EXTRA_SELECT_SEED)
    chosen_idx_extra = rng_extra.choice(len(available_params), size=n_needed, replace=False)
    chosen_idx_extra = np.sort(chosen_idx_extra)
    lhc_theta_new = available_params[chosen_idx_extra].astype(np.float32)

    # Verify no overlap before writing (double-check)
    overlap = np.array([
        np.any(np.all(np.isclose(existing_grid, row, rtol=0.0, atol=1e-6), axis=1))
        for row in np.asarray(lhc_theta_new, dtype=np.float64)
    ], dtype=bool)
    if overlap.any():
        raise RuntimeError(f"BUG: {int(overlap.sum())} overlapping rows detected before appending!")

    # Append in-place and overwrite lhc_theta_grid.npy
    lhc_theta_grid_full = np.vstack([lhc_theta_grid_full, lhc_theta_new])
    np.save(LHC_THETA_FILE, lhc_theta_grid_full)

    # Use rows 128..383 as the canonical extra set for generation.
    lhc_theta_extra = lhc_theta_grid_full[LHC_FIRST_NEW_IDX:]

    print(f"\nAppended {n_needed} new rows → lhc_theta_grid.npy now {lhc_theta_grid_full.shape}")
    print(f"  Overlap check   : 0 overlaps (verified)")
    print(f"  Range M_mw  : [{lhc_theta_grid_full[LHC_FIRST_NEW_IDX:,0].min():.3f}, {lhc_theta_grid_full[LHC_FIRST_NEW_IDX:,0].max():.3f}]")
    print(f"  Range M_lmc : [{lhc_theta_grid_full[LHC_FIRST_NEW_IDX:,1].min():.3f}, {lhc_theta_grid_full[LHC_FIRST_NEW_IDX:,1].max():.3f}]")
    print(f"  Range c     : [{lhc_theta_grid_full[LHC_FIRST_NEW_IDX:,2].min():.3f}, {lhc_theta_grid_full[LHC_FIRST_NEW_IDX:,2].max():.3f}]")
    print(f"  Range q     : [{lhc_theta_grid_full[LHC_FIRST_NEW_IDX:,3].min():.3f}, {lhc_theta_grid_full[LHC_FIRST_NEW_IDX:,3].max():.3f}]")

# ---- Step 4: Generate FITS files ----------------------------------------
last_lhc_idx = LHC_FIRST_NEW_IDX + LHC_N_NEW_POINTS - 1
last_seed    = LHC_BASE_SEED_EXTRA + last_lhc_idx * LHC_N_REALIZ_EXTRA + (LHC_N_REALIZ_EXTRA - 1)

print("\n" + "="*60)
print("GENERATING LHC EXTRA TRAINING SAMPLES (256 new grids)")
print("="*60)
print(f"New LHC points     : {LHC_N_NEW_POINTS}")
print(f"Realizations each  : {LHC_N_REALIZ_EXTRA}")
print(f"Samples per file   : {N_SAMPLES_LHC_EXTRA}")
print(f"lhc_idx range      : {LHC_FIRST_NEW_IDX:04d} – {last_lhc_idx:04d}")
print(f"Seed range         : {LHC_BASE_SEED_EXTRA + LHC_FIRST_NEW_IDX * LHC_N_REALIZ_EXTRA:05d}"
      f" – {last_seed:05d}")
print(f"Output directory   : {LHC_DIR_EXTRA}")
print(f"Total files        : {LHC_N_NEW_POINTS * LHC_N_REALIZ_EXTRA}")

n_skipped_extra = 0
n_written_extra = 0

for offset, theta_lhc in enumerate(lhc_theta_extra):
    lhc_idx       = LHC_FIRST_NEW_IDX + offset
    theta_lhc_arr = theta_lhc.astype(np.float32)

    for r in range(LHC_N_REALIZ_EXTRA):
        seed     = LHC_BASE_SEED_EXTRA + lhc_idx * LHC_N_REALIZ_EXTRA + r
        out_path = os.path.join(LHC_DIR_EXTRA,
                                f"lhc_theta{lhc_idx:04d}_seed{seed:05d}.fits")

        if os.path.exists(out_path):
            n_skipped_extra += 1
            continue

        x_lhc = sample_cond(
            model, theta_lhc_arr,
            n_samples=N_SAMPLES_LHC_EXTRA,
            nfe=cfg.ode_nfe,
            seed=seed,
        )
        save_samples_to_fits(x_lhc, out_path, theta_phys=theta_lhc_arr)
        n_written_extra += 1

    if (offset + 1) % 32 == 0 or offset == 0:
        print(f"  [{offset+1:>3d}/{LHC_N_NEW_POINTS}]  lhc_idx={lhc_idx:04d}"
              f"  written={n_written_extra}  skipped={n_skipped_extra}"
              f"  theta={theta_lhc_arr.tolist()}")

print("\n" + "="*60)
print("✓ LHC extra training samples done!")
print(f"  Written  : {n_written_extra}")
print(f"  Skipped  : {n_skipped_extra}  (already existed)")
print(f"  Total    : {n_written_extra + n_skipped_extra} / {LHC_N_NEW_POINTS * LHC_N_REALIZ_EXTRA}")
print(f"  Output   : {LHC_DIR_EXTRA}")
print(f"  Grid     : {LHC_THETA_FILE}  ({lhc_theta_grid_full.shape[0]} total rows)")
print("="*60)


# ============================================================
# ANCHOR PERTURBATION SAMPLES (for multi-point Fisher loss)
# ============================================================
# Select N_ANCHORS well-spaced points from the 384-point LHC grid
# via farthest-point selection in normalised θ-space.
# N_ANCHORS = 20 is sufficient for a 4-D parameter space and keeps
# the total file count (20 × 160 = 3200) proportionate to the LHC set.
#
# For each anchor θ_a, generate:
#   4 params × 2 directions × N_ANCHOR_DERIV realisations
#   at θ_a ± δθ_p  (used for the multi-point Fisher loss).
#
# No per-anchor fiducial samples are needed: the training loss uses the
# global fiducial noise (group-0, N=500) as the noise denominator for
# all anchor SNR terms. This is valid because the decorrelation loss
# normalises encoder output to unit variance at fiducial, so
# Var(t|θ_a) / Var(t|θ_fid) stays within ~2-3× across the prior.
# Any residual bias simply re-weights anchor contributions, not their
# gradient direction. The log(SNR+1) form caps outlier anchors.
#
# Seed arithmetic (disjoint from all earlier pools):
#   ANCHOR_BASE_SEED = 13000
#   anchor_stride    = 2 * N_PARAMS * N_ANCHOR_DERIV  (= 160)
#   For anchor a (0-based index into the selected anchor list):
#     deriv seeds: ANCHOR_BASE_SEED + a * anchor_stride
#                  + pdir_idx * N_ANCHOR_DERIV + r
#       pdir_idx = param_idx * 2 + (0 for plus, 1 for minus)
#
# File naming (all in anchor_samples/ subdirectory):
#   anchor_{param}_{direction}_{anchor_idx:04d}_seed{seed:05d}.fits
#
# Outputs:
#   anchor_samples/               – FITS files
#   anchor_samples/anchor_meta.npz – anchor thetas, indices, seeds, etc.
# ============================================================

LHC_THETA_FILE_ANCHOR = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k/lhc_theta_grid.npy"
ANCHOR_DIR            = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k/anchor_samples"
ANCHOR_BASE_SEED      = 13000       # disjoint from LHC (5215-7134) and old ext (7135-12894)
N_ANCHORS             = 20          # well-spaced anchors via farthest-point (4-D space: 20 is sufficient)
N_ANCHOR_DERIV        = 20          # realisations per perturbation direction
N_SAMPLES_ANCHOR      = 5000        # particles per FITS file
N_PARAMS              = 4
ANCHOR_STRIDE         = 2 * N_PARAMS * N_ANCHOR_DERIV  # = 160 (no per-anchor fiducial needed)

os.makedirs(ANCHOR_DIR, exist_ok=True)

# ---- Load LHC theta grid (384 rows) ------------------------------------
if not os.path.exists(LHC_THETA_FILE_ANCHOR):
    raise FileNotFoundError(
        f"LHC theta grid not found: {LHC_THETA_FILE_ANCHOR}\n"
        "Run the LHC generation blocks first."
    )
lhc_theta_all = np.load(LHC_THETA_FILE_ANCHOR)   # (384, 4)
n_lhc_total = lhc_theta_all.shape[0]

if n_lhc_total < N_ANCHORS:
    raise RuntimeError(
        f"LHC grid has only {n_lhc_total} points, need at least {N_ANCHORS} anchors."
    )


# ---- Farthest-point selection in normalised θ-space --------------------
def select_anchors_farthest_point(theta_grid, n_select, rng_seed=42):
    """
    Greedy farthest-point selection.
    Returns indices (into theta_grid rows) of the n_select chosen anchors.
    """
    # Normalise to [0, 1] per dimension for balanced distance
    lo = theta_grid.min(axis=0)
    hi = theta_grid.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    theta_norm = (theta_grid - lo) / span            # (N, 4) in [0, 1]

    N = theta_norm.shape[0]
    rng = np.random.default_rng(rng_seed)

    # Start from a random point
    selected = [int(rng.integers(N))]
    # Distance from each candidate to the nearest selected point
    min_dist = np.full(N, np.inf)

    for _ in range(n_select - 1):
        last = theta_norm[selected[-1]]
        d = np.linalg.norm(theta_norm - last[None, :], axis=1)
        min_dist = np.minimum(min_dist, d)
        # Pick the candidate farthest from its nearest selected point
        # (exclude already-selected)
        min_dist_masked = min_dist.copy()
        min_dist_masked[selected] = -1.0
        next_idx = int(np.argmax(min_dist_masked))
        selected.append(next_idx)

    return np.array(selected, dtype=np.int64)


anchor_meta_path = os.path.join(ANCHOR_DIR, "anchor_meta.npz")

# Force-regenerate anchor_meta.npz with corrected step_size_abs key
if os.path.exists(anchor_meta_path):
    _meta = np.load(anchor_meta_path)
    anchor_lhc_indices = _meta["anchor_lhc_indices"]
    anchor_thetas = _meta["anchor_thetas"]
    print(f"Loaded existing anchor meta (will overwrite with fixed step_size_abs): {anchor_meta_path}")
    print(f"  {len(anchor_lhc_indices)} anchors")
    if len(anchor_lhc_indices) != N_ANCHORS:
        raise RuntimeError(
            f"Existing anchor_meta.npz has {len(anchor_lhc_indices)} anchors, "
            f"but N_ANCHORS={N_ANCHORS}. Delete anchor_meta.npz to regenerate."
        )
else:
    anchor_lhc_indices = select_anchors_farthest_point(
        lhc_theta_all, N_ANCHORS, rng_seed=42
    )
    anchor_thetas = lhc_theta_all[anchor_lhc_indices].astype(np.float32)
    print(f"Selected {N_ANCHORS} anchors via farthest-point from {n_lhc_total}-point grid")

# Always (re-)write anchor_meta.npz so the key is step_size_abs (not step_frac)
np.savez(
    anchor_meta_path,
    anchor_lhc_indices=anchor_lhc_indices,
    anchor_thetas=anchor_thetas,
    lhc_theta_grid_shape=np.array(lhc_theta_all.shape),
    params=np.array(params),
    step_size_abs=np.array([step_size_abs[p] for p in params], dtype=np.float32),
    n_anchors=np.int64(N_ANCHORS),
    n_anchor_deriv=np.int64(N_ANCHOR_DERIV),
    anchor_base_seed=np.int64(ANCHOR_BASE_SEED),
    anchor_stride=np.int64(ANCHOR_STRIDE),
)
print(f"Overwrote anchor meta with corrected step_size_abs: {anchor_meta_path}")

# ---- Generate FITS files (force-overwrite to fix corrupted samples) -----
expected_files_per_anchor = 2 * N_PARAMS * N_ANCHOR_DERIV  # = 160
expected_total_files = N_ANCHORS * expected_files_per_anchor  # = 3200
last_seed = ANCHOR_BASE_SEED + (N_ANCHORS - 1) * ANCHOR_STRIDE + ANCHOR_STRIDE - 1

print("\n" + "="*60)
print("GENERATING ANCHOR PERTURBATION SAMPLES")
print("="*60)
print(f"Anchors            : {N_ANCHORS}")
print(f"Deriv per dir      : {N_ANCHOR_DERIV}")
print(f"Files per anchor   : {expected_files_per_anchor}")
print(f"Total files        : {expected_total_files}")
print(f"Samples per file   : {N_SAMPLES_ANCHOR}")
print(f"Seed range         : {ANCHOR_BASE_SEED:05d} – {last_seed:05d}")
print(f"Output directory   : {ANCHOR_DIR}")

n_written_anchor = 0
n_skipped_anchor = 0
t0_anchor = time.time()

for a_idx in range(N_ANCHORS):
    theta_anchor = anchor_thetas[a_idx]
    theta_anchor_arr = theta_anchor.astype(np.float32)
    base_seed_a = ANCHOR_BASE_SEED + a_idx * ANCHOR_STRIDE

    # Perturbed realisations: 4 params × 2 dirs × N_ANCHOR_DERIV
    # Global fiducial noise (group-0) used as denominator in training loss.
    for pidx, pname in enumerate(params):
        dp = step_size_abs[pname]  # absolute step, NOT fractional

        for dir_idx, (direction, sign) in enumerate([("plus", +1.0), ("minus", -1.0)]):
            theta_pert = theta_anchor_arr.copy()
            theta_pert[pidx] += sign * dp
            pdir_idx = pidx * 2 + dir_idx

            for r in range(N_ANCHOR_DERIV):
                seed = base_seed_a + pdir_idx * N_ANCHOR_DERIV + r
                out_path = os.path.join(
                    ANCHOR_DIR,
                    f"anchor_{pname}_{direction}_{a_idx:04d}_seed{seed:05d}.fits",
                )
                # Force-overwrite: old files used wrong fractional step sizes
                x = sample_cond(
                    model, theta_pert,
                    n_samples=N_SAMPLES_ANCHOR,
                    nfe=cfg.ode_nfe,
                    seed=seed,
                )
                save_samples_to_fits(x, out_path, theta_phys=theta_pert)
                n_written_anchor += 1

    if (a_idx + 1) % 10 == 0 or a_idx == 0:
        elapsed = time.time() - t0_anchor
        rate = (n_written_anchor + n_skipped_anchor) / max(elapsed, 1e-6)
        print(
            f"  [{a_idx+1:>3d}/{N_ANCHORS}]  "
            f"written={n_written_anchor}  skipped={n_skipped_anchor}  "
            f"rate={rate:.1f} files/s  "
            f"theta={theta_anchor_arr.tolist()}"
        )

elapsed_total = time.time() - t0_anchor
print("\n" + "="*60)
print("\u2713 Anchor perturbation samples done!")
print(f"  Written  : {n_written_anchor}")
print(f"  Skipped  : {n_skipped_anchor}  (already existed)")
print(f"  Total    : {n_written_anchor + n_skipped_anchor} / {expected_total_files}")
print(f"  Output   : {ANCHOR_DIR}")
print(f"  Meta     : {anchor_meta_path}")
print(f"  Wall time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
print("="*60)


# ============================================================
# FISHER SAMPLES: REGENERATE SINGLE-PARAM PERTURBED SAMPLES
# ============================================================
# Regenerate the 4000 single-param perturbed samples in fisher_samples/
# with corrected absolute step sizes (step_size_abs, not fractional).
#
# Seed convention (preserved from original generation):
#   Block 1 (older runs): seeds 2015..2314  (300 seeds)
#   Block 2 (newer runs): seeds 5015..5214  (200 seeds)
# All 8 param/direction combos share the same 500-seed pool.
#
# File naming : {param}_{direction}_seed{seed:05d}.fits
# Theta       : fiducial ± step_size_abs[param]
# ============================================================

FISHER_OUTDIR       = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k/fisher_samples"
FISHER_THETA_FID    = np.array([0.7, 15.0, 9.415, 1.0], dtype=np.float32)
FISHER_SEEDS        = list(range(2015, 2315)) + list(range(5015, 5215))  # 300 + 200 = 500
FISHER_N_SAMPLES    = 5000

os.makedirs(FISHER_OUTDIR, exist_ok=True)

expected_fisher_total = 2 * len(params) * len(FISHER_SEEDS)  # = 4000

print("\n" + "="*60)
print("REGENERATING FISHER SINGLE-PARAM PERTURBED SAMPLES")
print("="*60)
print(f"Params × dirs      : {len(params)} × 2 = {2*len(params)}")
print(f"Seeds per direction: {len(FISHER_SEEDS)}  (2015-2314 + 5015-5214)")
print(f"Total files        : {expected_fisher_total}")
print(f"Samples per file   : {FISHER_N_SAMPLES}")
print(f"Output directory   : {FISHER_OUTDIR}")

n_written_fisher = 0
t0_fisher = time.time()

for pidx, pname in enumerate(params):
    for direction, sign in [("plus", +1.0), ("minus", -1.0)]:
        theta_pert = FISHER_THETA_FID.copy()
        theta_pert[pidx] += sign * step_size_abs[pname]

        for seed in FISHER_SEEDS:
            out_path = os.path.join(FISHER_OUTDIR,
                                    f"{pname}_{direction}_seed{seed:05d}.fits")
            # Force-overwrite: old files used wrong fractional step sizes
            x = sample_cond(model, theta_pert,
                            n_samples=FISHER_N_SAMPLES,
                            nfe=cfg.ode_nfe,
                            seed=seed)
            save_samples_to_fits(x, out_path, theta_phys=theta_pert)
            n_written_fisher += 1

        elapsed = time.time() - t0_fisher
        print(f"  \u2713 {pname}_{direction}: 500 files  "
              f"(total written={n_written_fisher}  elapsed={elapsed:.0f}s)")

elapsed_fisher = time.time() - t0_fisher
print("\n" + "="*60)
print("\u2713 Fisher single-param samples regenerated!")
print(f"  Written  : {n_written_fisher} / {expected_fisher_total}")
print(f"  Wall time: {elapsed_fisher:.1f}s ({elapsed_fisher/60:.1f} min)")
print("="*60)


# ============================================================
# TEST THETA SUITES: REGENERATE CORRUPTED PERTURBED SAMPLES
# ============================================================
# The fiducial files in each suite are at theta_suite (unperturbed) —
# those are fine.  Only the perturbed files used the fractional dp bug.
#
# For suite sidx:
#   theta_suite    loaded from suite_meta.npz
#   deriv seeds    deriv_seed_start .. deriv_seed_end  (100 seeds, shared
#                  across all 8 param/dir combos)
#   File naming  : {param}_{direction}_seed{seed:05d}.fits
#   theta_pert   : theta_suite[pidx] ± step_size_abs[pname]  (absolute)
# ============================================================

TEST_SUITES_ROOT    = "/g/data/y89/ys9983/CFM_MW_LMC/mcmc_runs/BFE_samples_5k/test_theta_suites"
TEST_N_SUITES_REGEN = 10
TEST_N_SAMPLES      = 5000

print("\n" + "="*60)
print("REGENERATING TEST SUITE PERTURBED SAMPLES (suites 000-009)")
print("="*60)

n_written_suites = 0
t0_suites = time.time()

for sidx in range(TEST_N_SUITES_REGEN):
    suite_dir  = os.path.join(TEST_SUITES_ROOT, f"suite_{sidx:03d}")
    meta_path  = os.path.join(suite_dir, "suite_meta.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"suite_meta.npz missing: {meta_path}")

    meta            = np.load(meta_path)
    theta_suite     = meta["theta_suite"].astype(np.float32)
    deriv_seed_start = int(meta["deriv_seed_start"])
    deriv_seed_end   = int(meta["deriv_seed_end"])
    deriv_seeds      = list(range(deriv_seed_start, deriv_seed_end + 1))

    # Invalidate stale BFE summary cache computed from corrupted FITS
    cache_path = os.path.join(suite_dir, "compressed_summaries_cache.npz")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"  Deleted stale cache: {cache_path}")

    suite_written = 0
    for pidx, pname in enumerate(params):
        for direction, sign in [("plus", +1.0), ("minus", -1.0)]:
            theta_pert = theta_suite.copy()
            theta_pert[pidx] += sign * step_size_abs[pname]

            for seed in deriv_seeds:
                out_path = os.path.join(suite_dir,
                                        f"{pname}_{direction}_seed{seed:05d}.fits")
                # Force-overwrite: old files used wrong fractional step sizes
                x = sample_cond(model, theta_pert,
                                n_samples=TEST_N_SAMPLES,
                                nfe=cfg.ode_nfe,
                                seed=seed)
                save_samples_to_fits(x, out_path, theta_phys=theta_pert)
                suite_written += 1
                n_written_suites += 1

    elapsed = time.time() - t0_suites
    print(f"  suite_{sidx:03d}: {suite_written} files written  "
          f"theta={theta_suite.tolist()}  elapsed={elapsed:.0f}s")

elapsed_suites = time.time() - t0_suites
print("\n" + "="*60)
print("\u2713 Test suite perturbed samples regenerated!")
print(f"  Written  : {n_written_suites}")
print(f"  Wall time: {elapsed_suites:.1f}s ({elapsed_suites/60:.1f} min)")
print("="*60)
