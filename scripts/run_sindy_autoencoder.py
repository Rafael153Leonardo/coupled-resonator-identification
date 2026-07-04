"""E2 — SINDy autoencoder (Champion, Lusch, Brunton & Kutz, PNAS 2019).

Jointly learns coordinates AND a sparse polynomial dynamics model: a
near-identity encoder/decoder wraps the rank-4 latents, and a trainable
coefficient matrix Xi advances the encoded state through a cubic library,

    y_{k+1} = Theta(y_k) Xi,

with L1 regularization plus sequential hard-thresholding (the SINDy sparsity
mechanism). Adapted to *discrete time* to keep the project's
no-differentiation hygiene; the linear block of Xi is warm-started at the DMD
propagator, so E2 literally stands on the previous rungs of the ladder.

The exam that E0 failed: with learnable coordinates, does the sparse cubic map
now reproduce the measured mode-2 softening on held-out records?

Requires the "learned" extra: pip install -e ".[learned]"
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coupled_id import (
    bandpass,
    decimate_to,
    detect_release_events,
    identify_modal,
    instantaneous_ridge,
    load_scope_csv,
)

DATA = ROOT / "data"
FIGURES = ROOT / "figures"

TRAIN_RECORDS = [11, 12, 13, 14, 15]
TEST_RECORDS = [23, 24]
TARGET_FS = 500.0
DELAYS = 150
RANK = 4
DT = 1.0 / TARGET_FS
BACKBONE_BAND_2 = (37.6, 42.0)

SEED = 7
TRAIN_STEPS = 6000
LR = 1e-3
LR_DROP_STEP = 4000
BATCH = 256
# curriculum: short horizons while the map is raw, long ones (which teach the
# amplitude dependence) only after it is locally accurate — iterated cubic
# maps blow up in training otherwise
HORIZONS_WARMUP = (1, 2, 4, 8)
HORIZONS_MAIN = (1, 4, 16, 32)
CURRICULUM_STEP = 2500
L1_WEIGHT = 1e-5
THRESHOLD = 5e-4  # sequential-thresholding cutoff on |Xi|
THRESHOLD_EVERY = 1000  # steps between thresholding passes (after warm-up)
THRESHOLD_START = 2000

MONOMIALS = (
    [(i,) for i in range(4)]
    + [(i, j) for i in range(4) for j in range(i, 4)]
    + [(i, j, k) for i in range(4) for j in range(i, 4) for k in range(j, 4)]
)
FEATURE_NAMES = ["*".join(f"y{i}" for i in mono) for mono in MONOMIALS]


def collect(records: list[int]) -> list[np.ndarray]:
    """Rank-4 latent trajectories (T x 4) of every long decay."""

    out = []
    for record in records:
        t, x = load_scope_csv(DATA / f"scope_{record}_1.csv")
        x = x - np.mean(x)
        for start, stop in detect_release_events(t, x):
            t_ev = t[start:stop] - t[start]
            if float(t_ev[-1]) < 8.0:
                continue
            t_d, x_d = decimate_to(t_ev, x[start:stop], TARGET_FS)
            out.append(identify_modal(t_d, x_d, delays=DELAYS, rank=RANK).z.T)
    return out


def dmd_warm_start(train_std: list[np.ndarray]) -> np.ndarray:
    """Least-squares one-step propagator over all training pairs (4 x 4)."""

    x_all = np.concatenate([z[:-1] for z in train_std])
    y_all = np.concatenate([z[1:] for z in train_std])
    return np.linalg.lstsq(x_all, y_all, rcond=None)[0]  # y_next = y @ M


def mode2_ridge(t: np.ndarray, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(amplitude, frequency) ridge of the component with most mode-2 energy."""

    best, best_energy = None, -1.0
    for component in range(trajectory.shape[1]):
        x_band = bandpass(t, trajectory[:, component], *BACKBONE_BAND_2)
        energy = float(np.sum(x_band**2))
        if energy > best_energy:
            best, best_energy = x_band, energy
    freq, amp = instantaneous_ridge(t, best)
    edge = int(round(0.5 / DT))
    freq, amp = freq[edge:-edge], amp[edge:-edge]
    keep = amp > 0.12 * float(np.max(amp))
    return amp[keep], freq[keep]


def main() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("E2 requires torch: pip install -e '.[learned]'")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    nn = torch.nn

    train = collect(TRAIN_RECORDS)
    test = collect(TEST_RECORDS)
    scale = np.concatenate(train).std(axis=0)
    train_std = [z / scale for z in train]
    test_std = [z / scale for z in test]
    print(f"train decays: {len(train)} | test decays: {len(test)}")

    def poly_features(y):
        feats = [y]
        feats += [y[..., i : i + 1] * y[..., j : j + 1] for i in range(4) for j in range(i, 4)]
        feats += [
            y[..., i : i + 1] * y[..., j : j + 1] * y[..., k : k + 1]
            for i in range(4)
            for j in range(i, 4)
            for k in range(j, 4)
        ]
        return torch.cat(feats, dim=-1)  # (..., 34)

    class SindyAE(nn.Module):
        def __init__(self, m_init: np.ndarray) -> None:
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 4))
            self.dec = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 4))
            for net in (self.enc, self.dec):
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
            xi0 = np.zeros((len(MONOMIALS), 4), dtype=np.float32)
            xi0[:4] = m_init.astype(np.float32)
            self.xi = nn.Parameter(torch.tensor(xi0))
            self.register_buffer("mask", torch.ones_like(self.xi))

        def encode(self, z):
            return z + self.enc(z)

        def decode(self, y):
            return y + self.dec(y)

        def step(self, y):
            return poly_features(y) @ (self.xi * self.mask)

        def threshold(self, cutoff: float) -> int:
            with torch.no_grad():
                self.mask *= (self.xi.abs() >= cutoff).float()
                self.xi *= self.mask
            return int(self.mask.sum())

    model = SindyAE(dmd_warm_start(train_std))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    sequences = [torch.tensor(z, dtype=torch.float32) for z in train_std]
    max_h = max(HORIZONS_MAIN)
    skipped = 0

    for step in range(TRAIN_STEPS):
        horizons = HORIZONS_WARMUP if step < CURRICULUM_STEP else HORIZONS_MAIN
        if step == LR_DROP_STEP:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        if step >= THRESHOLD_START and step % THRESHOLD_EVERY == 0:
            active = model.threshold(THRESHOLD)
            print(f"step {step:5d}  thresholding -> {active} active terms")
        seq = sequences[rng.integers(len(sequences))]
        idx = torch.tensor(rng.integers(0, len(seq) - max_h - 1, size=BATCH))
        z0 = seq[idx]
        optimizer.zero_grad()
        y = model.encode(z0)
        loss = mse(model.decode(y), z0)
        prev_h = 0
        for h in horizons:
            for _ in range(h - prev_h):
                y = model.step(y)
            prev_h = h
            z_h = seq[idx + h]
            loss = loss + mse(model.decode(y), z_h) + 0.5 * mse(y, model.encode(z_h))
        loss = loss + L1_WEIGHT * (model.xi * model.mask).abs().sum()
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == TRAIN_STEPS - 1:
            print(f"step {step:5d}  loss {float(loss.detach()):.5f}")
    if skipped:
        print(f"(skipped {skipped} non-finite batches)")

    # ------------------------------------------------------------------
    # Discovered model: sparsity and linear-part spectrum
    # ------------------------------------------------------------------
    xi = (model.xi * model.mask).detach().numpy()
    active = int(np.count_nonzero(np.abs(xi) > 1e-6))
    print()
    print(f"discovered model: {active} active terms / {xi.size}")
    eigenvalues = np.log(np.linalg.eigvals(xi[:4].T).astype(complex)) / DT
    for lam in eigenvalues:
        if lam.imag > 0:
            print(f"  linear part: f = {lam.imag / (2 * np.pi):8.4f} Hz   zeta = {-lam.real / abs(lam):.5f}")
    print("  strongest nonlinear terms per equation:")
    nonlinear = np.abs(xi[4:])
    for equation in range(4):
        order = np.argsort(nonlinear[:, equation])[::-1][:2]
        terms = ", ".join(
            f"{xi[4 + i, equation]:+.2e} {FEATURE_NAMES[4 + i]}" for i in order if nonlinear[i, equation] > 1e-6
        )
        print(f"    y{equation}' ~ ... {terms}")

    # ------------------------------------------------------------------
    # EXAM: held-out rollout must reproduce the measured mode-2 backbone
    # ------------------------------------------------------------------
    z_test = test_std[0]
    n_steps = min(len(z_test), int(10.0 / DT))
    with torch.no_grad():
        y = model.encode(torch.tensor(z_test[0:1], dtype=torch.float32))
        rollout = [model.decode(y).numpy()[0]]
        for _ in range(n_steps - 1):
            y = model.step(y)
            if not torch.isfinite(y).all() or float(y.abs().max()) > 1e3:
                break
            rollout.append(model.decode(y).numpy()[0])
    simulated = np.array(rollout)
    t_sim = np.arange(len(simulated)) * DT
    print()
    print(f"EXAM - held-out rollout: {len(simulated)} steps ({len(simulated) * DT:.1f} s) before any blow-up")

    amp_sim, freq_sim = mode2_ridge(t_sim, simulated * scale)
    amp_meas, freq_meas = mode2_ridge(t_sim, z_test[: len(simulated)] * scale)
    softening_sim = float(np.percentile(freq_sim, 95) - np.percentile(freq_sim, 5))
    softening_meas = float(np.percentile(freq_meas, 95) - np.percentile(freq_meas, 5))
    print(
        f"  mode-2 frequency swing: simulated {softening_sim:.2f} Hz vs measured {softening_meas:.2f} Hz "
        f"({100 * softening_sim / softening_meas:.0f}% captured; E0 captured ~15%)"
    )

    fig, (ax_bb, ax_xi) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]})
    ax_bb.plot(amp_meas, freq_meas, ".", ms=2, alpha=0.35, color="tab:orange", label="measured (held-out decay)")
    ax_bb.plot(amp_sim, freq_sim, ".", ms=2, alpha=0.35, color="tab:purple", label="SINDy-AE rollout (E2)")
    ax_bb.set_xlabel("Mode-2 amplitude [latent units]")
    ax_bb.set_ylabel("Instantaneous frequency [Hz]")
    ax_bb.set_title("E2 exam: learned coordinates + sparse cubic map\nvs measured softening")
    ax_bb.legend()
    ax_bb.grid(True, alpha=0.3)

    magnitude = np.abs(xi)
    ax_xi.imshow(np.log10(magnitude + 1e-8), aspect="auto", cmap="viridis")
    ax_xi.set_xticks(range(4), [f"y{i}'" for i in range(4)])
    ax_xi.set_yticks([0, 4, 14], ["linear", "quadratic", "cubic"])
    ax_xi.set_title(f"Discovered coefficients |Xi| (log10)\n{active} active of {xi.size}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "e2_sindy_ae.png", dpi=160)
    plt.close(fig)
    print(f"figure: {FIGURES / 'e2_sindy_ae.png'}")


if __name__ == "__main__":
    main()
