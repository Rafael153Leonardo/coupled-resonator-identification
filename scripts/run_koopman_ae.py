"""E1 — parametrized Koopman autoencoder (Lusch, Kutz & Brunton 2018).

Learns coordinates in which the coupled dynamics advance by two
rotation-scaling pairs whose decay and frequency are *functions of the pair
radius* — the paper's construction for continuous Koopman spectra, whose
flagship example (a pendulum with amplitude-dependent frequency) matches our
softening mode 2 exactly.

Exam (docs/learned-models-plan.md): on records never seen in training, the
learned omega_2(amplitude) must overlay the measured Hilbert-ridge backbone
while omega_1 stays flat at ~36.36 Hz. E0 (discrete SINDy) captured only a
fraction of the softening; this architecture encodes it natively.

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

# warm-start carrier frequencies (Hz) for the two latent pairs. SVD orders the
# latent pairs by energy and mode 2 dominates these records (A2/A1 > 1), so
# pair 0 starts at the mode-2 carrier. Initialization only — the radius
# dependence is entirely learned.
OMEGA0_HZ = (39.2, 36.36)

SEED = 7
TRAIN_STEPS = 6000
LR_DROP_STEP = 4000
BATCH = 256
HORIZONS = (1, 4, 16, 32)


def collect(records: list[int]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(t, x, z) per long decay: time, decimated signal, rank-4 latents (T x 4)."""

    out = []
    for record in records:
        t, x = load_scope_csv(DATA / f"scope_{record}_1.csv")
        x = x - np.mean(x)
        for start, stop in detect_release_events(t, x):
            t_ev = t[start:stop] - t[start]
            if float(t_ev[-1]) < 8.0:
                continue
            t_d, x_d = decimate_to(t_ev, x[start:stop], TARGET_FS)
            result = identify_modal(t_d, x_d, delays=DELAYS, rank=RANK)
            out.append((t_d[: result.z.shape[1]], x_d[: result.z.shape[1]], result.z.T))
    return out


def build_model(torch, scale: np.ndarray):
    nn = torch.nn

    class KoopmanAE(nn.Module):
        """Near-identity autoencoder + two radius-parametrized rotation pairs."""

        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 4))
            self.dec = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 4))
            self.aux = nn.ModuleList([nn.Sequential(nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 2)) for _ in range(2)])
            for net in (self.enc, self.dec):
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
            for aux in self.aux:
                nn.init.zeros_(aux[-1].weight)
                nn.init.zeros_(aux[-1].bias)
            self.register_buffer("omega0", torch.tensor([2 * np.pi * f for f in OMEGA0_HZ], dtype=torch.float32))

        def encode(self, z):
            return z + self.enc(z)

        def decode(self, y):
            return y + self.dec(y)

        def pair_lambda(self, y, i: int):
            """(mu, omega) of pair i as functions of its radius."""

            pair = y[..., 2 * i : 2 * i + 2]
            radius = pair.norm(dim=-1, keepdim=True)
            mu_domega = self.aux[i](radius)
            mu = mu_domega[..., :1]
            omega = self.omega0[i] + mu_domega[..., 1:]
            return mu, omega

        def advance(self, y, steps: int):
            for _ in range(steps):
                new_pairs = []
                for i in range(2):
                    pair = y[..., 2 * i : 2 * i + 2]
                    mu, omega = self.pair_lambda(y, i)
                    scale_f = torch.exp(mu * DT)
                    theta = omega * DT
                    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                    y1 = pair[..., :1]
                    y2 = pair[..., 1:]
                    new_pairs.append(
                        torch.cat(
                            [scale_f * (cos_t * y1 - sin_t * y2), scale_f * (sin_t * y1 + cos_t * y2)],
                            dim=-1,
                        )
                    )
                y = torch.cat(new_pairs, dim=-1)
            return y

    return KoopmanAE()


def main() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("E1 requires torch: pip install -e '.[learned]'")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    train = collect(TRAIN_RECORDS)
    test = collect(TEST_RECORDS)
    scale = np.concatenate([z for _, _, z in train]).std(axis=0)
    print(f"train decays: {len(train)} | test decays: {len(test)}")

    model = build_model(torch, scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()

    sequences = [torch.tensor(z / scale, dtype=torch.float32) for _, _, z in train]
    max_h = max(HORIZONS)

    for step in range(TRAIN_STEPS):
        if step == LR_DROP_STEP:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        seq = sequences[rng.integers(len(sequences))]
        idx = torch.tensor(rng.integers(0, len(seq) - max_h - 1, size=BATCH))
        z0 = seq[idx]
        optimizer.zero_grad()
        y0 = model.encode(z0)
        loss = mse(model.decode(y0), z0)
        y = y0
        prev_h = 0
        for h in HORIZONS:
            y = model.advance(y, h - prev_h)
            prev_h = h
            z_h = seq[idx + h]
            loss = loss + mse(model.decode(y), z_h) + 0.5 * mse(y, model.encode(z_h))
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == TRAIN_STEPS - 1:
            print(f"step {step:5d}  loss {float(loss.detach()):.5f}")

    # ------------------------------------------------------------------
    # EXAM: learned omega_i(radius) vs the measured backbone, on held-out data
    # (pair 0 = mode 2, the energy-dominant SVD pair; pair 1 = mode 1)
    # ------------------------------------------------------------------
    def evaluate(t_d, x_d, z_test):
        with torch.no_grad():
            y_test = model.encode(torch.tensor(z_test / scale, dtype=torch.float32))
            _, omega2 = model.pair_lambda(y_test, 0)
            _, omega1 = model.pair_lambda(y_test, 1)
        f2_learned = omega2.numpy().ravel() / (2 * np.pi)
        f1_learned = omega1.numpy().ravel() / (2 * np.pi)
        x_band = bandpass(t_d, x_d, *BACKBONE_BAND_2)
        f2_meas, a2_meas = instantaneous_ridge(t_d, x_band)
        edge = int(0.5 / DT)
        sl = slice(edge, -edge)
        keep = a2_meas[sl] > 0.12 * float(np.max(a2_meas[sl]))
        return (
            t_d[sl][keep],
            a2_meas[sl][keep],
            f2_meas[sl][keep],
            f2_learned[sl][keep],
            f1_learned[sl][keep],
        )

    print()
    print("EXAM - held-out records (learned vs measured mode-2 frequency):")
    all_corr, all_rmse, all_f1 = [], [], []
    for t_d, x_d, z_test in test:
        tt, a2, f2_m, f2_l, f1_l = evaluate(t_d, x_d, z_test)
        all_corr.append(float(np.corrcoef(f2_l, f2_m)[0, 1]))
        all_rmse.append(float(np.sqrt(np.mean((f2_l - f2_m) ** 2))))
        all_f1.append(f1_l)
    f1_all = np.concatenate(all_f1)
    print(f"  corr = {np.mean(all_corr):.3f} +- {np.std(all_corr):.3f} over {len(test)} decays")
    print(f"  RMSE = {np.mean(all_rmse):.3f} +- {np.std(all_rmse):.3f} Hz")
    print(f"  learned f1 = {f1_all.mean():.4f} Hz (flatness std {f1_all.std():.4f}; ref 36.36)")

    tt, a2_meas_k, f2_meas_k, f2_learned_k, f1_learned_k = evaluate(*test[0])
    f2_meas, f2_learned, f1_learned = f2_meas_k, f2_learned_k, f1_learned_k

    fig, (ax_bb, ax_t) = plt.subplots(1, 2, figsize=(13, 5))
    ax_bb.plot(a2_meas_k, f2_meas, ".", ms=2, alpha=0.3, color="tab:orange", label="measured ridge (held-out)")
    ax_bb.plot(a2_meas_k, f2_learned, ".", ms=2, alpha=0.3, color="tab:blue", label="Koopman AE $\\omega_2(r)/2\\pi$")
    ax_bb.set_xlabel("Measured mode-2 amplitude [V]")
    ax_bb.set_ylabel("Frequency [Hz]")
    ax_bb.set_title("E1 exam: learned eigenvalue-vs-amplitude\nvs measured backbone")
    ax_bb.legend()
    ax_bb.grid(True, alpha=0.3)

    ax_t.plot(tt, f2_meas, color="tab:orange", lw=1.0, label="measured")
    ax_t.plot(tt, f2_learned, color="tab:blue", lw=1.0, label="learned")
    ax_t.plot(tt, f1_learned, color="tab:green", lw=1.0, label="learned mode 1")
    ax_t.axhline(36.36, color="gray", ls=":", lw=1)
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel("Frequency [Hz]")
    ax_t.set_title("Instantaneous frequencies through the decay")
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "e1_koopman_backbone.png", dpi=160)
    plt.close(fig)
    print(f"figure: {FIGURES / 'e1_koopman_backbone.png'}")


if __name__ == "__main__":
    main()
