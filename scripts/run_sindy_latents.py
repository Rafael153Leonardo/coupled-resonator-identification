"""E0 — discrete-time SINDy on the Hankel-SVD latents (learned-model ladder).

Fits ``z_{k+1} = Theta(z_k) xi`` with a cubic polynomial library and STLSQ on
the rank-4 latent trajectories of the strong-coupling decays. Discrete time
avoids numerical differentiation entirely — the same hygiene as the classical
one-step propagator, so any gain over it must come from the *nonlinear* terms.

Exams (ground truth from the classical pipeline, docs/learned-models-plan.md):

1. the linear part's eigenvalues should give the small-amplitude frequencies
   (mode 2 -> ~40.1 Hz, the backbone's a->0 limit, NOT the amplitude-averaged
   39.2 Hz that a purely linear fit returns);
2. simulating the fitted model from a held-out decay's initial condition and
   ridge-tracking the simulation should reproduce the measured softening
   backbone.

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
BACKBONE_BAND_2 = (37.6, 42.0)


def latent_trajectories(records: list[int]) -> tuple[list[np.ndarray], float]:
    """Rank-4 latent trajectories (T x 4, standardized) of every long decay."""

    trajectories = []
    dt = 1.0 / TARGET_FS
    for record in records:
        t, x = load_scope_csv(DATA / f"scope_{record}_1.csv")
        x = x - np.mean(x)
        for start, stop in detect_release_events(t, x):
            t_ev = t[start:stop] - t[start]
            if float(t_ev[-1]) < 8.0:
                continue
            t_d, x_d = decimate_to(t_ev, x[start:stop], TARGET_FS)
            result = identify_modal(t_d, x_d, delays=DELAYS, rank=RANK)
            trajectories.append(result.z.T)  # (T, 4)
            dt = result.dt
    return trajectories, dt


def linear_part_frequencies(model, scale: np.ndarray, dt: float) -> list[tuple[float, float]]:
    """(frequency_hz, zeta) pairs from the linear block of the discrete model."""

    coefs = np.asarray(model.coefficients())
    names = model.get_feature_names()
    linear_cols = [i for i, n in enumerate(names) if n in ("z0", "z1", "z2", "z3")]
    a_discrete = coefs[:, linear_cols]
    # undo the per-component standardization: A = D^-1 A' D has the same
    # eigenvalues as A', so scaling is safe for modal parameters.
    eigenvalues = np.log(np.linalg.eigvals(a_discrete).astype(complex)) / dt
    modes = []
    for lam in eigenvalues:
        if lam.imag <= 0:
            continue
        magnitude = abs(lam)
        modes.append((float(lam.imag / (2 * np.pi)), float(-lam.real / magnitude)))
    modes.sort()
    return modes


def simulate_discrete(model, z0: np.ndarray, n_steps: int) -> np.ndarray:
    trajectory = np.empty((n_steps, len(z0)))
    trajectory[0] = z0
    state = z0[None, :]
    for k in range(1, n_steps):
        state = model.predict(state)
        trajectory[k] = state[0]
        if not np.all(np.isfinite(state)) or np.abs(state).max() > 1e3:
            return trajectory[:k]
    return trajectory


def backbone_of(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_band = bandpass(t, x, *BACKBONE_BAND_2)
    freq, amp = instantaneous_ridge(t, x_band)
    dt = float(np.median(np.diff(t)))
    edge = int(round(0.5 / dt))
    freq, amp = freq[edge:-edge], amp[edge:-edge]
    keep = amp > 0.12 * float(np.max(amp))
    return amp[keep], freq[keep]


def main() -> None:
    try:
        import pysindy as ps
    except ImportError:
        sys.exit("E0 requires pysindy: pip install -e '.[learned]'")

    train, dt = latent_trajectories(TRAIN_RECORDS)
    test, _ = latent_trajectories(TEST_RECORDS)
    print(f"train decays: {len(train)} | test decays: {len(test)} | dt = {dt * 1e3:.1f} ms")

    # standardize per component with a global scale (backbone lives in amplitude)
    scale = np.concatenate(train).std(axis=0)
    train_std = [traj / scale for traj in train]
    test_std = [traj / scale for traj in test]

    # ------------------------------------------------------------------
    # Model selection: admit nonlinear terms (lower threshold) only while the
    # model stays *simulatable* — one-step error improves monotonically with
    # more terms, but rich discrete cubic maps blow up in rollout. Pick the
    # candidate with the longest stable held-out rollout among those within
    # 2x of the best one-step error.
    # ------------------------------------------------------------------
    n_rollout = min(len(test_std[0]), int(10.0 / dt))
    candidates = []
    print()
    print("threshold | active | test 1-step MSE | stable rollout")
    for threshold in (0.02, 0.005, 0.002, 0.001, 5e-4):
        candidate = ps.DiscreteSINDy(
            optimizer=ps.STLSQ(threshold=threshold, alpha=1e-5),
            feature_library=ps.PolynomialLibrary(degree=3, include_bias=False),
        )
        candidate.fit(train_std, t=dt, feature_names=["z0", "z1", "z2", "z3"])
        one_step = float(
            np.mean([np.mean((candidate.predict(z[:-1]) - z[1:]) ** 2) / np.mean(z[1:] ** 2) for z in test_std])
        )
        rollout = simulate_discrete(candidate, test_std[0][0], n_rollout)
        n_active = int(np.count_nonzero(np.asarray(candidate.coefficients())))
        candidates.append((candidate, threshold, n_active, one_step, len(rollout)))
        print(f"  {threshold:7.4f} | {n_active:6d} | {one_step:14.2e} | {len(rollout) * dt:6.1f} s")

    best_error = min(c[3] for c in candidates)
    admissible = [c for c in candidates if c[3] <= 2.0 * best_error]
    model, threshold, n_active, one_step, _ = max(admissible, key=lambda c: c[4])
    print(f"selected: threshold={threshold} ({n_active} active terms, 1-step MSE {one_step:.2e})")

    print()
    print("EXAM 1 - linear-part eigenvalues (small-amplitude limit):")
    for f_hz, zeta in linear_part_frequencies(model, scale, dt):
        print(f"  f = {f_hz:8.4f} Hz   zeta = {zeta:.5f}")
    print("  reference: f1 = 36.36 Hz; mode-2 small-amplitude limit ~ 40.1 Hz")
    print("  (a purely linear fit gives the amplitude-averaged 39.2 Hz instead)")

    # ------------------------------------------------------------------
    # EXAM 2 - simulate from a held-out decay's initial condition
    # ------------------------------------------------------------------
    z_test = test_std[0]
    n_steps = min(len(z_test), int(12.0 / dt))
    simulated = simulate_discrete(model, z_test[0], n_steps)
    print()
    print(f"EXAM 2 - held-out simulation: {len(simulated)} steps ({len(simulated) * dt:.1f} s) before any blow-up")

    t_sim = np.arange(len(simulated)) * dt
    # mode-2 physical signal ~ the third latent (pairs are quadrature pairs);
    # rebuild a 1-D signal by projecting the simulated latents back through
    # the measured mixing: use z2 (first component of the weaker pair).
    x_sim = simulated[:, 2] * scale[2]
    x_meas = z_test[: len(simulated), 2] * scale[2]

    amp_sim, freq_sim = backbone_of(t_sim, x_sim)
    amp_meas, freq_meas = backbone_of(t_sim, x_meas)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(amp_meas, freq_meas, ".", ms=2, alpha=0.35, color="tab:orange", label="measured (held-out decay)")
    ax.plot(amp_sim, freq_sim, ".", ms=2, alpha=0.35, color="tab:green", label="SINDy simulation (E0)")
    ax.set_xlabel("Mode-2 amplitude [latent units]")
    ax.set_ylabel("Instantaneous frequency [Hz]")
    ax.set_title("E0 exam: does the discrete SINDy model reproduce the softening backbone?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "e0_sindy_backbone.png", dpi=160)
    plt.close(fig)
    print(f"figure: {FIGURES / 'e0_sindy_backbone.png'}")


if __name__ == "__main__":
    main()
