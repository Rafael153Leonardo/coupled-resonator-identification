"""Latent modal identification from a single measured channel.

The key idea: a delay (Hankel) embedding of one channel makes the *unmeasured*
partner of a coupled pair observable (Takens); a one-step propagator fitted in
the reduced SVD basis (DMD/ERA style) then yields both modes' frequencies and
damping without any numerical differentiation — one-sided differences lag the
state by half a sample (which regressions absorb as artificial damping) and
even central differences warp the frequency by ``sin(w dt)/dt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def hankel_matrix(x: np.ndarray, delays: int) -> np.ndarray:
    """Delay-embedding (Hankel) matrix with ``delays`` rows."""

    x = np.asarray(x, dtype=float)
    if delays >= len(x) // 2:
        raise ValueError("delays should be well below half the number of samples.")
    cols = len(x) - delays + 1
    return np.vstack([x[i : i + cols] for i in range(delays)])


@dataclass(frozen=True)
class Mode:
    """One oscillatory mode extracted from the identified propagator."""

    frequency_hz: float  # damped frequency, imag(lambda)/2pi
    natural_frequency_hz: float  # |lambda|/2pi
    zeta: float  # -real(lambda)/|lambda|
    decay_rate: float  # -real(lambda) [1/s]


@dataclass(frozen=True)
class ModalIdentification:
    modes: list[Mode]
    singular_values: np.ndarray
    rank_energy: float  # energy fraction captured by the used rank
    propagator: np.ndarray  # one-step map M: z_{k+1} = M z_k
    z: np.ndarray  # latent trajectories used for the regression
    dt: float


def identify_modal(
    t: np.ndarray,
    x: np.ndarray,
    *,
    delays: int,
    rank: int,
) -> ModalIdentification:
    """Identify oscillatory modes of a single channel from its delay embedding.

    Hankel -> SVD -> keep ``rank`` latent states -> one-step propagator
    ``z_{k+1} = M z_k`` -> continuous poles ``log(eig(M))/dt``.
    """

    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    H = hankel_matrix(x, delays)
    _, S, Vt = np.linalg.svd(H, full_matrices=False)
    rank = min(rank, len(S))
    z = np.diag(S[:rank]) @ Vt[:rank, :]

    M = z[:, 1:] @ np.linalg.pinv(z[:, :-1])
    eigenvalues = np.log(np.linalg.eigvals(M).astype(complex)) / dt

    modes: list[Mode] = []
    for lam in eigenvalues:
        if lam.imag <= 0.0:
            continue  # keep one of each conjugate pair
        magnitude = float(abs(lam))
        modes.append(
            Mode(
                frequency_hz=float(lam.imag / (2.0 * np.pi)),
                natural_frequency_hz=magnitude / (2.0 * np.pi),
                zeta=float(-lam.real / magnitude) if magnitude else float("nan"),
                decay_rate=float(-lam.real),
            )
        )
    modes.sort(key=lambda m: m.frequency_hz)

    energy = float(np.sum(S[:rank] ** 2) / np.sum(S**2))
    return ModalIdentification(modes=modes, singular_values=S, rank_energy=energy, propagator=M, z=z, dt=dt)
