"""Coupled-pair physics: coupling constant, driven response and bistability."""

from __future__ import annotations

import numpy as np


def coupling_over_mass(f1_hz: float, f2_hz: float) -> float:
    """``k_c / m_eff`` for two identical resonators, from the mode splitting.

    For identical masses and springs coupled by ``k_c``, the symmetric and
    antisymmetric modes satisfy ``w2^2 - w1^2 = 2 k_c / m``. Returns
    ``k_c / m`` in (rad/s)^2; multiply by the measured effective mass to get
    ``k_c`` in N/m. Not valid for dissimilar resonators.
    """

    w1 = 2.0 * np.pi * f1_hz
    w2 = 2.0 * np.pi * f2_hz
    return float((w2**2 - w1**2) / 2.0)


def driven_response_amplitude(
    f_drive: np.ndarray,
    *,
    f0: float,
    gamma: float,
    eta: float = 0.0,
    alpha: float = 0.0,
    force: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state amplitude(s) of the driven nonlinear oscillator.

    Harmonic balance for ``x'' + gamma x' + eta x^2 x' + w0^2 x + alpha x^3
    = F cos(w t)`` gives, with ``s = a^2``,

        ((w0^2 - w^2) + (3/4) alpha s)^2 s + w^2 (gamma + eta s / 4)^2 s = F^2

    which is cubic in ``s``. Returns ``(amplitude, n_branches)`` per drive
    frequency: the largest physical amplitude and the number of positive
    solutions (``n_branches == 3`` marks the bistable region).
    """

    f_drive = np.atleast_1d(np.asarray(f_drive, dtype=float))
    w0_sq = (2.0 * np.pi * f0) ** 2
    amplitude = np.empty(len(f_drive))
    n_branches = np.empty(len(f_drive), dtype=int)
    for i, f in enumerate(f_drive):
        w = 2.0 * np.pi * f
        a_coef = w0_sq - w**2
        b_coef = 0.75 * alpha
        c_coef = gamma
        d_coef = eta / 4.0
        poly = [
            b_coef**2 + w**2 * d_coef**2,
            2.0 * a_coef * b_coef + 2.0 * w**2 * c_coef * d_coef,
            a_coef**2 + w**2 * c_coef**2,
            -(force**2),
        ]
        roots = np.roots(poly)
        physical = [
            float(np.sqrt(r.real)) for r in roots if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)) and r.real > 0.0
        ]
        amplitude[i] = max(physical) if physical else np.nan
        n_branches[i] = len(physical)
    return amplitude, n_branches


def backbone_shift_criterion(delta_f_hz: float, gamma: float) -> float:
    """Ratio between a backbone shift and the bistability threshold.

    A Duffing-type resonator becomes bistable when the backbone shift at the
    operating amplitude exceeds ``sqrt(3)/2`` times the full linewidth
    ``gamma / (2 pi)``. Returns ``delta_f / threshold``: values > 1 mean the
    resonator can be driven into the bistable regime at that amplitude.
    """

    threshold = np.sqrt(3.0) / 2.0 * gamma / (2.0 * np.pi)
    return float(abs(delta_f_hz) / threshold)
