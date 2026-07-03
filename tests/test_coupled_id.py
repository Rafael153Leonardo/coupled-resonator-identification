"""Synthetic validation of the identification toolkit.

Every estimator is exercised against signals with known ground truth; the
real-data pipeline in ``scripts/run_coupled_modes.py`` relies on exactly these
code paths.
"""

import numpy as np
import pytest
from scipy.integrate import odeint

from coupled_id import (
    backbone_shift_criterion,
    coupling_over_mass,
    decimate_to,
    detect_release_events,
    driven_response_amplitude,
    envelope_exchange_rate,
    identify_modal,
    spectral_peaks,
)


def _coupled_pair(f_base=36.0, kc_over_m=None, gamma=0.15, fs=2000.0, duration=20.0):
    """Free decay of two identical coupled oscillators; returns t, x1 and the mode freqs."""

    w1 = 2.0 * np.pi * f_base
    if kc_over_m is None:
        kc_over_m = 0.20 * w1**2 / 2.0  # ~10% splitting
    w2 = np.sqrt(w1**2 + 2.0 * kc_over_m)
    k, kc = w1**2, kc_over_m

    def rhs(state, _t):
        x1, v1, x2, v2 = state
        return [
            v1,
            -k * x1 - kc * (x1 - x2) - gamma * v1,
            v2,
            -k * x2 - kc * (x2 - x1) - gamma * v2,
        ]

    sol = odeint(rhs, [1.0, 0.0, 0.0, 0.0], np.arange(0.0, duration, 1.0 / fs))
    t = np.arange(0.0, duration, 1.0 / fs)
    return t, sol[:, 0], w1 / (2.0 * np.pi), w2 / (2.0 * np.pi)


def test_identify_modal_recovers_both_coupled_modes():
    t, x, f1, f2 = _coupled_pair()
    t, x = decimate_to(t, x, 500.0)

    result = identify_modal(t, x, delays=150, rank=4)
    oscillatory = [m for m in result.modes if m.frequency_hz > 1.0]

    assert len(oscillatory) == 2
    assert oscillatory[0].natural_frequency_hz == pytest.approx(f1, rel=0.01)
    assert oscillatory[1].natural_frequency_hz == pytest.approx(f2, rel=0.01)
    # light damping must come out light: no artificial damping from the method
    assert all(m.zeta < 0.01 for m in oscillatory)
    assert result.rank_energy > 0.99


def test_spectral_peaks_resolve_the_pair():
    t, x, f1, f2 = _coupled_pair()
    peaks = spectral_peaks(t, x, n_peaks=2, fmin=10.0, fmax=100.0, min_separation_hz=1.0)

    assert len(peaks) == 2
    assert peaks[0][0] == pytest.approx(f1, abs=0.1)
    assert peaks[1][0] == pytest.approx(f2, abs=0.1)


def test_envelope_exchange_rate_matches_mode_splitting():
    t, x, f1, f2 = _coupled_pair()
    rate = envelope_exchange_rate(t, x, fmin=0.3, fmax=10.0)

    assert rate == pytest.approx(f2 - f1, rel=0.05)


def test_coupling_over_mass_roundtrip():
    f1 = 36.0
    kc_over_m = 5000.0
    f2 = np.sqrt((2.0 * np.pi * f1) ** 2 + 2.0 * kc_over_m) / (2.0 * np.pi)

    assert coupling_over_mass(f1, f2) == pytest.approx(kc_over_m, rel=1e-9)


def test_detect_release_events_finds_repeated_strikes():
    fs = 1000.0
    t = np.arange(0.0, 30.0, 1.0 / fs)
    x = np.zeros_like(t)
    strikes = [3.0, 12.0, 21.0]
    for t0 in strikes:
        mask = t >= t0
        x[mask] += np.sin(2.0 * np.pi * 20.0 * (t[mask] - t0)) * np.exp(-(t[mask] - t0) / 2.0)

    events = detect_release_events(t, x)

    assert len(events) == 3
    for (start, _stop), t0 in zip(events, strikes, strict=True):
        assert t[start] == pytest.approx(t0 + 0.3, abs=0.3)
    bounds = [b for ev in events for b in ev]
    assert bounds == sorted(bounds)


def test_detect_release_events_does_not_split_beating_decays():
    t, x, f1, f2 = _coupled_pair(gamma=0.3, duration=20.0)

    events = detect_release_events(t, x)

    assert len(events) == 1  # beat bellies must not re-trigger the detector


def test_driven_response_linear_and_bistable_regimes():
    f = np.linspace(35.0, 37.5, 300)

    # linear: single branch everywhere, peak at f0
    amp, branches = driven_response_amplitude(f, f0=36.0, gamma=0.4, force=50.0)
    assert int(branches.max()) == 1
    assert f[int(np.argmax(amp))] == pytest.approx(36.0, abs=0.05)

    # strong softening Duffing: a bistable region must appear below f0
    amp_nl, branches_nl = driven_response_amplitude(f, f0=36.0, gamma=0.4, alpha=-2.0e4, force=50.0)
    assert int(branches_nl.max()) == 3
    assert f[branches_nl == 3].mean() < 36.0


def test_backbone_shift_criterion_thresholds():
    gamma = 0.5
    threshold_hz = np.sqrt(3.0) / 2.0 * gamma / (2.0 * np.pi)

    assert backbone_shift_criterion(threshold_hz, gamma) == pytest.approx(1.0)
    assert backbone_shift_criterion(2.0, gamma) > 25.0
