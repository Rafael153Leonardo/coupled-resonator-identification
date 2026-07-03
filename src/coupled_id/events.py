"""Loading, conditioning and release-event segmentation of scope records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import decimate as _decimate
from scipy.signal import hilbert


def load_scope_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a bench-scope CSV export (header lines ``x-axis,1`` / ``second,Volt``)."""

    data = np.loadtxt(path, delimiter=",", skiprows=2)
    t = data[:, 0]
    return t - t[0], data[:, 1]


def decimate_to(t: np.ndarray, x: np.ndarray, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Anti-aliased decimation to approximately ``target_fs``.

    Uses staged FIR decimation (scipy) with integer factors; a plain strided
    subsample is never acceptable for identification work (non-uniform grids
    and aliasing bias every downstream estimate).
    """

    fs = 1.0 / float(np.median(np.diff(t)))
    q_total = max(1, int(round(fs / target_fs)))
    x_out = np.asarray(x, dtype=float)
    q_left = q_total
    while q_left > 1:
        q = min(q_left, 10)
        while q_left % q:
            q -= 1
        x_out = _decimate(x_out, q, ftype="fir", zero_phase=True)
        q_left //= q
    t_out = np.arange(len(x_out)) * (q_total / fs)
    return t_out, x_out


def smoothed_envelope(t: np.ndarray, x: np.ndarray, *, window_s: float = 0.2) -> np.ndarray:
    """Hilbert amplitude envelope smoothed by a moving average."""

    envelope = np.abs(hilbert(np.asarray(x, dtype=float) - float(np.mean(x))))
    k = max(1, int(round(window_s / float(np.median(np.diff(t))))))
    return np.convolve(envelope, np.ones(k) / k, mode="same")


def detect_release_events(
    t: np.ndarray,
    x: np.ndarray,
    *,
    on_fraction: float = 0.35,
    off_fraction: float = 0.10,
    margin_s: float = 0.3,
    min_duration_s: float = 1.5,
) -> list[tuple[int, int]]:
    """Detect strike/release events and return (start, stop) decay segments.

    Records may hold several manual excitations, each followed by a free decay
    whose envelope may *beat* (coupled modes). Detection therefore uses
    hysteresis: a new event triggers when the envelope rises above
    ``on_fraction`` of the global maximum, and the trigger only re-arms after
    the envelope falls below ``off_fraction``. Beat bellies never fall that
    low, so a single decay is never split at its envelope minima.

    Each event spans from ``margin_s`` after its envelope peak to the next
    trigger (or the record end).
    """

    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    env = smoothed_envelope(t, x)
    peak_value = float(np.max(env))
    on_threshold = on_fraction * peak_value
    off_threshold = off_fraction * peak_value

    triggers: list[int] = []
    armed = True
    for i, value in enumerate(env):
        if armed and value >= on_threshold:
            triggers.append(i)
            armed = False
        elif not armed and value < off_threshold:
            armed = True

    margin = int(round(margin_s / dt))
    events: list[tuple[int, int]] = []
    for k, trigger in enumerate(triggers):
        span_stop = triggers[k + 1] if k + 1 < len(triggers) else len(env)
        peak_idx = trigger + int(np.argmax(env[trigger:span_stop]))
        start = peak_idx + margin
        if (span_stop - start) * dt >= min_duration_s:
            events.append((start, span_stop))
    return events
