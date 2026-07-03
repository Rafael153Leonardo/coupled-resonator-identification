"""Spectral helpers: refined peaks, band-limited ridges and beat rates."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def _parabolic_refine(freqs: np.ndarray, amp: np.ndarray, idx: int) -> float:
    """Sub-bin peak refinement by parabolic interpolation (clamped to one bin)."""

    if idx <= 0 or idx >= len(amp) - 1:
        return float(freqs[idx])
    a, b, c = amp[idx - 1], amp[idx], amp[idx + 1]
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return float(freqs[idx])
    shift = 0.5 * (a - c) / denom
    if abs(shift) > 1.0:  # not a parabolic peak (flat top / edge); keep the bin
        return float(freqs[idx])
    return float(freqs[idx] + shift * (freqs[1] - freqs[0]))


def spectral_peaks(
    t: np.ndarray,
    x: np.ndarray,
    *,
    n_peaks: int = 2,
    fmin: float = 0.5,
    fmax: float = 200.0,
    min_separation_hz: float = 0.5,
) -> list[tuple[float, float]]:
    """Return the ``n_peaks`` strongest spectral peaks as (frequency, amplitude).

    Hann-windowed FFT with sub-bin parabolic refinement, peaks separated by at
    least ``min_separation_hz``.
    """

    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t)))
    window = np.hanning(len(x))
    amp = np.abs(np.fft.rfft((x - np.mean(x)) * window))
    freqs = np.fft.rfftfreq(len(x), d=dt)
    band = (freqs >= fmin) & (freqs <= fmax)

    peaks: list[tuple[float, float]] = []
    order = np.argsort(amp * band)[::-1]
    for idx in order:
        if not band[idx]:
            continue
        f_ref = _parabolic_refine(freqs, amp, int(idx))
        if all(abs(f_ref - f) >= min_separation_hz for f, _ in peaks):
            peaks.append((f_ref, float(amp[idx])))
        if len(peaks) == n_peaks:
            break
    peaks.sort(key=lambda p: p[0])
    return peaks


def bandpass(t: np.ndarray, x: np.ndarray, fmin: float, fmax: float, *, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""

    fs = 1.0 / float(np.median(np.diff(t)))
    sos = butter(order, [fmin, fmax], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def instantaneous_ridge(
    t: np.ndarray,
    x_band: np.ndarray,
    *,
    smooth_s: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous (frequency, amplitude) of a band-limited component.

    Hilbert phase derivative and amplitude, both smoothed over ``smooth_s``.
    Used to trace a mode's backbone: frequency as a function of its own
    envelope amplitude during a free decay.
    """

    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    analytic = hilbert(np.asarray(x_band, dtype=float))
    amplitude = np.abs(analytic)
    frequency = np.gradient(np.unwrap(np.angle(analytic)), dt) / (2.0 * np.pi)
    k = max(1, int(round(smooth_s / dt)))
    kernel = np.ones(k) / k
    return np.convolve(frequency, kernel, mode="same"), np.convolve(amplitude, kernel, mode="same")


def envelope_exchange_rate(
    t: np.ndarray,
    x: np.ndarray,
    *,
    fmin: float = 0.2,
    fmax: float = 8.0,
) -> float:
    """Dominant repetition rate of the amplitude envelope, in Hz.

    For a two-mode free decay this is the energy-exchange (beat) rate
    ``f2 - f1``: the envelope ``|2 cos(pi (f2-f1) t)|`` repeats at ``f2 - f1``.
    """

    envelope = np.abs(hilbert(np.asarray(x, dtype=float)))
    envelope = envelope - np.mean(envelope)
    peaks = spectral_peaks(t, envelope, n_peaks=1, fmin=fmin, fmax=fmax, min_separation_hz=0.1)
    return peaks[0][0] if peaks else float("nan")
