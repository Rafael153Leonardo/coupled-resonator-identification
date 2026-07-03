"""Generate the narrative figures used by the README.

Each figure is one act of the story told by a single sensor watching a
coupled-cantilever pair:

01 - the experiment: strikes and free decays (raw record + envelope)
02 - the clue: the decay envelope *breathes* (beats) and its spectrum
03 - the suspects: one knife-sharp mode, one broad anharmonic mode
04 - the reveal: the unmeasured oscillator, visible in the delay embedding
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
    decimate_to,
    detect_release_events,
    identify_modal,
    load_scope_csv,
    smoothed_envelope,
    spectral_peaks,
)

DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def first_decay(record: int) -> tuple[np.ndarray, np.ndarray]:
    t, x = load_scope_csv(DATA / f"scope_{record}_1.csv")
    x = x - np.mean(x)
    start, stop = detect_release_events(t, x)[0]
    t_ev = t[start:stop] - t[start]
    return decimate_to(t_ev, x[start:stop], 500.0)


def fig_01_raw_record() -> None:
    t, x = load_scope_csv(DATA / "scope_24_1.csv")
    x = x - np.mean(x)
    env = smoothed_envelope(t, x)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, x, color="gray", linewidth=0.3, alpha=0.6)
    ax.plot(t, env, color="crimson", linewidth=1.4, label="amplitude envelope")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("PSD position signal [V]")
    ax.set_title("The experiment: strike, release, listen — five free decays in one record (scope_24)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_raw_record.png", dpi=160)
    plt.close(fig)


def fig_02_breathing_envelope() -> None:
    t, x = first_decay(11)
    env = smoothed_envelope(t, x, window_s=0.05)

    fig, (ax_time, ax_spec) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    mask = t <= 6.0
    ax_time.plot(t[mask], x[mask], color="gray", linewidth=0.4, alpha=0.6)
    ax_time.plot(t[mask], env[mask], color="crimson", linewidth=1.6, label="envelope")
    ax_time.set_xlabel("Time after release [s]")
    ax_time.set_ylabel("Signal [V]")
    ax_time.set_title("The clue: the decay does not just fade — it breathes")
    ax_time.annotate(
        "energy leaves...",
        xy=(0.62, float(env[np.argmin(np.abs(t - 0.62))])),
        xytext=(1.15, 2.3),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=10,
    )
    ax_time.annotate(
        "...and comes back",
        xy=(0.88, float(env[np.argmin(np.abs(t - 0.88))])),
        xytext=(1.6, 1.75),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=10,
    )
    ax_time.legend(loc="upper right")
    ax_time.grid(True, alpha=0.3)

    env_centered = env - np.mean(env)
    window = np.hanning(len(env_centered))
    amp = np.abs(np.fft.rfft(env_centered * window))
    freqs = np.fft.rfftfreq(len(env_centered), d=float(np.median(np.diff(t))))
    band = (freqs > 0.3) & (freqs < 8.0)
    ax_spec.plot(freqs[band], amp[band], color="tab:purple", linewidth=1.2)
    f_peak = float(freqs[band][np.argmax(amp[band])])
    ax_spec.axvline(f_peak, color="gray", linestyle=":", linewidth=1)
    ax_spec.set_xlabel("Envelope frequency [Hz]")
    ax_spec.set_ylabel("Envelope spectrum")
    ax_spec.set_title(f"The breathing has a pulse: {f_peak:.2f} Hz")
    ax_spec.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "02_breathing_envelope.png", dpi=160)
    plt.close(fig)


def fig_03_two_modes() -> None:
    t, x = first_decay(11)
    window = np.hanning(len(x))
    amp = np.abs(np.fft.rfft((x - np.mean(x)) * window))
    freqs = np.fft.rfftfreq(len(x), d=float(np.median(np.diff(t))))
    band = (freqs > 33.0) & (freqs < 44.0)
    peaks = spectral_peaks(t, x, n_peaks=2, fmin=30.0, fmax=45.0, min_separation_hz=1.0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(freqs[band], amp[band], color="black", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude (log)")
    ax.set_title("The suspects: two modes hide in one channel — and they are not alike")
    if len(peaks) == 2:
        ax.annotate(
            f"mode 1: {peaks[0][0]:.2f} Hz\nknife-sharp = linear",
            xy=(peaks[0][0], peaks[0][1]),
            xytext=(33.4, peaks[0][1] * 0.25),
            arrowprops={"arrowstyle": "->", "color": "tab:blue"},
            fontsize=10,
            color="tab:blue",
        )
        ax.annotate(
            "mode 2: broad hump\n= frequency slides with amplitude",
            xy=(40.6, peaks[1][1] * 0.2),
            xytext=(41.3, peaks[1][1] * 0.02),
            arrowprops={"arrowstyle": "->", "color": "tab:orange"},
            fontsize=10,
            color="tab:orange",
        )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_two_modes.png", dpi=160)
    plt.close(fig)


def fig_04_hidden_oscillator() -> None:
    t, x = first_decay(11)
    result = identify_modal(t, x, delays=150, rank=4)
    z = result.z

    fig, (ax_meas, ax_ghost) = plt.subplots(1, 2, figsize=(11, 5))
    n = min(z.shape[1], 4000)  # ~8 s of latent trajectory
    ax_meas.plot(z[0, :n], z[1, :n], color="tab:blue", linewidth=0.4, alpha=0.8)
    ax_meas.set_xlabel("$z_1$")
    ax_meas.set_ylabel("$z_2$")
    ax_meas.set_title("Latent pair ($z_1$, $z_2$):\nthe oscillator the sensor watches")
    ax_meas.grid(True, alpha=0.3)
    ax_meas.set_aspect("equal", adjustable="datalim")

    ax_ghost.plot(z[2, :n], z[3, :n], color="tab:purple", linewidth=0.4, alpha=0.8)
    ax_ghost.set_xlabel("$z_3$")
    ax_ghost.set_ylabel("$z_4$")
    ax_ghost.set_title("Latent pair ($z_3$, $z_4$):\nthe partner no sensor ever touched")
    ax_ghost.grid(True, alpha=0.3)
    ax_ghost.set_aspect("equal", adjustable="datalim")

    fig.suptitle("The reveal: a delay embedding of ONE channel exposes BOTH oscillators (Takens)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_hidden_oscillator.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_01_raw_record()
    fig_02_breathing_envelope()
    fig_03_two_modes()
    fig_04_hidden_oscillator()
    for name in ["01_raw_record", "02_breathing_envelope", "03_two_modes", "04_hidden_oscillator"]:
        print(f"figure: {FIGURES / (name + '.png')}")


if __name__ == "__main__":
    main()
