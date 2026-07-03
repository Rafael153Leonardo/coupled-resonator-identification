"""Mass-free coupling numbers and a testable hysteresis prediction for mode 2.

Everything here works in sensor units (volts) and needs no mass measurement:

1. Coupling rate ``g = (f2 - f1) / 2`` — the standard coupled-oscillator
   coupling quantifier — plus the dimensionless splitting, per strong-coupling
   decay.
2. Beat cross-check: the envelope energy-exchange rate of each long decay must
   equal its own mode splitting ``f2 - f1``.
3. Mode-2 backbone fit ``f(a) = f0 + c a^2`` on the measured decay ridges
   -> effective Duffing coefficient ``alpha = 32 pi^2 f0 c / 3`` (V^-2 s^-2)
   -> harmonic-balance response curves at increasing drive levels, with the
   bistable region (three amplitude branches) and jump frequencies. This is a
   quantitative, testable prediction: a driven up/down sweep of mode 2 must
   show hysteresis between the marked jump frequencies.

Assumption stated: mode-2 nonlinear damping is not identifiable from these
records, so the prediction uses its linear damping (from the DMD poles) only.
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
    driven_response_amplitude,
    envelope_exchange_rate,
    identify_modal,
    instantaneous_ridge,
    load_scope_csv,
)

DATA = ROOT / "data"
FIGURES = ROOT / "figures"

STRONG_COUPLING_RECORDS = [11, 12, 13, 14, 15, 23, 24]
TARGET_FS = 500.0
MODE_BAND = (30.0, 45.0)
BACKBONE_BAND_2 = (37.6, 42.0)
MIN_BACKBONE_AMPLITUDE = 0.15  # V; below this the Hilbert ridge is noise-dominated


def collect_events() -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    events_out = []
    for record in STRONG_COUPLING_RECORDS:
        t, x = load_scope_csv(DATA / f"scope_{record}_1.csv")
        x = x - np.mean(x)
        for n_event, (start, stop) in enumerate(detect_release_events(t, x), start=1):
            t_ev = t[start:stop] - t[start]
            if float(t_ev[-1]) < 8.0:
                continue
            t_d, x_d = decimate_to(t_ev, x[start:stop], TARGET_FS)
            events_out.append((record, n_event, t_d, x_d))
    return events_out


def main() -> None:
    events = collect_events()
    print(f"strong-coupling decays >= 8 s: {len(events)}")
    print()

    # ------------------------------------------------------------------
    # 1-2. Coupling rate and beat cross-check, per decay
    # ------------------------------------------------------------------
    print("rec/ev |  f1 [Hz]  |  f2 [Hz]  | g=(f2-f1)/2 | beat rate | beat/split")
    print("-" * 76)
    g_values, splits, zeta2s, f2s = [], [], [], []
    for record, n_event, t_d, x_d in events:
        result = identify_modal(t_d, x_d, delays=min(150, len(x_d) // 3), rank=4)
        in_band = [m for m in result.modes if MODE_BAND[0] < m.natural_frequency_hz < MODE_BAND[1]]
        if len(in_band) < 2:
            continue
        m1, m2 = in_band[0], in_band[-1]
        split = m2.natural_frequency_hz - m1.natural_frequency_hz
        beat = envelope_exchange_rate(t_d, x_d, fmin=1.5, fmax=5.0)
        g_values.append(split / 2.0)
        splits.append(split)
        zeta2s.append(m2.zeta)
        f2s.append(m2.natural_frequency_hz)
        print(
            f"{record:>3}/{n_event:<2} | {m1.natural_frequency_hz:9.4f} | {m2.natural_frequency_hz:9.4f} | "
            f"{split / 2.0:11.4f} | {beat:9.4f} | {beat / split:10.3f}"
        )

    g_arr = np.array(g_values)
    print()
    print(f"coupling rate g = {g_arr.mean():.4f} +- {g_arr.std():.4f} Hz (mass-free)")
    print(f"dimensionless splitting (f2-f1)/f_mean = {np.mean(splits) / np.mean(f2s):.4f}")

    # ------------------------------------------------------------------
    # 3. Mode-2 backbone -> effective Duffing alpha -> hysteresis prediction
    # ------------------------------------------------------------------
    amps, freqs = [], []
    for _record, _n_event, t_d, x_d in events:
        x_band = bandpass(t_d, x_d, *BACKBONE_BAND_2)
        f_inst, a_inst = instantaneous_ridge(t_d, x_band)
        dt = float(np.median(np.diff(t_d)))
        edge = int(round(0.5 / dt))
        f_inst, a_inst = f_inst[edge:-edge], a_inst[edge:-edge]
        keep = a_inst > MIN_BACKBONE_AMPLITUDE
        amps.append(a_inst[keep])
        freqs.append(f_inst[keep])
    amp_all = np.concatenate(amps)
    freq_all = np.concatenate(freqs)

    c_fit, f0_fit = np.polyfit(amp_all**2, freq_all, 1)
    alpha_eff = 32.0 * np.pi**2 * f0_fit * c_fit / 3.0
    gamma_2 = float(2.0 * np.mean(zeta2s) * 2.0 * np.pi * np.mean(f2s))
    print()
    print(f"mode-2 backbone fit f(a) = f0 + c a^2 over {len(amp_all)} ridge samples:")
    print(f"  f0 (a->0) = {f0_fit:.3f} Hz | c = {c_fit:.3f} Hz/V^2 | alpha_eff = {alpha_eff:.0f} V^-2 s^-2")
    print(f"  gamma_2 (from DMD poles) = {gamma_2:.3f} 1/s")

    threshold_a = float(np.sqrt(np.sqrt(3.0) / 2.0 * gamma_2 / (2.0 * np.pi) / abs(c_fit)))
    print(f"  bistability onset amplitude: a > {threshold_a:.3f} V (measured decays start at ~1.2 V)")

    # Response curves at drive levels reaching ~0.3 / 0.7 / 1.2 V
    f_grid = np.linspace(f0_fit - 3.5, f0_fit + 1.5, 900)
    w0 = 2.0 * np.pi * f0_fit
    fig, (ax_bb, ax_resp) = plt.subplots(1, 2, figsize=(13, 5))

    ax_bb.plot(amp_all, freq_all, ".", ms=1.5, alpha=0.2, color="tab:orange", label="measured ridge (all decays)")
    a_grid = np.linspace(0.0, 1.3, 200)
    ax_bb.plot(a_grid, f0_fit + c_fit * a_grid**2, "k-", lw=2, label="backbone fit $f_0 + c\\,a^2$")
    ax_bb.axvline(threshold_a, color="gray", ls=":", label="bistability onset")
    ax_bb.set_xlabel("Amplitude [V]")
    ax_bb.set_ylabel("Instantaneous frequency [Hz]")
    ax_bb.set_title("Mode-2 backbone (measured)")
    ax_bb.legend(fontsize=8)
    ax_bb.grid(True, alpha=0.3)

    print()
    print("predicted driven response of mode 2 (up/down sweep must jump):")
    for a_target, color in [(0.3, "tab:green"), (0.7, "tab:blue"), (1.2, "tab:red")]:
        force = gamma_2 * w0 * a_target
        amp_resp, branches = driven_response_amplitude(f_grid, f0=f0_fit, gamma=gamma_2, alpha=alpha_eff, force=force)
        bistable = branches == 3
        label = f"drive -> {a_target:.1f} V"
        ax_resp.plot(f_grid, amp_resp, color=color, lw=1.4, label=label)
        if bistable.any():
            f_lo, f_hi = float(f_grid[bistable].min()), float(f_grid[bistable].max())
            ax_resp.axvspan(f_lo, f_hi, color=color, alpha=0.10)
            print(f"  {label}: bistable between {f_lo:.3f} and {f_hi:.3f} Hz (width {f_hi - f_lo:.3f} Hz)")
        else:
            print(f"  {label}: single-valued (below onset)")

    ax_resp.set_xlabel("Drive frequency [Hz]")
    ax_resp.set_ylabel("Steady-state amplitude [V]")
    ax_resp.set_title("Predicted mode-2 response\n(shaded: bistable region -> hysteresis)")
    ax_resp.legend(fontsize=8)
    ax_resp.grid(True, alpha=0.3)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "mode2_hysteresis_prediction.png", dpi=160)
    plt.close(fig)
    print()
    print(f"figure: {FIGURES / 'mode2_hysteresis_prediction.png'}")


if __name__ == "__main__":
    main()
