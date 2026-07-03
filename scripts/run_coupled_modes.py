"""Coupled-resonator modal identification over the measured scope records.

Each record holds one or several manual strike/release events, each followed
by a free decay. The pipeline works per event:

    detect releases (hysteresis) -> cut each decay -> anti-aliased decimation
    -> latent propagator identification (Hankel/SVD/DMD) -> modal table
    -> splitting and k_c/m -> per-mode backbone (instantaneous frequency vs
    amplitude), which exposes the upper mode's softening.

scope_16 is analyzed but excluded from statistics: it is byte-identical to
scope_15 (duplicated export). Independent reference: an earlier analysis of
the same rig tagged the lower mode at 36.334 / 36.35 / 36.311 Hz.
"""

from __future__ import annotations

import csv
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
    coupling_over_mass,
    decimate_to,
    detect_release_events,
    identify_modal,
    instantaneous_ridge,
    load_scope_csv,
    spectral_peaks,
)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RECORDS = list(range(11, 26))
DUPLICATES = {16: 15}  # byte-identical exports, kept in data/ for completeness

CONFIGURATIONS = {
    "A: strong coupling (~2.9 Hz split)": [11, 12, 13, 14, 15, 23, 24],
    "B: weak coupling": [17, 18],
    "C: single 44.6 Hz resonator": [20, 21, 22],
    "unclassified": [19, 25],
}

TARGET_FS = 500.0
RANK = 4
MODE_BAND = (30.0, 45.0)
BACKBONE_BAND_1 = (35.4, 37.2)
BACKBONE_BAND_2 = (37.6, 42.0)


def modal_row(record: int, event: int, t: np.ndarray, x: np.ndarray) -> dict:
    """Identify the coupled pair on one decay segment."""

    t_d, x_d = decimate_to(t, x, TARGET_FS)
    delays = min(150, len(x_d) // 3)
    result = identify_modal(t_d, x_d, delays=delays, rank=RANK)
    in_band = [m for m in result.modes if MODE_BAND[0] < m.natural_frequency_hz < MODE_BAND[1]]
    if len(in_band) < 2:
        result = identify_modal(t_d, x_d, delays=delays, rank=6)
        in_band = [m for m in result.modes if MODE_BAND[0] < m.natural_frequency_hz < MODE_BAND[1]]

    peaks = spectral_peaks(t_d, x_d, n_peaks=2, fmin=MODE_BAND[0], fmax=MODE_BAND[1], min_separation_hz=0.5)

    row: dict = {
        "record": record,
        "event": event,
        "duration_s": round(float(t_d[-1] - t_d[0]), 1),
        "rank_energy": round(result.rank_energy, 5),
        "is_duplicate": record in DUPLICATES,
    }
    if len(in_band) >= 2:
        m1, m2 = in_band[0], in_band[-1]
        row.update(
            f1_hz=round(m1.natural_frequency_hz, 4),
            f2_hz=round(m2.natural_frequency_hz, 4),
            zeta1=round(m1.zeta, 6),
            zeta2=round(m2.zeta, 6),
            splitting_hz=round(m2.natural_frequency_hz - m1.natural_frequency_hz, 4),
            kc_over_m=round(coupling_over_mass(m1.natural_frequency_hz, m2.natural_frequency_hz), 1),
        )
    elif len(in_band) == 1:
        row.update(f1_hz=round(in_band[0].natural_frequency_hz, 4), zeta1=round(in_band[0].zeta, 6))
    if len(peaks) == 2:
        row.update(fft_f1_hz=round(peaks[0][0], 4), fft_f2_hz=round(peaks[1][0], 4))
    elif len(peaks) == 1:
        row.update(fft_f1_hz=round(peaks[0][0], 4))
    return row


def backbone_points(t: np.ndarray, x: np.ndarray, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """(amplitude, frequency) samples of one mode's decay within ``band``."""

    x_band = bandpass(t, x, band[0], band[1])
    freq, amp = instantaneous_ridge(t, x_band)
    dt = float(np.median(np.diff(t)))
    edge = int(round(0.5 / dt))  # drop filter/Hilbert edge transients
    freq, amp = freq[edge:-edge], amp[edge:-edge]
    keep = amp > 0.08 * float(np.max(amp))
    return amp[keep], freq[keep]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("Coupled-resonator modal identification (per release event)")
    print(f"target_fs={TARGET_FS:.0f} Hz  rank={RANK}  band={MODE_BAND} Hz")
    print()
    print("rec/ev | dur[s] |    f1 [Hz] |    f2 [Hz] | split |   zeta1  |   zeta2  | FFT f1/f2")
    print("-" * 95)

    rows: list[dict] = []
    backbone_1: list[tuple[np.ndarray, np.ndarray]] = []
    backbone_2: list[tuple[np.ndarray, np.ndarray]] = []

    for record in RECORDS:
        path = DATA / f"scope_{record}_1.csv"
        if not path.exists():
            continue
        t, x = load_scope_csv(path)
        x = x - np.mean(x)
        events = detect_release_events(t, x)
        if not events:
            print(f"{record:>6} | no release events found")
            continue
        for n_event, (start, stop) in enumerate(events, start=1):
            t_ev = t[start:stop] - t[start]
            x_ev = x[start:stop]
            row = modal_row(record, n_event, t_ev, x_ev)
            rows.append(row)
            duplicate_note = f"  [duplicate of scope_{DUPLICATES[record]}]" if record in DUPLICATES else ""
            print(
                f"{record:>3}/{n_event:<2} | {row['duration_s']:>6.1f} | "
                f"{row.get('f1_hz', '-'):>10} | {row.get('f2_hz', '-'):>10} | "
                f"{row.get('splitting_hz', '-'):>5} | {row.get('zeta1', '-'):>8} | "
                f"{row.get('zeta2', '-'):>8} | {row.get('fft_f1_hz', '-')} / "
                f"{row.get('fft_f2_hz', '-')}{duplicate_note}"
            )
            if row["duration_s"] >= 8.0 and "f2_hz" in row and record not in DUPLICATES:
                t_d, x_d = decimate_to(t_ev, x_ev, TARGET_FS)
                backbone_1.append(backbone_points(t_d, x_d, BACKBONE_BAND_1))
                backbone_2.append(backbone_points(t_d, x_d, BACKBONE_BAND_2))

    write_csv(RESULTS / "coupled_modes_events.csv", rows)

    good = [r for r in rows if "splitting_hz" in r and not r["is_duplicate"]]
    print()
    for label, records in CONFIGURATIONS.items():
        sel = [r for r in good if r["record"] in records]
        if not sel:
            print(f"{label}: no identified pairs")
            continue
        f1s = np.array([r["f1_hz"] for r in sel])
        f2s = np.array([r["f2_hz"] for r in sel])
        splits = np.array([r["splitting_hz"] for r in sel])
        kcs = np.array([r["kc_over_m"] for r in sel])
        print(
            f"{label}: n={len(sel)}  f1={f1s.mean():.4f}+-{f1s.std():.4f}  "
            f"f2={f2s.mean():.4f}+-{f2s.std():.4f}  split={splits.mean():.4f}+-{splits.std():.4f} Hz  "
            f"kc/m={kcs.mean():.0f} (rad/s)^2"
        )

    # ------------------------------------------------------------------
    # Figures: per-event modes and the mode backbones
    # ------------------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, (ax_modes, ax_bb) = plt.subplots(1, 2, figsize=(13.5, 5))

    labels = [f"{r['record']}.{r['event']}" for r in good]
    xpos = np.arange(len(good))
    f1_all = np.array([r["f1_hz"] for r in good])
    f2_all = np.array([r["f2_hz"] for r in good])
    ax_modes.plot(xpos, f1_all, "o", label="mode 1")
    ax_modes.plot(xpos, f2_all, "s", label="mode 2")
    ax_modes.set_xticks(xpos, labels, rotation=60, fontsize=7)
    ax_modes.set_xlabel("record.event")
    ax_modes.set_ylabel("Natural frequency [Hz]")
    ax_modes.set_title("Identified coupled modes per release event")
    ax_modes.legend()
    ax_modes.grid(True, alpha=0.3)

    for amp, freq in backbone_2:
        ax_bb.plot(amp, freq, ".", markersize=1.5, alpha=0.25, color="tab:orange")
    for amp, freq in backbone_1:
        ax_bb.plot(amp, freq, ".", markersize=1.5, alpha=0.25, color="tab:blue")
    ax_bb.set_xlabel("Instantaneous amplitude [V]")
    ax_bb.set_ylabel("Instantaneous frequency [Hz]")
    ax_bb.set_title("Backbones during free decay\n(blue: mode 1, orange: mode 2)")
    ax_bb.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "coupled_modes_summary.png", dpi=160)
    plt.close(fig)

    print()
    print(f"results: {RESULTS / 'coupled_modes_events.csv'}")
    print(f"figure:  {FIGURES / 'coupled_modes_summary.png'}")


if __name__ == "__main__":
    main()
