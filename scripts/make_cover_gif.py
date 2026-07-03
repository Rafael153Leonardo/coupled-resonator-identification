"""Generate the README cover GIF: the beating decay drawing itself in time.

An oscilloscope-style sweep over the first seconds of a measured decay
(scope_11): the raw signal traces in gray, the amplitude envelope in red, and
a marker rides the envelope tip so the beat — energy leaving and coming back —
is felt rather than explained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coupled_id import decimate_to, detect_release_events, load_scope_csv, smoothed_envelope

DATA = ROOT / "data"
FIGURES = ROOT / "figures"

SWEEP_SECONDS = 6.0
N_FRAMES = 90
FPS = 18
HOLD_FRAMES = 18  # freeze the completed sweep before the loop restarts


def main() -> None:
    t, x = load_scope_csv(DATA / "scope_11_1.csv")
    x = x - np.mean(x)
    start, stop = detect_release_events(t, x)[0]
    t_ev = t[start:stop] - t[start]
    t_d, x_d = decimate_to(t_ev, x[start:stop], 500.0)
    mask = t_d <= SWEEP_SECONDS
    t_d, x_d = t_d[mask], x_d[mask]
    env = smoothed_envelope(t_d, x_d, window_s=0.05)

    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=100)
    ax.set_xlim(0.0, SWEEP_SECONDS)
    y_max = 1.15 * float(np.max(np.abs(x_d)))
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("Time after release [s]")
    ax.set_ylabel("Signal [V]")
    ax.set_title("One sensor, two oscillators: the decay breathes as energy is exchanged")
    ax.grid(True, alpha=0.3)

    (line_signal,) = ax.plot([], [], color="gray", linewidth=0.5, alpha=0.7)
    (line_env_top,) = ax.plot([], [], color="crimson", linewidth=1.8)
    (line_env_bot,) = ax.plot([], [], color="crimson", linewidth=1.8, alpha=0.55)
    (tip,) = ax.plot([], [], "o", color="crimson", markersize=6)
    fig.tight_layout()

    # Render each frame explicitly and assemble the GIF with PIL: full control
    # over per-frame durations, no animation-writer quirks.
    cut_indices = np.linspace(2, len(t_d) - 1, N_FRAMES).astype(int)
    frames: list[Image.Image] = []
    for cut in cut_indices:
        line_signal.set_data(t_d[:cut], x_d[:cut])
        line_env_top.set_data(t_d[:cut], env[:cut])
        line_env_bot.set_data(t_d[:cut], -env[:cut])
        tip.set_data([t_d[cut]], [env[cut]])
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[..., :3]).convert("P", palette=Image.Palette.ADAPTIVE))
    plt.close(fig)

    frame_ms = int(round(1000.0 / FPS))
    durations = [frame_ms] * (len(frames) - 1) + [frame_ms * HOLD_FRAMES]
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "cover_beat.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"gif: {out} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames @ {FPS} fps + hold)")


if __name__ == "__main__":
    main()
