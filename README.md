# Coupled-Resonator Identification

[![CI](https://github.com/Rafael153Leonardo/coupled-resonator-identification/actions/workflows/ci.yml/badge.svg)](https://github.com/Rafael153Leonardo/coupled-resonator-identification/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**Both modes of a coupled-resonator pair, identified from a single sensor.**
A coupled pair exchanges energy, so one measured channel carries the full
two-mode dynamics — this toolkit extracts both natural frequencies, both
damping ratios, the coupling constant and each mode's *backbone*
(frequency-vs-amplitude curve) from free-decay records alone.

| Identified modes per release event | Backbones during free decay |
| --- | --- |
| ![Modes](figures/coupled_modes_summary.png) | (right panel of the same figure: mode 1 rigidly linear, mode 2 softening by ~2 Hz) |

## Method — the contributions in this repo

1. **Single-channel latent embedding.** A large-window Hankel (delay)
   embedding of the one measured channel makes the *unmeasured* resonator
   observable (Takens): rank-4 SVD latents behave as ``(x1, v1, x2, v2)``, and
   their portraits expose the energy exchange with the "hidden" partner.
2. **Differentiation-free modal identification.** A one-step propagator
   ``z_{k+1} = M z_k`` fitted in the latent basis (DMD/ERA style) yields the
   continuous poles as ``log(eig(M))/dt``. This avoids two classic biases that
   plague derivative-based fits: one-sided differences lag the state by half a
   sample (absorbed as *artificial damping*), and even central differences
   warp the frequency by ``sin(w dt)/dt`` (−3.4% at 13 samples/period).
3. **Hysteresis event segmentation.** The records hold repeated manual
   strike/release events, and a beating envelope re-crosses any single
   threshold — a Schmitt-trigger detector (on/off thresholds) cuts whole
   decays without splitting them at beat bellies.
4. **Backbone tracing.** Band-passing each mode and following its Hilbert
   ridge gives frequency as a function of the mode's own amplitude — the
   honest description when a mode is anharmonic, since any single-pole fit is
   then amplitude-averaged.
5. **Bistability assessment.** A harmonic-balance driven-response solver
   (cubic in amplitude²; root multiplicity marks the bistable region) and the
   backbone-vs-linewidth criterion quantify how far each resonator is from the
   bistable regime.

Everything is validated against synthetic ground truth in
[`tests/test_coupled_id.py`](tests/test_coupled_id.py).

## Results on the measured data

`data/` holds 15 oscilloscope records (`scope_11..25`, 1–5 kHz, 20–100 s) of a
coupled-cantilever experiment, measured with a Hamamatsu 1-D PSD position
sensor (µm-scale resolution). Running
[`scripts/run_coupled_modes.py`](scripts/run_coupled_modes.py):

| Configuration | Records | Result |
| --- | --- | --- |
| A: strong coupling | 11–15, 23, 24 (14 decays) | f₁ = **36.356 ± 0.032 Hz**, f₂ = 39.213 ± 0.074 Hz, splitting **2.857 ± 0.065 Hz**, ζ₁ ≈ 0.0033, ζ₂ ≈ 0.0011, k_c/m = 4262 (rad/s)² |
| B: weak coupling | 17, 18 | close pairs, splitting 0.28 / 0.69 Hz |
| C: single resonator | 20–22 | f = 44.62 ± 0.05 Hz, ζ ≈ 0.0155 |

The lower-mode frequency agrees with an independent analysis of the same rig
(tagged 36.334 / 36.35 / 36.311 Hz) to within the ensemble spread.

**Key finding:** the backbone traces show mode 1 rigidly linear across the
full amplitude range while **mode 2 softens by ~2 Hz** — that shift is ~27×
the bistability threshold (√3/2 × linewidth), so the upper mode is deep in the
bistable-capable regime and a driven up/down sweep must show hysteresis. The
toolkit's `driven_response_amplitude` predicts the multivalued region.

Data notes: `scope_16` is a byte-identical duplicate of `scope_15` (kept for
completeness, excluded from statistics); records 19 and 25 show no stable
modes and are left unclassified. Each `scope_N.txt` is the oscilloscope
settings dump paired with `scope_N_1.csv`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q                      # 8 synthetic validations
python scripts/run_coupled_modes.py   # per-event modal table + figures
```

## Acknowledgments

The measurements were taken on the coupled-cantilever rig of the 2025.2
instrumentation course laboratory (prof. A. A. Batista), whose independent
analysis of the rig provided the reference values above. The identification
techniques, code and analysis in this repository are the author's.

## License

MIT — see [`LICENSE`](LICENSE).
