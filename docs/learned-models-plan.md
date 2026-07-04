# Learned-model roadmap (Brunton & Kutz concepts + autoencoders)

Goal: reproduce the classical identification results with the data-driven
methods of *Data-Driven Science and Engineering* (Brunton & Kutz) and with
autoencoders — using the classical pipeline's numbers as ground truth.

## Ground truth to reproduce

| Quantity | Classical value |
| --- | --- |
| f1 | 36.356 ± 0.032 Hz |
| Mode-2 backbone | 40.12 Hz (a→0) → ~38.9 Hz at 1.2 V (softening) |
| Coupling rate g | 1.430 ± 0.033 Hz |
| ζ1 / ζ2 | 0.0033 / 0.0011 |
| Final exam | drive the learned model → the published bistable windows |

## Experiment ladder

- **E0 — discrete-time SINDy on the Hankel-SVD latents**
  (`scripts/run_sindy_latents.py`). Fit `z_{k+1} = Θ(z_k)ξ` with a cubic
  library and STLSQ on the rank-4 latents. Discrete time avoids derivative
  bias entirely (the same reason the classical pipeline uses a one-step
  propagator). Success: the linear part's eigenvalues give the *small-
  amplitude* frequencies (f2 → ~40.1 Hz, not the amplitude-averaged 39.2),
  and simulating the model from a held-out decay's initial condition
  reproduces the measured mode-2 backbone.
- **E1 — parametrized Koopman autoencoder** (Lusch, Kutz & Brunton 2018)
  (`scripts/run_koopman_ae.py`). Encoder/decoder with near-identity skip
  connections around the latents; dynamics advanced by two rotation-scaling
  pairs whose (decay, frequency) are functions of the pair radius — the
  paper's construction for continuous Koopman spectra, whose flagship example
  (a pendulum with amplitude-dependent frequency) is exactly our mode 2.
  Success: the learned ω2(amplitude) overlays the measured backbone while
  ω1 stays flat at 36.36 Hz, on records never seen in training.
- **E2 — SINDy autoencoder** (Champion et al. 2019): discover coordinates and
  sparse governing equations jointly. Only if E0/E1 pass their exams.

## Data hygiene (lessons already paid for in this project)

1. Normalize amplitudes globally per record, never per window — the backbone
   *lives* in the amplitude.
2. Split by record/session, never by window (11–15 train, 23/24 test; the
   A2/A1 mixing shows they are distinct sessions).
3. Uniform Nyquist-safe grids only (anti-aliased decimation to 500 Hz).
4. The Hilbert ridge below ~0.15 V is noise-dominated; weight or trim.
