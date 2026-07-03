"""Modal identification of coupled resonators from a single measured channel."""

from coupled_id.events import decimate_to, detect_release_events, load_scope_csv, smoothed_envelope
from coupled_id.modal import ModalIdentification, Mode, hankel_matrix, identify_modal
from coupled_id.physics import backbone_shift_criterion, coupling_over_mass, driven_response_amplitude
from coupled_id.spectral import bandpass, envelope_exchange_rate, instantaneous_ridge, spectral_peaks

__all__ = [
    "ModalIdentification",
    "Mode",
    "backbone_shift_criterion",
    "bandpass",
    "coupling_over_mass",
    "decimate_to",
    "detect_release_events",
    "driven_response_amplitude",
    "envelope_exchange_rate",
    "hankel_matrix",
    "identify_modal",
    "instantaneous_ridge",
    "load_scope_csv",
    "smoothed_envelope",
    "spectral_peaks",
]
