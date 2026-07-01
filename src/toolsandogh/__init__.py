"""
A collection of tools for working with large-scale microscopy data.

Provided to you by the Sandoghdar Division of the Max Planck Institute for the Physics of Light.
"""

from ._canonicalize_video import canonicalize_video
from ._generate_video import generate_video
from ._link import link
from ._load_video import load_video
from ._locate import locate, locate_in_chunk
from ._rolling import differential_rolling_average, rolling_average, rolling_sum
from ._rvt import radial_variance_transform
from ._simulate_particles import simulate_particles
from ._simulate_psf import (
    Beam,
    Camera,
    IScatResult,
    Objective,
    Particle,
    Sample,
    capture,
    dda_solve,
    dipole_far_field,
    dipole_lattice_ellipsoid,
    dipole_lattice_sphere,
    focused_incident_field,
    image_field_from_pupil,
    iscat_contrast,
    polarizability,
    reference_field,
    scattered_field,
    simulate_iscat,
    simulate_iscat_stack,
)
from ._store_video import store_video

__all__: list[str] = [
    "Beam",
    "Camera",
    "IScatResult",
    "Objective",
    "Particle",
    "Sample",
    "canonicalize_video",
    "capture",
    "dda_solve",
    "dipole_far_field",
    "dipole_lattice_ellipsoid",
    "dipole_lattice_sphere",
    "differential_rolling_average",
    "link",
    "locate",
    "locate_in_chunk",
    "focused_incident_field",
    "generate_video",
    "image_field_from_pupil",
    "iscat_contrast",
    "load_video",
    "polarizability",
    "radial_variance_transform",
    "reference_field",
    "rolling_average",
    "rolling_sum",
    "scattered_field",
    "simulate_particles",
    "simulate_iscat",
    "simulate_iscat_stack",
    "store_video",
]
