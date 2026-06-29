"""
Vectorial iSCAT point-spread function simulation using JAX and DDA.

The simulation keeps complex vector fields for as long as possible and only
forms intensities / contrast at the very end.  Units are micrometers and
seconds unless otherwise noted.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, Complex, Float

# ---------------------------------------------------------------------------
# Optical component dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Camera:
    """
    Digital camera model.

    Attributes
    ----------
    full_well_capacity_e : float
        Full-well capacity in electrons.
    quantum_efficiency : float
        Quantum efficiency in the range [0, 1].
    pixel_size_um : float
        Physical pixel size in micrometers.
    readout_noise_e : float
        Gaussian readout-noise standard deviation in electrons.
    dark_current_e_per_s : float
        Dark current in electrons per pixel per second.
    exposure_time_s : float
        Exposure time in seconds.
    bit_depth : int
        ADC bit depth.
    offset_adu : float
        Digital offset in analog-to-digital units.
    """

    full_well_capacity_e: float = 20_000.0
    quantum_efficiency: float = 0.9
    pixel_size_um: float = 6.5
    readout_noise_e: float = 1.0
    dark_current_e_per_s: float = 0.1
    exposure_time_s: float = 1.0
    bit_depth: int = 16
    offset_adu: float = 100.0

    def sample_pixel_size_um(self, objective: Objective) -> float:
        """
        Return the sample-space pixel size for the given objective.

        Parameters
        ----------
        objective : Objective
            The imaging objective that sets the magnification.

        Returns
        -------
        float
            Sample-space pixel size in micrometers.
        """
        return self.pixel_size_um / objective.magnification

    def electrons_to_adu(self, electrons: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        Convert electrons to digital counts including offset.

        Parameters
        ----------
        electrons : Float[Array, "..."]
            Electron count per pixel before ADC conversion.

        Returns
        -------
        Float[Array, "..."]
            Quantized pixel value in ADU, clipped to ``[0, 2**bit_depth - 1]``.
        """
        max_adu = 2.0**self.bit_depth - 1.0
        gain = max_adu / self.full_well_capacity_e
        return jnp.clip(electrons * gain + self.offset_adu, 0.0, max_adu)

    def capture(
        self,
        key: jax.Array,
        photons: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """
        Return a noisy digital image from an ideal photon-count image.

        The model includes Poisson shot noise on the incoming photons, dark
        current, read noise, full-well clipping, and ADC quantization.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        photons : Float[Array, "..."]
            Ideal photon count per pixel.

        Returns
        -------
        Float[Array, "..."]
            Noisy image in ADU.
        """
        electrons_mean = photons * self.quantum_efficiency
        electrons_mean += self.dark_current_e_per_s * self.exposure_time_s

        key, k1, k2 = jax.random.split(key, 3)
        shot = jax.random.poisson(k1, electrons_mean)
        read = self.readout_noise_e * jax.random.normal(k2, electrons_mean.shape)
        electrons = jnp.clip(shot + read, 0.0, self.full_well_capacity_e)
        return self.electrons_to_adu(electrons)


@dataclasses.dataclass(frozen=True)
class Objective:
    """
    Vectorial microscope objective.

    Attributes
    ----------
    na : float
        Numerical aperture.
    magnification : float
        Magnification of the objective-tube-lens combination.
    immersion_index : float
        Refractive index of the immersion medium.
    working_distance_um : float | None
        Working distance in micrometers (optional, for future use).
    defocus_um : float
        Axial defocus of the image plane relative to nominal focus.
    """

    na: float
    magnification: float
    immersion_index: float = 1.0
    working_distance_um: float | None = None
    defocus_um: float = 0.0


@dataclasses.dataclass(frozen=True)
class Beam:
    """
    Illumination beam and cover-slip reflection reference.

    Attributes
    ----------
    wavelength_um : float
        Vacuum wavelength in micrometers.
    power_w : float
        Total beam power in Watts.
    n_medium : float
        Refractive index of the medium in which the beam is focused.
    polarization : tuple[float, float, float]
        Incident polarization direction before the objective.  Normalized
        internally.
    beam_type : {"plane", "gaussian"}
        ``plane`` fills the objective pupil uniformly; ``gaussian`` uses a
        Gaussian angular spectrum.
    waist_um : float | None
        1/e^2 intensity radius of a Gaussian beam at focus.  If ``None``,
        a default based on the wavelength is used.
    z_focus_um : float
        Axial position of the beam focus relative to the nominal sample plane.
    reflection_coefficient : complex
        Complex Fresnel reflection coefficient of the cover-slip--medium
        interface that produces the reference field.  For a plane wave at
        normal incidence the convention is
        ``(n_medium - n_glass) / (n_medium + n_glass)``, which is negative
        for typical dielectric cover slips (a lower-index medium on top of
        a higher-index glass).  The default value
        ``(1.33 - 1.52) / (1.33 + 1.52)`` corresponds to a water/glass
        interface; override it whenever ``n_medium`` is not water.
    """

    wavelength_um: float
    power_w: float = 1.0
    n_medium: float = 1.33
    polarization: tuple[float, float, float] = (1.0, 0.0, 0.0)
    beam_type: Literal["plane", "gaussian"] = "gaussian"
    waist_um: float | None = None
    z_focus_um: float = 0.0
    reflection_coefficient: complex | Array = complex((1.33 - 1.52) / (1.33 + 1.52))

    def wavenumber(self, n_medium: float) -> float:
        """
        Return the wavenumber ``k = 2π n / λ`` in µm^-1.

        Parameters
        ----------
        n_medium : float
            Refractive index of the medium in which the wave propagates.

        Returns
        -------
        float
            Wavenumber in reciprocal micrometers.
        """
        return 2.0 * jnp.pi * n_medium / self.wavelength_um

    def photon_energy_j(self) -> float:
        """
        Return the photon energy ``h c / λ`` in Joules.

        Returns
        -------
        float
            Photon energy in Joules.
        """
        return 1.98644586e-19 / self.wavelength_um

    def focal_waist_um(self) -> float:
        """
        Return the Gaussian focal waist in micrometers.

        Returns
        -------
        float
            1/e^2 intensity radius of the focused Gaussian beam in micrometers,
            or ``+inf`` for a plane-wave beam.
        """
        if self.beam_type == "plane":
            return float("inf")
        if self.waist_um is not None:
            return float(self.waist_um)
        return float(self.wavelength_um / jnp.pi)

    def reflected_power_w(self) -> float:
        """
        Return the reflected reference power in Watts.

        Returns
        -------
        float
            Power of the reference field in Watts, equal to
            ``power_w * |reflection_coefficient|^2``.
        """
        return float(self.power_w * jnp.abs(self.reflection_coefficient) ** 2)


@dataclasses.dataclass(frozen=True)
class Particle:
    """
    A particle represented as a set of point dipoles.

    Attributes
    ----------
    positions_um : Float[Array, "N 3"]
        Dipole positions in micrometers.
    polarizabilities_um3 : Complex[Array, "N 3"]
        Diagonal components of the dipole polarizability tensor in µm^3.
    """

    positions_um: Float[Array, "N 3"]
    polarizabilities_um3: Complex[Array, "N 3"]

    @staticmethod
    def sphere(
        radius_um: float,
        n_particle: complex | float,
        n_medium: float,
        spacing_um: float | None = None,
        wavelength_um: float | None = None,
        center_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Particle:
        """
        Build a spherical :class:`Particle` discretized with DDA.

        Parameters
        ----------
        radius_um : float
            Sphere radius in micrometers.
        n_particle : complex | float
            Particle refractive index.
        n_medium : float
            Medium refractive index.
        spacing_um : float | None
            Cubic lattice spacing.  Defaults to a sensible fraction of the
            wavelength in the medium if ``wavelength_um`` is provided.
        wavelength_um : float | None
            Wavelength used to pick a default spacing.
        center_um : tuple[float, float, float]
            Sphere center.

        Returns
        -------
        Particle
            A :class:`Particle` instance populated with the lattice positions
            and the (isotropic) LDR polarizabilities.
        """
        if spacing_um is None:
            if wavelength_um is None:
                raise ValueError("Provide spacing_um or wavelength_um.")
            spacing_um = float(wavelength_um / (10.0 * n_medium))
        positions = dipole_lattice_sphere(
            radius_um=radius_um,
            spacing_um=spacing_um,
            center_um=center_um,
        )
        alpha = _lattice_alpha(n_particle, n_medium, spacing_um, wavelength_um)
        alphas = jnp.full((positions.shape[0], 3), alpha)
        return Particle(positions_um=positions, polarizabilities_um3=alphas)

    @staticmethod
    def ellipsoid(
        semi_axes_um: tuple[float, float, float],
        n_particle: complex | float,
        n_medium: float,
        spacing_um: float | None = None,
        wavelength_um: float | None = None,
        center_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Particle:
        """
        Build an ellipsoidal :class:`Particle` discretized with DDA.

        Parameters
        ----------
        semi_axes_um : tuple[float, float, float]
            Semi-axes along x, y, z in micrometers.
        n_particle : complex | float
            Particle refractive index.
        n_medium : float
            Medium refractive index.
        spacing_um : float | None
            Cubic lattice spacing.  Defaults to a sensible fraction of the
            wavelength in the medium if ``wavelength_um`` is provided.
        wavelength_um : float | None
            Wavelength used to pick a default spacing.
        center_um : tuple[float, float, float]
            Ellipsoid center.

        Returns
        -------
        Particle
            A :class:`Particle` instance populated with the lattice positions
            and the (isotropic) LDR polarizabilities.
        """
        if spacing_um is None:
            if wavelength_um is None:
                raise ValueError("Provide spacing_um or wavelength_um.")
            spacing_um = float(wavelength_um / (10.0 * n_medium))
        positions = dipole_lattice_ellipsoid(
            semi_axes_um=semi_axes_um,
            spacing_um=spacing_um,
            center_um=center_um,
        )
        alpha = _lattice_alpha(n_particle, n_medium, spacing_um, wavelength_um)
        alphas = jnp.full((positions.shape[0], 3), alpha)
        return Particle(positions_um=positions, polarizabilities_um3=alphas)


@dataclasses.dataclass(frozen=True)
class Sample:
    """
    A collection of particles embedded in a homogeneous medium on a cover slip.

    Attributes
    ----------
    medium_index : complex | float
        Refractive index of the surrounding medium.
    particles : Sequence[Particle]
        Particles to include in the simulation.
    """

    medium_index: complex | float
    particles: Sequence[Particle] = dataclasses.field(default_factory=tuple)

    def dipoles(self) -> tuple[Float[Array, "N 3"], Complex[Array, "N 3"]]:
        """
        Return concatenated dipole positions and polarizabilities.

        Returns
        -------
        Float[Array, "N 3"]
            Concatenated dipole positions in micrometers.
        Complex[Array, "N 3"]
            Concatenated diagonal polarizability tensor entries in µm^3.

        Raises
        ------
        ValueError
            If the sample contains no particles.
        """
        if not self.particles:
            raise ValueError("Sample contains no particles.")
        positions = jnp.concatenate([p.positions_um for p in self.particles], axis=0)
        alphas = jnp.concatenate([p.polarizabilities_um3 for p in self.particles], axis=0)
        return positions, alphas


# ---------------------------------------------------------------------------
# Public free functions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IScatResult:
    """
    Container for the output of :func:`simulate_iscat`.

    All arrays share the image-plane grid ``(H, W)``.  The scattered field
    carries the full vector (``E_x, E_y, E_z``); the other entries are scalar
    images.  ``camera_image`` is ``None`` unless camera noise was requested.
    """

    scattered_field: Complex[Array, "H W 3"]
    reference_field: Complex[Array, "H W"]
    intensity: Float[Array, "H W"]
    contrast: Float[Array, "H W"]
    contrast_linear: Float[Array, "H W"]
    photons: Float[Array, "H W"]
    camera_image: Float[Array, "H W"] | None = None


def simulate_iscat(
    sample: Sample,
    beam: Beam,
    objective: Objective,
    camera: Camera,
    grid_shape: tuple[int, int],
    z_plane_um: float = 0.0,
    key: jax.Array | None = None,
    return_camera_image: bool = False,
) -> IScatResult:
    """
    Simulate a 2D widefield iSCAT image.

    Parameters
    ----------
    sample : Sample
        The sample (medium + particles).
    beam : Beam
        The illumination beam and reference source.
    objective : Objective
        The imaging objective.
    camera : Camera
        The camera model.
    grid_shape : tuple[int, int]
        Image grid shape ``(H, W)``.
    z_plane_um : float
        Axial image-plane position relative to nominal focus.
    key : jax.Array | None
        PRNG key for camera noise.  Required if ``return_camera_image`` is True.
    return_camera_image : bool
        If True, also simulate a noisy camera image and attach it to the
        returned :class:`IScatResult`.

    Returns
    -------
    IScatResult
        Container holding the scattered and reference fields, intensity,
        contrast, linear contrast, photon counts, and (optionally) a noisy
        camera image.
    """
    pixel_size_sample_um = camera.sample_pixel_size_um(objective)

    E_s = scattered_field(sample, beam, objective, grid_shape, z_plane_um, pixel_size_sample_um)
    E_r = reference_field(beam, objective, grid_shape, pixel_size_sample_um, z_plane_um)

    intensity, contrast, contrast_linear, photons = _iscat_fields(E_s, E_r, beam, camera)

    camera_image = None
    if return_camera_image:
        if key is None:
            raise ValueError("A PRNG key is required for camera noise.")
        camera_image = camera.capture(key, photons)

    return IScatResult(
        scattered_field=E_s,
        reference_field=E_r,
        intensity=intensity,
        contrast=contrast,
        contrast_linear=contrast_linear,
        photons=photons,
        camera_image=camera_image,
    )


def simulate_iscat_stack(
    sample: Sample,
    beam: Beam,
    objective: Objective,
    camera: Camera,
    grid_shape: tuple[int, int],
    z_planes_um: Sequence[float] | Float[Array, " N"],
    *,
    field: str = "contrast",
) -> xr.DataArray:
    """
    Simulate an iSCAT defocus stack and return it as a canonical video.

    The DDA self-consistency equation is solved once; only the z-dependent
    field propagation (reference and scattered paths) is mapped over the
    supplied axial planes.  The requested scalar field is packed into a
    canonical ``TCZYX`` :class:`xarray.DataArray` backed by Dask, so the result
    composes directly with :func:`~toolsandogh.store_video`,
    :func:`~toolsandogh.rolling_average`,
    :func:`~toolsandogh.radial_variance_transform`, and friends.

    Parameters
    ----------
    sample : Sample
        The sample (medium + particles).
    beam : Beam
        The illumination beam and reference source.
    objective : Objective
        The imaging objective.
    camera : Camera
        The camera model (its pixel size sets the sample-space grid).
    grid_shape : tuple[int, int]
        Image grid shape ``(H, W)``.
    z_planes_um : Sequence[float] or Float[Array, "Z"]
        Axial image-plane positions relative to nominal focus, one per slice.
    field : {"contrast", "contrast_linear", "intensity", "photons"}
        Which scalar output of :class:`IScatResult` to place into the video.

    Returns
    -------
    xarray.DataArray
        A canonical ``TCZYX`` video with ``Z = len(z_planes_um)``.
    """
    from ._canonicalize_video import canonicalize_video

    valid_fields = ("contrast", "contrast_linear", "intensity", "photons")
    if field not in valid_fields:
        raise ValueError(f"field must be one of {valid_fields}, got {field!r}.")

    pixel_size_sample_um = camera.sample_pixel_size_um(objective)
    z_planes_um = jnp.array(z_planes_um)
    z_planes = jnp.asarray(jnp.atleast_1d(z_planes_um), dtype=jnp.float32)

    # Solve the DDA once: the dipole moments are independent of the image plane.
    positions, alphas = sample.dipoles()
    n_medium = float(abs(sample.medium_index))
    k = beam.wavenumber(n_medium)
    E_inc = focused_incident_field(
        positions=positions,
        beam=beam,
        objective=objective,
        n_medium=n_medium,
        wavelength_um=beam.wavelength_um,
    )
    dipole_moments = dda_solve(
        incident_field=E_inc,
        positions=positions,
        alphas=alphas,
        k=k,
    )

    # Map only the z-dependent propagation over the stack.
    def scattered_at(zp):
        return _dipole_field_on_grid(
            positions=positions,
            dipole_moments=dipole_moments,
            n_medium=n_medium,
            k=k,
            na=objective.na,
            grid_shape=grid_shape,
            pixel_size_sample_um=pixel_size_sample_um,
            z_plane_um=zp + objective.defocus_um,
        )

    def reference_at(zp):
        return reference_field(beam, objective, grid_shape, pixel_size_sample_um, zp)

    E_s = jax.vmap(scattered_at)(z_planes)  # (Z, H, W, 3)
    E_r = jax.vmap(reference_at)(z_planes)  # (Z, H, W)

    pol = jnp.array(beam.polarization, dtype=jnp.float32)
    pol = pol / jnp.linalg.norm(pol)
    E_s_proj = jnp.einsum("zhwc,c->zhw", E_s, pol)

    intensity = jnp.abs(E_r + E_s_proj) ** 2
    intensity_ref = jnp.abs(E_r) ** 2
    contrast = (intensity - intensity_ref) / jnp.maximum(intensity_ref, 1e-30)
    contrast_linear = 2.0 * jnp.real(E_s_proj / jnp.where(jnp.abs(E_r) > 1e-30, E_r, 1e-30))
    reflected_photons = beam.reflected_power_w() / beam.photon_energy_j() * camera.exposure_time_s
    scale = jnp.sqrt(reflected_photons / jnp.sum(intensity_ref))
    photons = (intensity * scale**2).real

    stack = {
        "contrast": contrast,
        "contrast_linear": contrast_linear,
        "intensity": intensity,
        "photons": photons,
    }[field]
    stack = np.asarray(stack, dtype=np.float32)

    z_values = np.asarray(z_planes, dtype=np.float64)
    dz = float(z_values[1] - z_values[0]) if len(z_values) > 1 else 1.0
    # Label the array with explicit (Z, Y, X) dims so canonicalize_video adds
    # the singleton T and C axes rather than reading the leading axis as T.
    stack_da = xr.DataArray(stack, dims=("Z", "Y", "X"), coords={"Z": z_values})
    return canonicalize_video(
        stack_da,
        Z=len(z_values),
        Y=grid_shape[0],
        X=grid_shape[1],
        dz=dz,
        dy=pixel_size_sample_um,
        dx=pixel_size_sample_um,
        dtype=np.float32,
    )


def _iscat_fields(
    E_s: Complex[Array, "H W 3"],
    E_r: Complex[Array, "H W"],
    beam: Beam,
    camera: Camera,
) -> tuple[Float[Array, "H W"], Float[Array, "H W"], Float[Array, "H W"], Float[Array, "H W"]]:
    """
    Build intensity, contrast, linear contrast, and photon counts from fields.

    This factors the post-propagation arithmetic out of :func:`simulate_iscat`
    so that :func:`simulate_iscat_stack` can reuse it without re-solving the DDA.

    Parameters
    ----------
    E_s : Complex[Array, "H W 3"]
        Vector scattered field at the image plane.
    E_r : Complex[Array, "H W"]
        Scalar reference field at the image plane.
    beam : Beam
        The illumination beam; sets the detection polarization and the
        reference power used for the photon-count calibration.
    camera : Camera
        The camera model; provides the exposure time and the photon-to-ADU
        scaling used to convert the calibrated intensity to photon counts.

    Returns
    -------
    tuple
        A 4-tuple of ``Float[Array, "H W"]`` arrays holding, in order:
        total interferometric intensity ``|E_r + E_s_proj|^2``, iSCAT
        contrast ``(I - I_ref) / I_ref``, linearized contrast
        ``2 Re(E_s / E_r)`` (accurate when ``|E_s| << |E_r|``), and photon
        counts per pixel calibrated against the reflected reference power
        and ready to be fed into :meth:`Camera.capture`.
    """
    pol = jnp.array(beam.polarization, dtype=jnp.float32)
    pol = pol / jnp.linalg.norm(pol)
    E_s_proj = jnp.einsum("hwc,c->hw", E_s, pol)

    intensity = jnp.abs(E_r + E_s_proj) ** 2
    intensity_ref = jnp.abs(E_r) ** 2
    contrast = (intensity - intensity_ref) / jnp.maximum(intensity_ref, 1e-30)
    contrast_linear = 2.0 * jnp.real(E_s_proj / jnp.where(jnp.abs(E_r) > 1e-30, E_r, 1e-30))

    # Scale normalized fields to physical photon counts using the reflected power.
    reflected_photons = beam.reflected_power_w() / beam.photon_energy_j() * camera.exposure_time_s
    scale = jnp.sqrt(reflected_photons / jnp.sum(intensity_ref))
    photons = (intensity * scale**2).real
    return intensity, contrast, contrast_linear, photons


def scattered_field(
    sample: Sample,
    beam: Beam,
    objective: Objective,
    grid_shape: tuple[int, int],
    z_plane_um: float = 0.0,
    pixel_size_sample_um: float | None = None,
) -> Complex[Array, "H W 3"]:
    """
    Compute the complex scattered vector field at the image plane.

    Parameters
    ----------
    sample : Sample
        The sample (medium + particles) under study.
    beam : Beam
        The illumination beam; sets the wavelength, polarization, and the
        focused incident field that drives the DDA.
    objective : Objective
        The imaging objective; provides the collection NA, magnification,
        and any axial defocus.
    grid_shape : tuple[int, int]
        Image grid shape ``(H, W)`` in pixels.
    z_plane_um : float
        Axial image-plane position relative to nominal focus, in micrometers.
    pixel_size_sample_um : float | None
        Sample-space pixel size in micrometers.  Inferred from the beam and
        objective when not provided.

    Returns
    -------
    Complex[Array, "H W 3"]
        Image-plane electric field components (E_x, E_y, E_z).
    """
    if pixel_size_sample_um is None:
        # Use a default pixel size based on the diffraction limit.
        pixel_size_sample_um = float(beam.wavelength_um / (8.0 * objective.na))

    positions, alphas = sample.dipoles()
    n_medium = float(abs(sample.medium_index))
    k = beam.wavenumber(n_medium)

    E_inc = focused_incident_field(
        positions=positions,
        beam=beam,
        objective=objective,
        n_medium=n_medium,
        wavelength_um=beam.wavelength_um,
    )
    dipole_moments = dda_solve(
        incident_field=E_inc,
        positions=positions,
        alphas=alphas,
        k=k,
    )

    return _dipole_field_on_grid(
        positions=positions,
        dipole_moments=dipole_moments,
        n_medium=n_medium,
        k=k,
        na=objective.na,
        grid_shape=grid_shape,
        pixel_size_sample_um=pixel_size_sample_um,
        z_plane_um=z_plane_um + objective.defocus_um,
    )


def reference_field(
    beam: Beam,
    objective: Objective,
    grid_shape: tuple[int, int],
    pixel_size_sample_um: float,
    z_plane_um: float = 0.0,
) -> Complex[Array, "H W"]:
    """
    Compute the complex reference field at the image plane.

    The reference field is the reflection of the incident beam at the cover
    slip, propagated through the same objective as the scattered light.

    Parameters
    ----------
    beam : Beam
        The illumination beam; sets the wavelength, polarization, pupil
        amplitude, and Fresnel reflection coefficient.
    objective : Objective
        The imaging objective; provides the collection NA and any defocus.
    grid_shape : tuple[int, int]
        Image grid shape ``(H, W)`` in pixels.
    pixel_size_sample_um : float
        Sample-space pixel size in micrometers.
    z_plane_um : float
        Axial image-plane position relative to nominal focus, in micrometers.

    Returns
    -------
    Complex[Array, "H W"]
        Reference field projected onto the incident polarization direction.
    """
    n_medium = float(beam.n_medium)
    k = beam.wavenumber(n_medium)
    KX, KY, KZ, mask = _fft_pupil_grid(grid_shape, pixel_size_sample_um, k, objective.na, n_medium)

    E_pupil = _incident_pupil_field(KX, KY, KZ, mask, beam, objective, k)
    # Apply defocus to the incident pupil before propagation.
    defocus = jnp.exp(1j * KZ * (z_plane_um + objective.defocus_um))
    E_pupil_defocused = E_pupil * defocus[..., None]
    E_img = image_field_from_pupil(E_pupil_defocused, grid_shape, pixel_size_sample_um)

    pol = jnp.array(beam.polarization, dtype=jnp.float32)
    pol = pol / jnp.linalg.norm(pol)
    return jnp.einsum("hwc,c->hw", E_img, pol) * beam.reflection_coefficient


def dda_solve(
    incident_field: Complex[Array, "N 3"],
    positions: Float[Array, "N 3"],
    alphas: Complex[Array, "N 3"],
    k: float,
) -> Complex[Array, "N 3"]:
    """
    Solve the DDA self-consistency equation for dipole moments.

    The equation solved is ``E_inc = (1/α) P - A P``.

    Parameters
    ----------
    incident_field : Complex[Array, "N 3"]
        Incident field at each dipole.
    positions : Float[Array, "N 3"]
        Dipole positions in micrometers.
    alphas : Complex[Array, "N 3"]
        Dipole polarizabilities in micrometers cubed.
    k : float
        Wavenumber in micrometers^-1.

    Returns
    -------
    Complex[Array, "N 3"]
        Self-consistent dipole moments.
    """
    matrix = _dda_matrix(positions, alphas, k)
    moments = jax.scipy.linalg.solve(matrix, incident_field.reshape(-1))
    return moments.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Polarizability and lattice generators
# ---------------------------------------------------------------------------


def polarizability(
    n_particle: complex | float,
    n_medium: float,
    spacing_um: float,
    wavelength_um: float | None = None,
) -> complex:
    """
    Return the lattice-dispersion-relation polarizability in µm^3.

    Parameters
    ----------
    n_particle : complex | float
        Particle refractive index.
    n_medium : float
        Medium refractive index.
    spacing_um : float
        Cubic lattice spacing in micrometers.
    wavelength_um : float | None
        Wavelength used for the lattice-dispersion correction.

    Returns
    -------
    complex
        Dipole polarizability in micrometers cubed.
    """
    m = n_particle / n_medium
    alpha_cm = spacing_um**3 * (3.0 / (4.0 * jnp.pi)) * (m**2 - 1.0) / (m**2 + 2.0)
    if wavelength_um is not None:
        k = 2.0 * jnp.pi * n_medium / wavelength_um
        correction = 1.0 + alpha_cm * (k**2 / spacing_um + (2.0 / 3.0) * 1j * k**3)
        alpha = alpha_cm / correction
    else:
        alpha = alpha_cm
    return complex(alpha)


_lattice_alpha = polarizability  # internal alias


def dipole_lattice_sphere(
    radius_um: float,
    spacing_um: float,
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Float[Array, "N 3"]:
    """
    Generate a cubic lattice of points inside a sphere.

    Parameters
    ----------
    radius_um : float
        Sphere radius in micrometers.
    spacing_um : float
        Cubic lattice spacing in micrometers.
    center_um : tuple[float, float, float]
        Sphere center in micrometers.

    Returns
    -------
    Float[Array, "N 3"]
        Lattice points inside the sphere, shape ``(N, 3)``.
    """
    n = int(jnp.ceil(radius_um / spacing_um))
    coords1d = jnp.arange(-n, n + 1) * spacing_um
    cx, cy, cz = center_um
    x = coords1d + cx
    y = coords1d + cy
    z = coords1d + cz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    positions = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    mask = jnp.sum((positions - jnp.array(center_um)) ** 2, axis=-1) <= radius_um**2
    return positions[mask]


def dipole_lattice_ellipsoid(
    semi_axes_um: tuple[float, float, float],
    spacing_um: float,
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Float[Array, "N 3"]:
    """
    Generate a cubic lattice of points inside an ellipsoid.

    Parameters
    ----------
    semi_axes_um : tuple[float, float, float]
        Semi-axes along x, y, z in micrometers.
    spacing_um : float
        Cubic lattice spacing in micrometers.
    center_um : tuple[float, float, float]
        Ellipsoid center in micrometers.

    Returns
    -------
    Float[Array, "N 3"]
        Lattice points inside the ellipsoid, shape ``(N, 3)``.
    """
    ax, ay, az = semi_axes_um
    n = max(
        int(jnp.ceil(ax / spacing_um)),
        int(jnp.ceil(ay / spacing_um)),
        int(jnp.ceil(az / spacing_um)),
    )
    coords1d = jnp.arange(-n, n + 1) * spacing_um
    cx, cy, cz = center_um
    x = coords1d + cx
    y = coords1d + cy
    z = coords1d + cz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    positions = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    rel = positions - jnp.array(center_um)
    ax2, ay2, az2 = ax**2, ay**2, az**2
    mask = (rel[:, 0] ** 2 / ax2 + rel[:, 1] ** 2 / ay2 + rel[:, 2] ** 2 / az2) <= 1.0
    return positions[mask]


# ---------------------------------------------------------------------------
# DDA matrix
# ---------------------------------------------------------------------------


def _dda_matrix(
    positions: Float[Array, "N 3"],
    alphas: Complex[Array, "N 3"],
    k: float,
) -> Complex[Array, "3N 3N"]:
    """
    Build the DDA interaction matrix ``(1/α - A)``.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Dipole positions in micrometers.
    alphas : Complex[Array, "N 3"]
        Diagonal polarizability entries in µm^3.
    k : float
        Wavenumber in the medium in reciprocal micrometers.

    Returns
    -------
    Complex[Array, "3N 3N"]
        The dense ``(3N, 3N)`` interaction matrix, with diagonal blocks
        ``1/α * I`` and off-diagonal blocks ``-A_{ij}``.
    """
    N = positions.shape[0]
    eye3 = jnp.eye(3, dtype=alphas.dtype)

    r_vec = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    dist = jnp.linalg.norm(r_vec, axis=-1, keepdims=True)  # (N, N, 1)
    mask = jnp.eye(N)[:, :, None]  # diagonal selector

    # Avoid division by zero on the diagonal; those blocks are replaced later.
    safe_dist = jnp.where(mask, 1.0, dist)
    rhat = r_vec / safe_dist  # (N, N, 3)
    rhat_outer = rhat[..., :, None] * rhat[..., None, :]  # (N, N, 3, 3)

    kr = k * safe_dist[..., 0]  # (N, N)
    expikr = jnp.exp(1j * kr)

    term1 = (kr[..., None, None] ** 2) * (eye3 - rhat_outer)
    term2 = (1.0 - 1j * kr[..., None, None]) * (3.0 * rhat_outer - eye3)
    offdiag = expikr[..., None, None] / (safe_dist[..., None] ** 3) * (term1 + term2)

    # Diagonal blocks are 1/α * I.
    inv_alpha = 1.0 / alphas  # (N, 3)
    diag = jnp.einsum("ia,ab->iab", inv_alpha, eye3)  # (N, 3, 3)

    blocks = jnp.where(mask[..., None], diag[:, None, :, :], -offdiag)
    return blocks.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


# ---------------------------------------------------------------------------
# Vectorial field propagation
# ---------------------------------------------------------------------------


def _fft_pupil_grid(
    grid_shape: tuple[int, int],
    pixel_size_sample_um: float,
    k: float,
    na: float | None = None,
    n_medium: float = 1.0,
) -> tuple[Float[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"], Float[Array, "H W"]]:
    """
    Return the k-space grid matching an image-plane FFT grid.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Image grid shape ``(H, W)``.
    pixel_size_sample_um : float
        Sample-space pixel size in micrometers.
    k : float
        Wavenumber in the medium in reciprocal micrometers.
    na : float | None
        Numerical aperture of the collecting objective.  If provided, the
        returned pupil mask selects the objective's collection disc
        ``kx^2 + ky^2 <= (NA / n_medium * k)^2``; otherwise it selects all
        propagating waves ``kx^2 + ky^2 <= k^2``.
    n_medium : float
        Refractive index of the medium.  Only used together with ``na`` to
        convert the objective NA into the medium half-angle
        ``sin(theta) = NA / n_medium``; the transverse cutoff is then
        ``k * sin(theta) = (NA / n_medium) * k``.

    Returns
    -------
    Float[Array, "H W"]
        Transverse x-component of the wavevector grid in reciprocal micrometers.
    Float[Array, "H W"]
        Transverse y-component of the wavevector grid in reciprocal micrometers.
    Complex[Array, "H W"]
        Longitudinal z-component of the wavevector grid in reciprocal micrometers.
    Float[Array, "H W"]
        Pupil mask, 1.0 inside the disc and 0.0 outside.
    """
    H, W = grid_shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(W, d=pixel_size_sample_um)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(H, d=pixel_size_sample_um)
    KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
    KZ = jnp.sqrt(k**2 - KX**2 - KY**2 + 0j)
    k_perp_sq = KX**2 + KY**2
    # Only strictly propagating waves carry power; intersecting the pupil with
    # the propagating cone also handles objectives with NA > n_medium
    # gracefully (the homogeneous-medium model cannot collect evanescent
    # components) and avoids the singular KZ -> 0 ring at the pupil edge.
    propagating = k_perp_sq < k**2
    if na is not None:
        # The objective collects a cone of half-angle theta_max with
        # sin(theta_max) = NA / n_medium; the corresponding transverse
        # wavevector cutoff is k * sin(theta_max) = (NA / n_medium) * k.
        k_perp_max = na * k / n_medium
        mask = propagating & (k_perp_sq <= k_perp_max**2)
    else:
        mask = propagating
    KZ = jnp.where(mask, KZ, 1.0)
    return KX, KY, KZ, mask.astype(jnp.float32)


def _incident_pupil_field(
    KX: Float[Array, "H W"],
    KY: Float[Array, "H W"],
    KZ: Complex[Array, "H W"],
    mask: Float[Array, "H W"],
    beam: Beam,
    objective: Objective,
    k: float,
) -> Complex[Array, "H W 3"]:
    """
    Build the incident vector field in the pupil (before reflection at sample).

    Parameters
    ----------
    KX : Float[Array, "H W"]
        Transverse x-component of the wavevector grid in reciprocal micrometers.
    KY : Float[Array, "H W"]
        Transverse y-component of the wavevector grid in reciprocal micrometers.
    KZ : Complex[Array, "H W"]
        Longitudinal z-component of the wavevector grid in reciprocal micrometers.
    mask : Float[Array, "H W"]
        Pupil mask, 1.0 inside the objective disc and 0.0 outside.
    beam : Beam
        The illumination beam; sets the polarization, beam shape, and power.
    objective : Objective
        The imaging objective; provides the collection NA.
    k : float
        Wavenumber in the medium in reciprocal micrometers.

    Returns
    -------
    Complex[Array, "H W 3"]
        Vector pupil field, with polarization projected onto the plane
        perpendicular to the unit Ewald-sphere vector and the aplanatic
        apodization ``sqrt(kz/k)`` applied.
    """
    pol = jnp.array(beam.polarization, dtype=jnp.float32)
    pol = pol / jnp.linalg.norm(pol)

    # Unit wavevector on the Ewald sphere: k_hat = (KX, KY, KZ) / k.
    k_hat = jnp.stack([KX, KY, KZ], axis=-1) / k

    # Project polarization onto the plane perpendicular to k_hat.
    k_dot_pol = jnp.einsum("hwc,c->hw", k_hat, pol)
    pol_perp = pol - k_dot_pol[..., None] * k_hat

    # Amplitude profile.
    k_perp2 = KX**2 + KY**2
    if beam.beam_type == "gaussian":
        w0 = beam.focal_waist_um()
        sigma_k = 2.0 / (k * w0)
        amplitude = jnp.exp(-k_perp2 / (2.0 * (k * sigma_k) ** 2))
    else:
        amplitude = jnp.ones_like(KX)

    # Aplanatic apodization for an aplanatic objective.
    apod = jnp.sqrt(jnp.real(KZ) / k)
    amplitude = amplitude * apod * mask

    return amplitude[..., None] * pol_perp


def image_field_from_pupil(
    E_pupil: Complex[Array, "H W 3"],
    grid_shape: tuple[int, int],
    pixel_size_sample_um: float,
) -> Complex[Array, "H W 3"]:
    """
    Propagate a vector pupil field to the image plane via inverse FFT.

    Parameters
    ----------
    E_pupil : Complex[Array, "H W 3"]
        Vector pupil field to propagate, shape ``(H, W, 3)``.
    grid_shape : tuple[int, int]
        Image grid shape ``(H, W)``.
    pixel_size_sample_um : float
        Sample-space pixel size in micrometers.

    Returns
    -------
    Complex[Array, "H W 3"]
        Image-plane vector field, centered via ``fftshift``.
    """
    H, W = grid_shape
    Lx = W * pixel_size_sample_um
    Ly = H * pixel_size_sample_um
    # Continuous IFT normalization factor.
    norm = (H * W) / (Lx * Ly)
    E_img = jnp.fft.ifft2(E_pupil, axes=(0, 1)) * norm
    return jnp.fft.fftshift(E_img, axes=(0, 1))


def focused_incident_field(
    positions: Float[Array, "N 3"],
    beam: Beam,
    objective: Objective,
    n_medium: float | None = None,
    wavelength_um: float | None = None,
    num_k: int = 128,
) -> Complex[Array, "N 3"]:
    """
    Compute the focused incident field at arbitrary 3D points.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Points where the field is evaluated, in micrometers.
    beam : Beam
        The illumination beam; sets the wavelength, polarization, beam shape,
        and axial focus.
    objective : Objective
        The focusing objective; provides the collection NA and hence the
        half-angle ``sin(theta_max) = NA / n_medium`` of the focused cone.
    n_medium : float | None
        Medium refractive index.  Defaults to ``beam.n_medium``.
    wavelength_um : float | None
        Vacuum wavelength in micrometers.  Defaults to ``beam.wavelength_um``.
    num_k : int
        Number of pupil quadrature points along each axis.

    Returns
    -------
    Complex[Array, "N 3"]
        Vector incident electric field at each point.
    """
    if n_medium is None:
        n_medium = float(beam.n_medium)
    if wavelength_um is None:
        wavelength_um = float(beam.wavelength_um)
    k = 2.0 * jnp.pi * n_medium / wavelength_um
    # The transverse pupil extent cannot exceed the propagating cone in the
    # medium; cap NA at n_medium so NA > n does not pull in evanescent waves.
    k_max = min(float(objective.na * k / n_medium), k)
    kx = jnp.linspace(-k_max, k_max, num_k)
    ky = jnp.linspace(-k_max, k_max, num_k)
    KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
    KZ = jnp.sqrt(k**2 - KX**2 - KY**2 + 0j)
    mask = KX**2 + KY**2 <= k_max**2
    KZ = jnp.where(mask, KZ, 1.0)

    pol = jnp.array(beam.polarization, dtype=jnp.float32)
    pol = pol / jnp.linalg.norm(pol)
    # Unit wavevector on the Ewald sphere.
    k_hat = jnp.stack([KX, KY, KZ], axis=-1) / k
    k_dot_pol = jnp.einsum("hwc,c->hw", k_hat, pol)
    pol_perp = pol - k_dot_pol[..., None] * k_hat

    k_perp2 = KX**2 + KY**2
    if beam.beam_type == "gaussian":
        w0 = beam.focal_waist_um()
        sigma_k = 2.0 / (k * w0)
        amplitude = jnp.exp(-k_perp2 / (2.0 * (k * sigma_k) ** 2))
    else:
        amplitude = jnp.ones_like(KX)

    apod = jnp.sqrt(jnp.real(KZ) / k)
    E_pupil = amplitude[..., None] * apod[..., None] * pol_perp * mask[..., None]

    # Focus/defocus phase.
    z_focus = beam.z_focus_um
    phases = jnp.exp(
        1j
        * (
            KX[None, :, :] * (positions[:, 0, None, None] - 0.0)
            + KY[None, :, :] * (positions[:, 1, None, None] - 0.0)
            + KZ[None, :, :] * (positions[:, 2, None, None] - z_focus)
        )
    )

    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    # Continuous inverse Fourier transform convention e^{ik.r} / (2 pi)^2,
    # matching the one used by :func:`image_field_from_pupil`.
    return jnp.einsum("nhw,hwc->nc", phases, E_pupil) * (dkx * dky / (2.0 * jnp.pi) ** 2)


def dipole_far_field(
    positions: Float[Array, "N 3"],
    dipole_moments: Complex[Array, "N 3"],
    directions: Float[Array, "M 3"],
    k: float,
) -> Complex[Array, "M 3"]:
    """
    Free-space far-field scattering amplitude of a dipole array.

    Returns the vector amplitude ``F(k_hat)`` such that the electric field
    radiated by the dipoles in the direction ``k_hat`` (with ``|k_hat| = 1``)
    at a large distance ``r`` is

        E_far(r, k_hat) = (exp(i k r) / r) F(k_hat),

    with

        F(k_hat) = k^2 * sum_j [p_j - (k_hat . p_j) k_hat] exp(-i k . r_j).

    The differential scattering cross-section follows as
    ``dC_sca/dOmega = |F(k_hat)|^2 / |E_inc|^2`` and the total scattering
    cross-section is ``C_sca = integral |F|^2 / |E_inc|^2 dOmega``.  This is the
    same far-field convention used internally by :func:`_dipole_field_on_grid`,
    so the two are directly comparable.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Dipole positions in micrometers.
    dipole_moments : Complex[Array, "N 3"]
        Dipole moments in micrometers cubed (times the local field).
    directions : Float[Array, "M 3"]
        Observation directions; normalized internally.  Need not be unit
        vectors.
    k : float
        Wavenumber in micrometers^-1.

    Returns
    -------
    Complex[Array, "M 3"]
        Far-field amplitude ``F`` at each direction, as a Cartesian
        3-vector.
    """
    khat = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    kf = float(k)
    # Phase factor exp(-i k k_hat . r_j) for each dipole and direction: (M, N).
    phases = jnp.exp(-1j * kf * jnp.einsum("mc,nc->mn", khat, positions))
    # Transverse dipole moment: p_j - (k_hat . p_j) k_hat.
    kdotp = jnp.einsum("mc,nc->mn", khat, dipole_moments)  # (M, N)
    p_perp = dipole_moments[None, :, :] - kdotp[..., None] * khat[:, None, :]
    return kf**2 * jnp.einsum("mn,mni->mi", phases, p_perp)


def _dipole_field_on_grid(
    positions: Float[Array, "N 3"],
    dipole_moments: Complex[Array, "N 3"],
    n_medium: float,
    k: float,
    na: float,
    grid_shape: tuple[int, int],
    pixel_size_sample_um: float,
    z_plane_um: float = 0.0,
) -> Complex[Array, "H W 3"]:
    """
    Compute the vector image-plane field radiated by a set of dipoles.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Dipole positions in micrometers.
    dipole_moments : Complex[Array, "N 3"]
        Solved complex dipole moments, shape ``(N, 3)``.
    n_medium : float
        Refractive index of the surrounding medium.
    k : float
        Wavenumber in the medium in reciprocal micrometers.
    na : float
        Numerical aperture of the collecting objective.
    grid_shape : tuple[int, int]
        Image grid shape ``(H, W)``.
    pixel_size_sample_um : float
        Sample-space pixel size in micrometers.
    z_plane_um : float
        Axial image-plane position relative to nominal focus, in micrometers.

    Returns
    -------
    Complex[Array, "H W 3"]
        Vector image-plane field, shape ``(H, W, 3)``.
    """
    KX, KY, KZ, mask = _fft_pupil_grid(grid_shape, pixel_size_sample_um, k, na, n_medium)
    # Unit wavevector on the Ewald sphere.
    k_hat = jnp.stack([KX, KY, KZ], axis=-1) / k

    # Phase factor exp(-i k · r_j) for each dipole.
    phases = jnp.exp(
        -1j
        * (
            KX[:, :, None] * positions[None, None, :, 0]
            + KY[:, :, None] * positions[None, None, :, 1]
            + KZ[:, :, None] * positions[None, None, :, 2]
        )
    )

    # Perpendicular component of each dipole moment: p - (k_hat·p) k_hat.
    k_dot_p = jnp.einsum("hwc,nc->hwn", k_hat, dipole_moments)
    p_perp = dipole_moments[None, None, :, :] - k_dot_p[..., None] * k_hat[:, :, None, :]

    # Far-field angular spectrum with k^2 prefactor.
    angular_spectrum = k**2 * jnp.einsum("hwn,hwni->hwi", phases, p_perp)

    # Aplanatic apodization and defocus.
    apod = jnp.sqrt(jnp.real(KZ) / k)
    defocus = jnp.exp(1j * KZ * z_plane_um)
    E_pupil = angular_spectrum * apod[..., None] * defocus[..., None] * mask[..., None]

    return image_field_from_pupil(E_pupil, grid_shape, pixel_size_sample_um)


def iscat_contrast(
    scattered_field: Complex[Array, "H W 3"],
    reference_field: Complex[Array, "H W"],
    polarization: tuple[float, float, float] | None = None,
    linear: bool = False,
) -> dict[str, Array]:
    """
    Compute the iSCAT intensity, contrast, and linear contrast.

    Parameters
    ----------
    scattered_field : Complex[Array, "H W 3"]
        Vector scattered field at the image plane.
    reference_field : Complex[Array, "H W"]
        Scalar reference field at the image plane.
    polarization : tuple[float, float, float] | None
        Detection polarization.  Defaults to x-polarization.
    linear : bool
        If True, also return the linearized contrast ``2 Re(E_s/E_r)``.

    Returns
    -------
    dict[str, Array]
        Keys ``intensity``, ``contrast``, and optionally ``contrast_linear``.
    """
    if polarization is None:
        polarization = (1.0, 0.0, 0.0)
    pol = jnp.array(polarization, dtype=jnp.float32)
    pol = pol / jnp.linalg.norm(pol)
    E_s_proj = jnp.einsum("hwc,c->hw", scattered_field, pol)

    intensity = jnp.abs(reference_field + E_s_proj) ** 2
    intensity_ref = jnp.abs(reference_field) ** 2
    contrast = (intensity - intensity_ref) / jnp.maximum(intensity_ref, 1e-30)
    result = {"intensity": intensity, "contrast": contrast}
    if linear:
        result["contrast_linear"] = 2.0 * jnp.real(
            E_s_proj / jnp.where(jnp.abs(reference_field) > 1e-30, reference_field, 1e-30)
        )
    return result


def capture(
    camera: Camera,
    key: jax.Array,
    photons: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Convenience wrapper around :meth:`Camera.capture`.

    Parameters
    ----------
    camera : Camera
        The camera model.
    key : jax.Array
        JAX PRNG key.
    photons : Float[Array, "..."]
        Ideal photon count per pixel.

    Returns
    -------
    Float[Array, "..."]
        Noisy digital image in ADU.
    """
    return camera.capture(key, photons)
