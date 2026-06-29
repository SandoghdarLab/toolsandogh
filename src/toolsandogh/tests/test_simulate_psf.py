"""Tests for the JAX/DDA iSCAT PSF simulation module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

import toolsandogh as tog


@pytest.fixture
def objective():
    return tog.Objective(na=1.4, magnification=60.0, immersion_index=1.518)


@pytest.fixture
def beam():
    return tog.Beam(
        wavelength_um=0.532,
        power_w=1e-3,
        n_medium=1.33,
        polarization=(1.0, 0.0, 0.0),
        beam_type="gaussian",
        waist_um=0.3,
    )


@pytest.fixture
def camera():
    return tog.Camera(
        pixel_size_um=6.5,
        full_well_capacity_e=20_000.0,
        quantum_efficiency=0.9,
        exposure_time_s=0.1,
    )


def test_polarizability_real():
    alpha = tog.polarizability(n_particle=1.59, n_medium=1.33, spacing_um=0.05, wavelength_um=0.532)
    assert alpha.real > 0
    assert abs(alpha.imag / alpha.real) < 1e-2


def test_dda_single_dipole():
    """For a single dipole, the DDA solution must equal α E_inc."""
    alpha = 1e-6 + 1e-7j
    positions = jnp.array([[0.0, 0.0, 0.0]])
    alphas = jnp.array([[alpha, alpha, alpha]], dtype=jnp.complex64)
    E_inc = jnp.array([[1.0 + 0.5j, 0.2j, -0.1]], dtype=jnp.complex64)
    k = 10.0
    P = tog.dda_solve(E_inc, positions, alphas, k)
    assert P.shape == (1, 3)
    assert jnp.allclose(P[0], alpha * E_inc[0], rtol=1e-5)


def test_dipole_lattice_sphere():
    positions = tog.dipole_lattice_sphere(radius_um=0.1, spacing_um=0.03)
    assert positions.shape[1] == 3
    assert positions.shape[0] > 0
    # All points inside sphere.
    assert jnp.all(jnp.sum(positions**2, axis=-1) <= 0.1**2 + 1e-6)


def test_focused_incident_field_peak_at_focus(beam, objective):
    n_medium = 1.33
    # Evaluate along x and y through focus.
    x = jnp.linspace(-0.05, 0.05, 101)
    positions_x = jnp.stack([x, jnp.zeros_like(x), jnp.zeros_like(x)], axis=-1)
    E_x = tog.focused_incident_field(
        positions=positions_x,
        beam=beam,
        objective=objective,
        n_medium=n_medium,
        num_k=128,
    )
    # |E_x| peaks at the focus (index 50 = x=0) along x.  The original
    # assertion (peak at the edge, index 0) encoded the buggy non-unit
    # k_hat construction, which produced an unphysical off-centre maximum.
    assert abs(int(jnp.argmax(jnp.abs(E_x[:, 0]))) - 50) <= 2

    y = jnp.linspace(-0.05, 0.05, 101)
    positions_y = jnp.stack([jnp.zeros_like(y), y, jnp.zeros_like(y)], axis=-1)
    E_y = tog.focused_incident_field(
        positions=positions_y,
        beam=beam,
        objective=objective,
        n_medium=n_medium,
        num_k=128,
    )
    # |E_x| peaks at the center along y.
    assert abs(jnp.argmax(jnp.abs(E_y[:, 0])).item() - 50) <= 1


def test_reference_field_nonzero(beam, objective):
    grid_shape = (64, 64)
    pixel_size_sample_um = 0.1
    E_r = tog.reference_field(beam, objective, grid_shape, pixel_size_sample_um)
    assert E_r.shape == grid_shape
    assert jnp.max(jnp.abs(E_r)) > 0


def test_scattered_field_small_particle(beam, objective):
    particle = tog.Particle.sphere(
        radius_um=0.02,
        n_particle=1.59,
        n_medium=1.33,
        spacing_um=0.02,
        wavelength_um=0.532,
    )
    sample = tog.Sample(medium_index=1.33, particles=[particle])
    E_s = tog.scattered_field(sample, beam, objective, grid_shape=(64, 64))
    assert E_s.shape == (64, 64, 3)
    # A small particle produces a finite signal.
    assert jnp.max(jnp.abs(E_s)) > 0


def test_simulate_iscat_full_pipeline(beam, objective, camera):
    particle = tog.Particle.sphere(
        radius_um=0.03,
        n_particle=1.59,
        n_medium=1.33,
        spacing_um=0.02,
        wavelength_um=0.532,
    )
    sample = tog.Sample(medium_index=1.33, particles=[particle])
    result = tog.simulate_iscat(
        sample=sample,
        beam=beam,
        objective=objective,
        camera=camera,
        grid_shape=(64, 64),
    )
    assert result.scattered_field.shape == (64, 64, 3)
    assert result.reference_field.shape == (64, 64)
    assert result.contrast.shape == (64, 64)
    assert jnp.all(jnp.isfinite(result.contrast))
    assert result.camera_image is None


def test_simulate_iscat_returns_camera_image(beam, objective, camera):
    """
    ``return_camera_image=True`` attaches a noisy image to the result.

    Parameters
    ----------
    beam : Beam
        The illumination beam fixture.
    objective : Objective
        The imaging objective fixture.
    camera : Camera
        The camera fixture.
    """
    particle = tog.Particle.sphere(
        radius_um=0.03,
        n_particle=1.59,
        n_medium=1.33,
        spacing_um=0.02,
        wavelength_um=0.532,
    )
    sample = tog.Sample(medium_index=1.33, particles=[particle])
    key = jax.random.PRNGKey(0)
    result = tog.simulate_iscat(
        sample=sample,
        beam=beam,
        objective=objective,
        camera=camera,
        grid_shape=(64, 64),
        key=key,
        return_camera_image=True,
    )
    assert result.camera_image is not None
    assert result.camera_image.shape == (64, 64)
    assert jnp.all(jnp.isfinite(result.camera_image))


def test_simulate_iscat_stack_returns_canonical_video(beam, objective, camera):
    """
    ``simulate_iscat_stack`` yields a canonical TCZYX video of contrast.

    The DDA must be solved exactly once regardless of the number of slices;
    the returned :class:`xarray.DataArray` must pass
    :func:`~toolsandogh.canonicalize_video` validation so that it composes with
    :func:`~toolsandogh.store_video` and the rest of the toolbox.

    Parameters
    ----------
    beam : Beam
        The illumination beam fixture.
    objective : Objective
        The imaging objective fixture.
    camera : Camera
        The camera fixture.
    """
    import dask.array as da

    from toolsandogh._validate_video import validate_video

    particle = tog.Particle.sphere(
        radius_um=0.03,
        n_particle=1.59,
        n_medium=1.33,
        spacing_um=0.02,
        wavelength_um=0.532,
    )
    sample = tog.Sample(medium_index=1.33, particles=[particle])
    z_planes = jnp.linspace(-1.0, 1.0, 5)

    video = tog.simulate_iscat_stack(
        sample=sample,
        beam=beam,
        objective=objective,
        camera=camera,
        grid_shape=(64, 64),
        z_planes_um=z_planes,
        field="contrast",
    )
    # Canonical TCZYX layout with Z matching the number of planes.
    assert video.dims == ("T", "C", "Z", "Y", "X")
    assert video.sizes == {"T": 1, "C": 1, "Z": 5, "Y": 64, "X": 64}
    assert isinstance(video.data, da.Array)
    validate_video(video)
    # The on-axis contrast must be finite for every defocus plane.
    contrast = np.asarray(video[0, 0, :, 32, 32])
    assert np.all(np.isfinite(contrast))


def test_simulate_iscat_stack_matches_single_plane(beam, objective, camera):
    """
    Each slice of the stack must match the corresponding single-plane call.

    This guards against the DDA being re-solved (which would be wasteful) and
    against any drift between the vmap'd propagation path and the scalar one.

    Parameters
    ----------
    beam : Beam
        The illumination beam fixture.
    objective : Objective
        The imaging objective fixture.
    camera : Camera
        The camera fixture.
    """
    particle = tog.Particle.sphere(
        radius_um=0.02,
        n_particle=1.59,
        n_medium=1.33,
        spacing_um=0.02,
        wavelength_um=0.532,
    )
    sample = tog.Sample(medium_index=1.33, particles=[particle])
    z_planes = jnp.linspace(-0.5, 0.5, 3)

    stack = tog.simulate_iscat_stack(
        sample=sample,
        beam=beam,
        objective=objective,
        camera=camera,
        grid_shape=(64, 64),
        z_planes_um=z_planes,
        field="contrast",
    )
    for i, zp in enumerate(z_planes):
        single = tog.simulate_iscat(
            sample=sample,
            beam=beam,
            objective=objective,
            camera=camera,
            grid_shape=(64, 64),
            z_plane_um=float(zp),
        )
        a = np.asarray(stack[0, 0, i])
        b = np.asarray(single.contrast)
        assert np.allclose(a, b, atol=1e-5), f"slice {i} mismatch"


def test_camera_capture_shape(camera):
    key = jnp.array([0, 0], dtype=jnp.uint32)
    photons = jnp.ones((32, 32)) * 1000.0
    img = tog.capture(camera, key, photons)
    assert img.shape == (32, 32)
    assert jnp.all(img >= 0)


# ---------------------------------------------------------------------------
# Validation tests for the NA-aperture and Fourier-convention fixes.
# ---------------------------------------------------------------------------


@pytest.fixture
def plane_beam():
    """
    A plane-wave beam (uniform pupil) in a unit-index medium.

    Returns
    -------
    Beam
        A unit-power plane-wave beam with normal Fresnel-like reflection.
    """
    return tog.Beam(
        wavelength_um=0.532,
        power_w=1.0,
        n_medium=1.0,
        polarization=(1.0, 0.0, 0.0),
        beam_type="plane",
        reflection_coefficient=1.0,
    )


def test_reference_field_airy_first_zero(plane_beam):
    """
    The reference PSF of a plane-wave beam must match the Airy disc.

    A uniform pupil of radius ``NA*k0`` produces a coherent Airy pattern whose
    first zero lies at ``0.61 * lambda / NA``.  Before the NA-aperture fix this
    test fails by roughly an order of magnitude because the propagating-wave
    cutoff (``k``) rather than the objective pupil (``NA*k``) was used, which
    widens the pupil and shrinks the PSF by a factor of ``1/NA``.

    Parameters
    ----------
    plane_beam : Beam
        The plane-wave beam fixture.
    """
    na = 0.5
    objective = tog.Objective(na=na, magnification=1.0)
    # A fine grid so that the zero is resolved to better than a few percent.
    pixel_size_um = 0.005
    grid_shape = (2048, 2048)
    E_r = tog.reference_field(plane_beam, objective, grid_shape, pixel_size_um, z_plane_um=0.0)
    # The PSF is slightly anisotropic for vectorial x-polarized illumination;
    # the first zero is smallest along y (closest to the scalar jinc) and
    # largest along x.  Use the y-axis where the scalar Airy result applies.
    import numpy as np

    arr = np.asarray(E_r)
    H, W = arr.shape
    mid_y, mid_x = H // 2, W // 2
    re = arr[:, mid_x].real  # type: ignore
    ys = (np.arange(H) - mid_y) * pixel_size_um
    measured = None
    for i in range(len(ys) - 1):
        if ys[i] > 0 and re[i] * re[i + 1] < 0:
            t = re[i] / (re[i] - re[i + 1])
            measured = float(ys[i] + t * (ys[i + 1] - ys[i]))
            break
    assert measured is not None, "No zero crossing found along the +y axis."
    expected = 0.61 * plane_beam.wavelength_um / na
    # Vectorial effects and finite sampling shift the zero by a few percent.
    assert abs(measured - expected) <= 0.08 * expected, (
        f"Airy first zero: measured {measured:.4f} µm, expected {expected:.4f} µm"
    )


def test_reference_field_airy_peak_amplitude(plane_beam):
    """
    Compute the peak amplitude of the scalar coherent Airy pattern.

    With the aplanatic apodization ``sqrt(kz/k) = (1 - (k_perp/k)^2)^{1/4}``
    and the continuous inverse transform ``1/(2*pi)^2 * int``, the on-axis
    field of a *scalar* uniform pupil is

        E(0) = (1 / (2*pi)) * int_0^{NA*k} (1 - (kp/k)^2)^{1/4} kp dkp.

    We compare against a scalar pupil built directly through
    :func:`image_field_from_pupil`, which isolates the normalization from the
    vectorial polarization-projection corrections.

    Parameters
    ----------
    plane_beam : Beam
        The plane-wave beam fixture.
    """

    from toolsandogh._simulate_psf import (
        _fft_pupil_grid,
        image_field_from_pupil,
    )

    na = 0.5
    lam = plane_beam.wavelength_um
    k = 2.0 * jnp.pi / lam

    # Analytic scalar peak.
    from scipy.integrate import quad

    expected = float(
        quad(lambda kp: (1.0 - (kp / k) ** 2) ** 0.25 * kp, 0.0, na * k)[0] / (2.0 * jnp.pi)
    )

    pixel_size_um = 0.005
    grid_shape = (2048, 2048)
    H, W = grid_shape
    KX, KY, KZ, mask = _fft_pupil_grid(grid_shape, pixel_size_um, k, na)
    apod = jnp.sqrt(jnp.real(KZ) / k)
    pupil = (apod * mask)[..., None] * jnp.array([1.0, 0.0, 0.0])
    image = image_field_from_pupil(pupil, grid_shape, pixel_size_um)
    measured = jnp.max(jnp.abs(image[..., 0]))
    assert jnp.allclose(measured, expected, rtol=2e-2), (
        f"Airy peak: measured {float(measured):.4f}, expected {expected:.4f}"
    )


def test_image_field_from_pupil_parseval():
    """
    Verify Parseval's theorem: the FFT propagator conserves power.

    For a unit-energy pupil, ``sum |image|^2 * dx * dy == sum |pupil|^2 * dkx * dky``.
    This pins down the normalization factor in :func:`image_field_from_pupil`.
    """
    import numpy as np

    from toolsandogh._simulate_psf import image_field_from_pupil

    grid_shape = (256, 256)
    pixel_size_um = 0.05
    H, W = grid_shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(W, d=pixel_size_um)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(H, d=pixel_size_um)
    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])

    rng = jnp.array(np.random.default_rng(0).standard_normal((*grid_shape, 3)) + 0j)
    pupil = rng.astype(jnp.complex64)
    image = image_field_from_pupil(pupil, grid_shape, pixel_size_um)

    dx = dy = pixel_size_um
    # The code implements the continuous inverse transform E(r) = (1/(2*pi)^2)
    # int pupil(k) e^{i k.r} dk, so Parseval reads
    #   int |E(r)|^2 dr = (1/(2*pi)^2) int |pupil(k)|^2 dk.
    pupil_energy = jnp.sum(jnp.abs(pupil) ** 2) * dkx * dky / (2.0 * jnp.pi) ** 2
    image_energy = jnp.sum(jnp.abs(image) ** 2) * dx * dy
    assert jnp.allclose(pupil_energy, image_energy, rtol=1e-4), (
        f"Parseval: pupil energy {float(pupil_energy):.4f}, image energy {float(image_energy):.4f}"
    )


def test_tilt_plane_wave_convention(plane_beam):
    """
    A single tilted plane wave in the pupil maps to a shifted delta in the image.

    A pupil component ``exp(-i kx0 * x)`` must produce, after propagation, a
    peak at ``x = -kx0 * pixel_size^2 / (2*pi)``... equivalently the image of a
    tilted plane ``exp(i k_perp . r)`` is a delta at the corresponding image
    position.  This validates that the FFT-based propagator and the Debye
    integral use the same Fourier sign/normalization convention.

    Parameters
    ----------
    plane_beam : Beam
        The plane-wave beam fixture.
    """

    from toolsandogh._simulate_psf import image_field_from_pupil

    grid_shape = (512, 512)
    pixel_size_um = 0.02
    H, W = grid_shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(W, d=pixel_size_um)

    # A single nonzero pupil sample at kx = kx[k_idx], ky = 0.
    k_idx = W // 2 + 17
    kx0 = float(kx[k_idx])
    pupil = jnp.zeros((H, W, 3), dtype=jnp.complex64)
    pupil = pupil.at[0, k_idx, 0].set(1.0 + 0j)

    image = image_field_from_pupil(pupil, grid_shape, pixel_size_um)
    # The image is a plane wave exp(+i kx0 * x); its |image|^2 is uniform.
    # Check that the phase ramps at the expected rate kx0.
    xs = (jnp.arange(W) - W // 2) * pixel_size_um
    row = image[H // 2, :, 0]
    phases = jnp.unwrap(jnp.angle(row))
    # Linear fit slope.
    slope = float(jnp.polyfit(xs, phases, 1)[0])
    assert abs(slope - kx0) <= 1e-2, f"Phase slope: measured {slope:.4f}, expected {kx0:.4f}"


def test_rayleigh_particle_contrast_sign_and_magnitude():
    """
    Verify that a small Rayleigh particle gives a *dark*, weak, volume-linear contrast.

    Polystyrene (n=1.59) in water (n=1.33) is a weak, lossless scatterer, so
    the on-axis iSCAT contrast must be negative and of order 10^-2 for a 15 nm
    bead.  This is the end-to-end check that catches a missing ``1/(2*pi)^2``
    in the focused-incident-field integral, which inflates the contrast by
    ``(2*pi)^2``.
    """
    import numpy as np

    # The iSCAT reference is the beam reflected at the water/glass
    # coverslip interface.  The Fresnel coefficient of that interface is
    # *negative*, and that pi phase flip is precisely what makes a lossless
    # dielectric particle appear dark in iSCAT.  Using a positive coefficient
    # would be unphysical for this interface and would yield a bright spot.
    reflection_coefficient = (1.33 - 1.52) / (1.33 + 1.52)
    beam = tog.Beam(
        wavelength_um=0.532,
        power_w=1e-3,
        n_medium=1.33,
        polarization=(1.0, 0.0, 0.0),
        beam_type="plane",
        reflection_coefficient=reflection_coefficient,
    )
    objective = tog.Objective(na=1.4, magnification=60.0, immersion_index=1.518)
    camera = tog.Camera(pixel_size_um=6.5, exposure_time_s=0.1)

    a = 0.015  # 15 nm radius
    particle = tog.Particle.sphere(
        radius_um=a,
        n_particle=1.59,
        n_medium=1.33,
        spacing_um=0.008,
        wavelength_um=0.532,
    )
    sample = tog.Sample(medium_index=1.33, particles=[particle])
    result = tog.simulate_iscat(
        sample=sample,
        beam=beam,
        objective=objective,
        camera=camera,
        grid_shape=(128, 128),
    )
    # Contrast at the Airy-pattern centre, where the reference is well-defined.
    contrast = np.asarray(result.contrast)
    centre = contrast[64, 64]
    assert centre < 0, f"Rayleigh particle should be dark, got contrast {centre}"
    assert 1e-3 < abs(centre) < 1e-1, (
        f"15 nm bead contrast {centre} outside the expected Rayleigh range "
        f"[1e-3, 1e-1]; check the focused-field Fourier convention."
    )


def test_contrast_linear_bounded_away_from_reference_zeros():
    """
    Verify that the linear contrast stays bounded away from reference-field zeros.

    ``jnp.maximum(E_r, eps)`` on a complex array silently returns ``eps``
    whenever ``Re(E_r) < 0`` (the Airy pattern oscillates through zero), which
    inflates the linear contrast by ~``1/eps``.  The fix is to floor ``|E_r|``
    instead.  With the fix, the on-axis linear contrast stays finite everywhere.
    """
    import numpy as np

    beam = tog.Beam(
        wavelength_um=0.532,
        n_medium=1.33,
        polarization=(1.0, 0.0, 0.0),
        beam_type="plane",
        reflection_coefficient=0.05,
    )
    objective = tog.Objective(na=1.4, magnification=60.0)
    camera = tog.Camera(pixel_size_um=6.5, exposure_time_s=0.1)
    particle = tog.Particle.sphere(
        radius_um=0.01,
        n_particle=1.59,
        n_medium=1.33,
        spacing_um=0.01,
        wavelength_um=0.532,
    )
    sample = tog.Sample(medium_index=1.33, particles=[particle])
    result = tog.simulate_iscat(
        sample=sample,
        beam=beam,
        objective=objective,
        camera=camera,
        grid_shape=(128, 128),
    )
    cl = np.asarray(result.contrast_linear)
    # Sanity: bounded away from the 1e30 regime that the complex-maximum bug produced.
    assert np.isfinite(cl).all()
    assert np.abs(cl).max() < 1e6, (
        f"contrast_linear diverges ({np.abs(cl).max():.2e}); the reference-field "
        f"floor is probably applied to the complex value instead of its magnitude."
    )


# ---------------------------------------------------------------------------
# DDA-vs-Mie validation.
#
# These tests compare the DDA implementation against analytic Mie theory for a
# lossless dielectric sphere driven by a unit plane wave.  They validate the
# two halves of the simulation chain independently:
#
#   (i)  the total free-space scattering cross-section, obtained by integrating
#        the dipole-array far-field amplitude (``dipole_far_field``) over the
#        full 4*pi sphere --- this exercises the DDA interaction matrix, the
#        lattice polarizability, and the far-field prefactor;
#   (ii) the objective-collected image-plane field, obtained by propagating the
#        *same* solved dipole moments through ``_dipole_field_on_grid`` and
#        ``image_field_from_pupil`` --- this exercises the collection optics
#        (aplanatic apodization, NA aperture, k^2 radiation prefactor,
#        transverse dipole projection, propagation phase, Fourier normalization).
#
# A shared prefactor error (a missing ``1/(2*pi)^2`` or ``1/(4*pi)``) would be
# invisible to (ii) alone, because the DDA and Mie fields share the same
# propagator; (i) compares the DDA against the *analytic* Mie cross-section and
# is therefore the decisive absolute check.  Passing both closes the loop.
# ---------------------------------------------------------------------------


def _mie_ab(m: complex, x: float):
    """
    Compute the Mie coefficients ``a_n``, ``b_n`` (Bohren & Huffman, ``h_n^{(1)}``).

    Parameters
    ----------
    m : complex
        Relative refractive index ``n_particle / n_medium``.
    x : float
        Size parameter ``k_medium * a`` with ``a`` the sphere radius.

    Returns
    -------
    numpy.ndarray
        Multipole orders ``n`` from 1 to ``nmax``.
    numpy.ndarray
        Complex ``a_n`` coefficients.
    numpy.ndarray
        Complex ``b_n`` coefficients.
    """
    nmax = int(np.floor(2 + x + 4 * x ** (1.0 / 3.0)))
    n = np.arange(1, nmax + 1)
    z, mz = x, m * x
    jz, yz = spherical_jn(n, z), spherical_yn(n, z)
    jmz = spherical_jn(n, mz)
    jzp, yzp = spherical_jn(n, z, derivative=True), spherical_yn(n, z, derivative=True)
    jmzp = spherical_jn(n, mz, derivative=True)
    psi, xi = z * jz, z * (jz + 1j * yz)
    psip, xip = jz + z * jzp, (jz + 1j * yz) + z * (jzp + 1j * yzp)
    psim, psimp = mz * jmz, jmz + mz * jmzp
    a = (m * psim * psip - psi * psimp) / (m * psim * xip - xi * psimp)
    b = (psim * psip - m * psi * psimp) / (psim * xip - m * xi * psimp)
    return n, a, b


def _mie_csca(m: complex, x: float, k: float) -> float:
    """
    Compute the total Mie scattering cross-section.

    ``C_sca = (2*pi/k^2) sum (2n + 1) (|a_n|^2 + |b_n|^2)``.

    Parameters
    ----------
    m : complex
        Relative refractive index ``n_particle / n_medium``.
    x : float
        Size parameter ``k_medium * a``.
    k : float
        Medium wavenumber in reciprocal micrometers.

    Returns
    -------
    float
        Total scattering cross-section in square micrometers.
    """
    n, a, b = _mie_ab(m, x)
    return float((2 * np.pi / k**2) * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)))


def _mie_amplitudes(m: complex, x: float, theta: np.ndarray):
    """
    Compute the Mie scattering amplitudes ``S1(theta)`` and ``S2(theta)``.

    Parameters
    ----------
    m : complex
        Relative refractive index ``n_particle / n_medium``.
    x : float
        Size parameter ``k_medium * a``.
    theta : numpy.ndarray
        Scattering angles in radians.

    Returns
    -------
    numpy.ndarray
        Complex ``S1`` amplitude at each angle.
    numpy.ndarray
        Complex ``S2`` amplitude at each angle.
    """
    n_arr, a, b = _mie_ab(m, x)
    nmax = len(n_arr)
    mu = np.cos(theta)
    pin = np.zeros((nmax + 1,) + mu.shape)
    pin[1] = 1.0
    if nmax >= 2:
        pin[2] = 3.0 * mu
    for nn in range(3, nmax + 1):
        pin[nn] = ((2 * nn - 1) * mu * pin[nn - 1] - nn * pin[nn - 2]) / (nn - 1)
    taun = np.zeros_like(pin)
    for nn in range(1, nmax + 1):
        taun[nn] = nn * mu * pin[nn] - (nn + 1) * pin[nn - 1]
    s1 = np.zeros(mu.shape, dtype=complex)
    s2 = np.zeros(mu.shape, dtype=complex)
    for i, nn in enumerate(n_arr):
        c = (2 * nn + 1.0) / (nn * (nn + 1.0))
        s1 += c * (a[i] * pin[nn] + b[i] * taun[nn])
        s2 += c * (a[i] * taun[nn] + b[i] * pin[nn])
    return s1, s2


def _solve_dda_sphere(
    ka: float, n_particle: float, n_medium: float, wavelength_um: float, ppd: int
):
    """
    Solve the DDA for a sphere driven by a unit x-polarized plane wave.

    The plane wave travels along ``+z`` with amplitude ``1``.

    Parameters
    ----------
    ka : float
        Size parameter (product of medium wavenumber and sphere radius).
    n_particle : float
        Real refractive index of the sphere.
    n_medium : float
        Real refractive index of the surrounding medium.
    wavelength_um : float
        Vacuum wavelength in micrometers.
    ppd : int
        Dipoles per diameter of the sphere.

    Returns
    -------
    numpy.ndarray
        Dipole positions in micrometers, shape ``(N, 3)``.
    numpy.ndarray
        Solved complex dipole moments, shape ``(N, 3)``.
    float
        Medium wavenumber in reciprocal micrometers.
    """
    k = 2 * np.pi * n_medium / wavelength_um
    a = ka / k
    spacing = 2 * a / ppd
    particle = tog.Particle.sphere(
        radius_um=a,
        n_particle=n_particle,
        n_medium=n_medium,
        spacing_um=spacing,
        wavelength_um=wavelength_um,
    )
    positions = np.asarray(particle.positions_um)
    alphas = np.asarray(particle.polarizabilities_um3)
    e_inc = np.zeros((len(positions), 3), dtype=complex)
    e_inc[:, 0] = np.exp(1j * k * positions[:, 2])
    p = tog.dda_solve(jnp.asarray(e_inc), jnp.asarray(positions), jnp.asarray(alphas), k)
    return positions, np.asarray(p), k


@pytest.mark.parametrize("ka", [0.3, 1.0, 2.0])
def test_dda_total_cross_section_vs_mie(ka):
    """
    Verify that the total scattering cross-section from solved dipoles matches Mie.

    Drives a lossless dielectric sphere with a unit plane wave, solves the DDA
    self-consistency equation, and integrates the free-space far-field amplitude
    built by :func:`~toolsandogh.dipole_far_field` over the full ``4*pi``
    sphere.  The result is compared to the analytic Mie cross-section.  This is
    the decisive absolute check on the DDA matrix, the lattice polarizability, and
    the far-field prefactor: a missing ``1/(2*pi)^2`` or ``1/(4*pi)`` would fail
    this by a large constant factor rather than the few-percent discretization
    error seen here.

    Parameters
    ----------
    ka : float
        Size parameter (product of medium wavenumber and sphere radius).
    """
    n_med, n_part, lam = 1.33, 1.59, 0.532
    m_rel = n_part / n_med
    positions, p, k = _solve_dda_sphere(ka, n_part, n_med, lam, ppd=12)

    n_theta, n_phi = 300, 96
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    directions = np.stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1
    ).reshape(-1, 3)
    d_omega = (np.sin(th) * (theta[1] - theta[0]) * (phi[1] - phi[0])).reshape(-1)

    f = np.asarray(
        tog.dipole_far_field(jnp.asarray(positions), jnp.asarray(p), jnp.asarray(directions), k)
    )
    c_sca_dda = float(np.sum(np.sum(np.abs(f) ** 2, axis=-1) * d_omega))
    c_sca_mie = _mie_csca(m_rel, ka, k)
    rel = abs(c_sca_dda - c_sca_mie) / c_sca_mie
    assert rel <= 0.08, f"ka={ka}: C_sca DDA={c_sca_dda:.4e} Mie={c_sca_mie:.4e} rel={rel:.3f}"


# ---------------------------------------------------------------------------
# Focused-beam collection-path validation.
#
# The on-axis image-plane field radiated by a single dipole sitting at the
# focus of an aplanatic objective has a closed-form Debye integral.  Comparing
# the full FFT/propagation path (``_dipole_field_on_grid`` +
# ``image_field_from_pupil``) against that integral is the decisive absolute
# check on the collection optics: the ``k^2`` radiation prefactor, the
# transverse dipole projection ``p - (k_hat.p) k_hat`` (built with the unit
# Ewald-sphere vector ``k_hat = (kx,ky,kz)/k``), the aplanatic apodization
# ``sqrt(kz/k) = (1 - (k_perp/k)^2)^{1/4}``, the propagation phase, and ---
# crucially --- the objective pupil radius ``k_perp <= (NA/n_medium) k``.
#
# The Mie image-plane test above cannot catch a pupil-radius error, because
# the DDA and the Mie reference are both propagated through the *same*
# ``_dipole_field_on_grid`` and a shared (wrong) pupil cancels in the L2
# comparison.  Likewise the Airy tests use ``n_medium = 1``, where the
# ``NA*k`` versus ``(NA/n)*k`` ambiguity vanishes.  Only this test, which
# compares against an *analytic* integral with the physical cone
# ``sin(theta) = NA/n_medium`` and runs at ``n_medium != 1``, can catch it.
# ---------------------------------------------------------------------------


def _analytic_onaxis_dipole_field(p_axis: str, na: float, n: float, lam: float) -> float:
    """
    Compute the on-axis image field of a unit dipole at the focus via a direct Debye integral.

    For a dipole ``p`` at the origin (at focus, ``z_plane = 0``) the on-axis
    image-plane field reduces to the zero-frequency bin of the pupil, i.e. the
    continuous inverse transform evaluated at the origin:

        E_s(0) = (k^2 / (2 pi)^2) int_disc [p - (k_hat.p) k_hat]
                 sqrt(k_z / k) d^2 k_perp ,

    with the unit Ewald-sphere vector ``k_hat = (kx, ky, kz)/k`` and the
    collection disc ``k_perp <= (NA / n) * k``.  By symmetry only the
    component along ``p`` survives; carrying out the azimuthal integral gives

        E_s,x(0) = (p_x k^4 / (4 pi)) int_0^{NA/n} (2 - u^2)(1 - u^2)^{1/4} u du
        E_s,z(0) = (p_z k^4 / (2 pi)) int_0^{NA/n} u^3 (1 - u^2)^{1/4} du

    with ``u = k_perp / k`` and ``k = 2 pi n / lambda``.

    Parameters
    ----------
    p_axis : str
        Polarization axis of the unit dipole, either ``"x"`` or ``"z"``.
    na : float
        Objective numerical aperture.
    n : float
        Medium refractive index.
    lam : float
        Vacuum wavelength in micrometers.

    Returns
    -------
    float
        On-axis (image-plane centre) electric field component along ``p_axis``.
    """
    from scipy.integrate import quad

    k = 2.0 * np.pi * n / lam
    u_max = na / n  # physical collection cone: sin(theta_max) = NA / n
    if p_axis == "x":
        integral, _ = quad(lambda u: (2.0 - u**2) * (1.0 - u**2) ** 0.25 * u, 0.0, u_max)
        return float(k**4 / (4.0 * np.pi) * integral)
    # z
    integral, _ = quad(lambda u: u**3 * (1.0 - u**2) ** 0.25, 0.0, u_max)
    return float(k**4 / (2.0 * np.pi) * integral)


@pytest.mark.parametrize("na,n", [(0.5, 1.0), (0.9, 1.33), (1.2, 1.33)])
@pytest.mark.parametrize("p_axis", ["x", "z"])
def test_dipole_onaxis_field_matches_debye_integral(na, n, p_axis):
    """
    Verify that the on-axis collected field of a focused dipole matches the Debye integral.

    A single unit dipole is placed at the focus and its image-plane field is
    propagated through the real collection path.  The on-axis (centre) value
    is compared to the closed-form Debye integral over the objective cone
    ``sin(theta_max) = NA / n_medium``.  This is the only test that pins the
    *absolute* normalization of ``_dipole_field_on_grid`` independently of the
    FFT propagator's own conventions, and the only one that can detect a
    wrong objective pupil radius in a medium with ``n_medium != 1``.

    Parameters
    ----------
    na : float
        Objective numerical aperture.
    n : float
        Medium refractive index.
    p_axis : str
        Polarization axis of the unit dipole, either ``"x"`` or ``"z"``.
    """
    from toolsandogh._simulate_psf import _dipole_field_on_grid

    lam = 0.5
    k = 2.0 * np.pi * n / lam
    if p_axis == "x":
        p = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.complex64)
        comp = 0
    else:
        p = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.complex64)
        comp = 2
    positions = jnp.zeros((1, 3))
    # Fine grid so the FFT Riemann sum approximates the continuous integral.
    grid_shape = (2048, 2048)
    pixel_um = 0.004
    E = np.asarray(
        _dipole_field_on_grid(
            positions, p, float(n), float(k), float(na), grid_shape, pixel_um, 0.0
        )
    )
    H, W = grid_shape
    measured = float(E[H // 2, W // 2, comp].real)
    expected = _analytic_onaxis_dipole_field(p_axis, na, n, lam)
    # ~1-3% residual from k-space Riemann quadrature at this grid resolution.
    assert abs(measured - expected) <= 0.05 * abs(expected), (
        f"NA={na} n={n} {p_axis}-dipole: on-axis field measured={measured:.4e}, "
        f"Debye={expected:.4e}, rel={abs(measured - expected) / abs(expected):.3f}"
    )


@pytest.mark.parametrize("ka", [0.3, 1.0, 2.0])
def test_dda_image_plane_field_vs_mie(ka):
    """
    Verify that the objective-collected scattered field matches the Mie far field.

    Propagates the *same* solved dipole moments through the real collection
    path (``_dipole_field_on_grid`` + ``image_field_from_pupil``, with the
    aplanatic apodization and the NA aperture) and compares the resulting
    image-plane vector field to the field obtained by propagating the analytic
    Mie angular spectrum through the identical optics.  This validates the
    collection optics end to end --- the ``k^2`` radiation prefactor, the
    transverse dipole projection ``p - (k_hat.p) k_hat``, the propagation phase,
    the aplanatic apodization, and the Fourier normalization.

    Parameters
    ----------
    ka : float
        Size parameter (product of medium wavenumber and sphere radius).
    """
    from toolsandogh._simulate_psf import (
        _dipole_field_on_grid,
        _fft_pupil_grid,
        image_field_from_pupil,
    )

    n_med, n_part, lam = 1.33, 1.59, 0.532
    m_rel = n_part / n_med
    positions, p, k = _solve_dda_sphere(ka, n_part, n_med, lam, ppd=12)

    na = 1.2
    grid_shape = (256, 256)
    pixel_um = 0.02

    e_s_dda = np.asarray(
        _dipole_field_on_grid(
            jnp.asarray(positions),
            jnp.asarray(p),
            n_med,
            k,
            na,
            grid_shape,
            pixel_um,
            0.0,
        )
    )

    kx, ky, kz, mask = _fft_pupil_grid(grid_shape, pixel_um, k, na, n_med)
    kx, ky, kz, mask = (np.asarray(v) for v in (kx, ky, kz, mask))
    theta = np.arccos(np.clip(kz.real / k, -1.0, 1.0))  # type: ignore
    phi = np.arctan2(ky, kx)
    s1, s2 = _mie_amplitudes(m_rel, ka, theta)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    theta_hat = np.stack([ct * cp, ct * sp, -st], -1)
    phi_hat = np.stack([-sp, cp, np.zeros_like(sp)], -1)
    # For a unit x-polarized incident plane wave, the Mie far-field amplitude in
    # the code's convention ``E_far = (exp(ikr)/r) F`` is
    # ``F = (i/k) (S2 cosphi theta_hat - S1 sinphi phi_hat)``.
    f_mie = (1j / k) * (
        s2[..., None] * cp[..., None] * theta_hat - s1[..., None] * sp[..., None] * phi_hat
    )
    apod = np.sqrt(np.where(mask, kz.real, 0.0) / k)  # type: ignore
    pupil = f_mie * apod[..., None] * mask[..., None]
    e_s_mie = np.asarray(image_field_from_pupil(jnp.asarray(pupil), grid_shape, pixel_um))

    rel = float(np.sqrt(np.sum(np.abs(e_s_dda - e_s_mie) ** 2) / np.sum(np.abs(e_s_mie) ** 2)))
    assert rel <= 0.06, f"ka={ka}: image-plane field relative L2 = {rel:.3f} (> 6%)"
