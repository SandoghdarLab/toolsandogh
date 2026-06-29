"""
ISCAT PSF demo of a 50 nm polystyrene bead at a glass/water interface.

The script runs the full DDA iSCAT pipeline for both x- and y-polarized
illumination at 500 nm and renders the reference intensity, scattered-field
modulus, iSCAT contrast, and linear contrast side by side with Matplotlib.

Run with the project venv:

    .venv/bin/python demos/demo_iscat_50nm_bead.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import toolsandogh as tog

# --- Optical setup -----------------------------------------------------------
# Vacuum wavelength and refractive indices.
WAVELENGTH_UM = 0.5
N_WATER = 1.33
N_GLASS = 1.52
N_POLYSTYRENE = 1.59  # ~ real part at 500 nm; absorption negligible here.

# Fresnel reflection coefficient of the water/glass interface that generates
# the iSCAT reference field: r = (n1 - n2) / (n1 + n2).
reflection_coefficient = (N_WATER - N_GLASS) / (N_WATER + N_GLASS)

# A 50 nm (radius) polystyrene bead resting on the cover slip, i.e. with its
# centre one radius above the interface at z = +radius.
RADIUS_UM = 0.025
CENTER_UM = (0.0, 0.0, RADIUS_UM)

# High-NA water-immersion objective imaging into water.
objective = tog.Objective(
    na=1.2,
    magnification=60.0,
    immersion_index=N_WATER,
)

camera = tog.Camera(
    pixel_size_um=6.5,
    full_well_capacity_e=20_000.0,
    quantum_efficiency=0.9,
    exposure_time_s=1e-3,
)

# Image grid.  Sample-space pixel size = camera pixel / magnification
# = 6.5 / 60 um ~ 0.108 um, comfortably Nyquist for NA = 1.2 at 500 nm.
GRID_SHAPE = (256, 256)
PIXEL_UM = camera.sample_pixel_size_um(objective)


def make_beam(polarization: str) -> tog.Beam:
    pol = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
    }[polarization]
    return tog.Beam(
        wavelength_um=WAVELENGTH_UM,
        power_w=1e-3,
        n_medium=N_WATER,
        polarization=pol,
        beam_type="gaussian",
        # Underfill the back pupil a little so the illumination is a clean
        # Gaussian focus rather than a hard-edged top hat.
        waist_um=0.4,
        z_focus_um=0.0,
        reflection_coefficient=reflection_coefficient,
    )


def make_sample() -> tog.Sample:
    particle = tog.Particle.sphere(
        radius_um=RADIUS_UM,
        n_particle=N_POLYSTYRENE,
        n_medium=N_WATER,
        spacing_um=0.005,  # ~ lambda / 27 in the medium -> a few hundred dipoles
        wavelength_um=WAVELENGTH_UM,
        center_um=CENTER_UM,
    )
    print(f"  discretized into {particle.positions_um.shape[0]} dipoles")
    return tog.Sample(medium_index=N_WATER, particles=[particle])


def run(polarization: str) -> tog.IScatResult:
    print(f"Simulating {polarization}-polarized illumination ...")
    sample = make_sample()
    beam = make_beam(polarization)
    result = tog.simulate_iscat(
        sample=sample,
        beam=beam,
        objective=objective,
        camera=camera,
        grid_shape=GRID_SHAPE,
    )
    return result


def extent_um():
    H, W = GRID_SHAPE
    return (
        -W * PIXEL_UM / 2,
        W * PIXEL_UM / 2,
        -H * PIXEL_UM / 2,
        H * PIXEL_UM / 2,
    )


def main() -> None:
    jax.config.update("jax_enable_x64", False)

    res_x = run("x")
    res_y = run("y")

    ext = extent_um()

    # The interferometric contrast (I - I_ref)/I_ref and its linearised form
    # 2 Re(E_s / E_r) are only meaningful where the reference field is
    # appreciable; at the Airy zero-rings of the reflected focused beam the
    # ratio diverges.  Mask those pixels for display and for the peak report.
    ref_x = jnp.abs(res_x.reference_field) ** 2
    ref_y = jnp.abs(res_y.reference_field) ** 2
    thr = 1e-2  # keep pixels with at least 1% of the peak reference intensity
    mask_x = ref_x > thr * float(jnp.max(ref_x))
    mask_y = ref_y > thr * float(jnp.max(ref_y))

    def masked(arr, mask):
        return jnp.where(mask, arr, jnp.nan)

    panels = [
        ("reference intensity (x-pol)", ref_x, None),
        (
            "scattered field |E_s| (x-pol)",
            jnp.linalg.norm(res_x.scattered_field, axis=-1),
            "viridis",
        ),
        ("iSCAT contrast (x-pol)", masked(res_x.contrast, mask_x), "RdBu_r"),
        ("linear contrast (x-pol)", masked(res_x.contrast_linear, mask_x), "RdBu_r"),
        ("reference intensity (y-pol)", ref_y, None),
        (
            "scattered field |E_s| (y-pol)",
            jnp.linalg.norm(res_y.scattered_field, axis=-1),
            "viridis",
        ),
        ("iSCAT contrast (y-pol)", masked(res_y.contrast, mask_y), "RdBu_r"),
        ("linear contrast (y-pol)", masked(res_y.contrast_linear, mask_y), "RdBu_r"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        arr = jnp.asarray(img)
        if cmap == "RdBu_r":
            vmax = float(jnp.nanmax(jnp.abs(arr)))
            im = ax.imshow(arr, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=+vmax)
        elif title.startswith("reference"):
            im = ax.imshow(arr, extent=ext, origin="lower", cmap="magma")
        else:
            im = ax.imshow(arr, extent=ext, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"iSCAT PSF: {2 * RADIUS_UM * 1e3:.0f} nm polystyrene bead on glass/water "
        f"(@ {WAVELENGTH_UM * 1e3:.0f} nm, NA={objective.na})",
        fontsize=14,
    )
    fig.tight_layout()
    out = __file__.rsplit("/", 1)[0] + "/demo_iscat_50nm_bead.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # On-axis (centre pixel) values: the cleanest single-number summary of
    # the bead's iSCAT signature, free of reference-node artefacts.
    cy, cx = GRID_SHAPE[0] // 2, GRID_SHAPE[1] // 2
    for name, res in (("x", res_x), ("y", res_y)):
        c_center = float(res.contrast[cy, cx])
        cl_center = float(res.contrast_linear[cy, cx])
        c = jnp.where(mask_x if name == "x" else mask_y, res.contrast, 0.0)
        print(
            f"  {name}-pol: on-axis contrast = {c_center:+.3e},"
            f" linear = {cl_center:+.3e};"
            f" masked-image contrast in [{float(c.min()):+.2e}, {float(c.max()):+.2e}]"
        )


if __name__ == "__main__":
    main()
