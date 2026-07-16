"""Tests for :func:`toolsandogh.canonicalize_video` coordinate handling."""

import numpy as np
import pytest
import xarray as xr

from toolsandogh import canonicalize_video
from toolsandogh._validate_video import validate_video


def test_coordinates_generated_from_scales() -> None:
    """Axes without an explicit coordinate are built from the scales."""
    v = canonicalize_video(np.zeros((4, 1, 1, 5, 5)), dt=2.0, dx=0.5, dy=0.5)
    np.testing.assert_allclose(v["T"].values, np.arange(4) * 2.0)
    np.testing.assert_allclose(v["X"].values, np.arange(5) * 0.5)
    np.testing.assert_allclose(v["Y"].values, np.arange(5) * 0.5)
    np.testing.assert_allclose(v["Z"].values, np.array([0.0]))


def test_coordinate_origins() -> None:
    """The ``t0``/``z0``/``y0``/``x0`` arguments offset the first sample."""
    v = canonicalize_video(np.zeros((3, 1, 1, 4, 4)), t0=10.0, y0=1.0, x0=2.0)
    np.testing.assert_allclose(v["T"].values, 10.0 + np.arange(3) * (1000.0 / 60.0))
    np.testing.assert_allclose(v["Y"].values, 1.0 + np.arange(4))
    np.testing.assert_allclose(v["X"].values, 2.0 + np.arange(4))


def test_default_time_spacing() -> None:
    """Time defaults to 60 fps (``1000 / 60`` ms) when ``dt`` is omitted."""
    v = canonicalize_video(np.zeros((3, 1, 1, 2, 2)))
    np.testing.assert_allclose(v["T"].values[1] - v["T"].values[0], 1000.0 / 60.0)


def test_explicit_coordinates_preserved() -> None:
    """Reader-supplied physical coordinates are kept verbatim."""
    t = np.arange(4) * (1000.0 / 60.0)
    y = np.arange(5) * 0.108
    x = np.arange(5) * 0.108
    src = xr.DataArray(
        np.zeros((4, 1, 1, 5, 5)),
        dims=("T", "C", "Z", "Y", "X"),
        coords={"T": t, "Y": y, "X": x},
    )
    v = canonicalize_video(src)
    np.testing.assert_allclose(v["T"].values, t)
    np.testing.assert_allclose(v["Y"].values, y)
    np.testing.assert_allclose(v["X"].values, x)


def test_explicit_scale_agreeing_with_coordinate_is_accepted() -> None:
    """An explicit scale that matches the coordinate spacing is accepted."""
    t = np.arange(4) * 5.0
    src = xr.DataArray(
        np.zeros((4, 1, 1, 3, 3)),
        dims=("T", "C", "Z", "Y", "X"),
        coords={"T": t},
    )
    # ``dt=5.0`` agrees with the coordinate spacing; no error, and the
    # coordinate values are retained (not regenerated).
    v = canonicalize_video(src, dt=5.0)
    np.testing.assert_allclose(v["T"].values, t)


def test_explicit_scale_conflicting_with_coordinate_raises() -> None:
    """An explicit scale that disagrees with the coordinate spacing raises."""
    t = np.arange(4) * 5.0
    src = xr.DataArray(
        np.zeros((4, 1, 1, 3, 3)),
        dims=("T", "C", "Z", "Y", "X"),
        coords={"T": t},
    )
    with pytest.raises(ValueError, match="does not match"):
        canonicalize_video(src, dt=6.0)


def test_explicit_origin_conflicting_with_coordinate_raises() -> None:
    """An explicit origin that disagrees with the coordinate origin raises."""
    t = np.arange(4) * 5.0
    src = xr.DataArray(
        np.zeros((4, 1, 1, 3, 3)),
        dims=("T", "C", "Z", "Y", "X"),
        coords={"T": t},
    )
    with pytest.raises(ValueError, match="does not match"):
        canonicalize_video(src, t0=1.0)


def test_units_attached() -> None:
    """Every TZYX axis carries the canonical physical units."""
    v = canonicalize_video(np.zeros((1, 1, 1, 2, 2)))
    assert v["T"].attrs["units"] == "ms"
    assert v["Z"].attrs["units"] == "µm"
    assert v["Y"].attrs["units"] == "µm"
    assert v["X"].attrs["units"] == "µm"


def test_c_axis_has_no_units() -> None:
    """The channel axis is categorical and carries no units."""
    v = canonicalize_video(np.zeros((1, 1, 1, 2, 2)))
    assert "units" not in v["C"].attrs


def test_canonicalize_rejects_non_uniform_explicit_coords() -> None:
    """A non-uniform reader-supplied coordinate must be rejected."""
    bad = xr.DataArray(
        np.zeros((4, 1, 1, 3, 3)),
        dims=("T", "C", "Z", "Y", "X"),
        coords={"T": np.array([0.0, 2.0, 3.0, 4.0])},  # first gap 2, then 1
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        canonicalize_video(bad)


def test_validate_rejects_non_uniform_spacing() -> None:
    """``validate_video`` flags a corrupted (non-uniform) axis."""
    v = canonicalize_video(np.zeros((4, 1, 1, 3, 3), dtype=np.float32))
    # Corrupt the T axis after canonicalization so the first and middle
    # spacings disagree.
    v = v.assign_coords({"T": np.array([0.0, 2.0, 3.0, 4.0])})
    with pytest.raises(ValueError, match="uniformly spaced"):
        validate_video(v)


def test_canonicalize_is_idempotent() -> None:
    """Re-canonicalizing a canonical video preserves its coordinates."""
    v = canonicalize_video(np.zeros((3, 1, 1, 4, 4)), dt=2.0, dx=0.5, t0=1.0, x0=0.5)
    v2 = canonicalize_video(v)
    np.testing.assert_allclose(v["T"].values, v2["T"].values)
    np.testing.assert_allclose(v["X"].values, v2["X"].values)
    assert v2["T"].attrs["units"] == "ms"
    assert v2["X"].attrs["units"] == "µm"
