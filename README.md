# toolsandogh

A Python toolbox for iSCAT microscopy data analysis, developed by the Sandoghdar Division of the Max Planck Institute for the Science of Light (MPL).

## Overview

`toolsandogh` provides a small, reusable library for loading, processing, analyzing, and storing microscopy videos.  It is designed around a canonical 5D data model---`T` (time), `C` (channel), `Z` (slice), `Y` (row), `X` (column)---implemented as lazy [xarray](https://docs.xarray.dev/) arrays backed by [Dask](https://www.dask.org/).

## Features

- **Lazy, out-of-core computation**: Videos are kept as Dask arrays so you can work with datasets larger than memory.
- **Labeled, metadata-rich arrays**: Results are `xarray.DataArray` objects with `TCZYX` dimensions and OME metadata.
- **Multiple microscopy formats**: Read and write RAW/BIN, TIFF, OME-TIFF, OME-Zarr, ND2, and other formats supported by [bioio](https://github.com/bioio-devs/bioio).
- **Reusable analysis primitives**: Rolling statistics, radial variance transform, synthetic data generation, and more.

## Installation

This project uses the [uv package manager](https://docs.astral.sh/uv/).  Clone the repository and run:

```bash
uv sync
```

To install in editable mode for development:

```bash
uv pip install -e ".[dev]"
```

## Quick start

```python
import toolsandogh as sand

# Load a video (any bioio-supported format or raw BIN/RAW file).
video = sand.load_video("movie.nd2")

# Apply a rolling average along the time axis.
smoothed = sand.rolling_average(video, window_size=5, dim="T")

# Save the result back to disk.
sand.store_video(smoothed, "smoothed.ome.tiff")
```

Use `help(sand.load_video)` or `help(sand.store_video)` for details on the available arguments.

## Repository layout

- `src/toolsandogh/` --- the main Python package.
- `src/toolsandogh/tests/` --- `pytest` test suite.
- `DESIGN.md` --- design conventions and coding standards.

## Contributing

Please see `DESIGN.md` for coding conventions, documentation standards, and architecture decisions.  In brief:

- All public functions must be type-hinted and documented in numpydoc style.
- Code is linted with `ruff` and type-checked with `pyright`.
- Prefer free functions; keep data in `xarray.DataArray`/`Dataset` form.
- Tests go in `src/toolsandogh/tests/`.

## License

GPL-3.0-or-later.
