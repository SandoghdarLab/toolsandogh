# toolsandogh

A collection of single-file Python scripts for iSCAT microscopy data analysis.


## Overview

This repository collects data analysis scripts of the Sandoghdar Division of the Max Planck Institute for the Science of Light (MPL).  It provides standalone Python scripts for processing, analyzing, and visualizing microscopy data.  Each script is designed to perform a specific task without dependencies on other scripts, making them easy to use and integrate into existing workflows.

## Features

- **Self-contained scripts**: Each script works independently
- **uv friendly**: Scripts install their dependencies automatically using the [uv package manager](https://docs.astral.sh/uv/)
- **Command-line friendly**: All scripts support CLI usage with argparse
- **Format support**: Works with common microscopy formats (RAW, TIFF, OME-TIFF, CZI, ND2, etc.)

## Scripts

### iSCAT Analysis

| Script | Description | Usage |
|--------|-------------|-------|
| ```deconvolution.py``` | Richardson-Lucy deconvolution for 3D stacks | ```python deconvolution.py -i input.tif -o output.tif -psf psf.tif -n 10``` |
| ```background_subtraction.py``` | Rolling ball background subtraction | ```python background_subtraction.py -i input.tif -o output.tif -r 50``` |
| ```channel_alignment.py``` | Align multi-channel images using phase correlation | ```python channel_alignment.py -i input.tif -o aligned.tif -r 1``` |
| ```z_projection.py``` | Maximum/mean/median Z-projections | ```python z_projection.py -i stack.tif -o projection.tif -m max``` |

## Guidelines for New Scripts

- Each script should be self-contained in a single file
- Include a detailed docstring explaining purpose and usage
- Provide command-line arguments with sensible defaults
- Include error handling and validation
- Add progress indicators for long-running operations
- Output should be clearly documented
