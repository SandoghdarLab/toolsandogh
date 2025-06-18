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

| Script                  | Description    | Usage                                                |
|-------------------------|----------------|------------------------------------------------------|
| ```iscat_analysis.py``` | iSCAT Analysis | ```./iscat_analysis.py -i input.tif -o output.tif``` |

## Troubleshooting

### "smudge filter lfs failed" on Windows when using uv

Solution: Set the following environment variable in Powershell

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
```

## Guidelines for New Scripts

- Each script should be self-contained in a single file
- Include a detailed docstring explaining purpose and usage
- Provide command-line arguments with sensible defaults
- Include error handling and validation
- Add progress indicators for long-running operations
- Output should be clearly documented
