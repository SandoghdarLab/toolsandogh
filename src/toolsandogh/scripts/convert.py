#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "argparse",
#     "numpy",
#     "xarray",
#     "bioio",
#     "bioio-czi",
#     "bioio-dv",
#     "bioio-imageio",
#     "bioio-lif",
#     "bioio-nd2",
#     "bioio-ome-tiff",
#     "bioio-ome-zarr",
#     "bioio-sldy",
#     "bioio-tifffile",
#     "bioio-tiff-glob",
# ]
# ///
"""A simple script for converting between various microscopy file formats."""

import argparse

from bioio import BioImage


def parse_slice(slice_string: str) -> slice:
    """
    Parse a string representing a slice (e.g., "2:5", ":5", "2:") into a slice object.

    Parameters
    ----------
    slice_string : str
        A string representing a slice.  Can be in the format "start:stop:step". If start, stop, or step are omitted, they are treated as None.

    Returns
    -------
    slice
        A slice object.
    """
    parts = slice_string.split(":")
    try:
        start = int(parts[0]) if len(parts) > 0 and parts[0] else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    except ValueError as err:
        raise ValueError("Only numbers and ':' are allowed in slice string.") from err

    return slice(start, stop, step)


def main():
    """The main entry point for the convert.py script."""
    parser = argparse.ArgumentParser(description="Convert Microscopy Data")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file or URI")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file or URI")
    parser.add_argument("-T", type=str, help="Time Slice", default="")
    parser.add_argument("-C", type=str, help="Channel Slice", default="")
    parser.add_argument("-Z", type=str, help="Z Slice", default="")
    parser.add_argument("-Y", type=str, help="Y Slice", default="")
    parser.add_argument("-X", type=str, help="X Slice", default="")

    args = parser.parse_args()
    selection = {
        "T": parse_slice(args.T),
        "C": parse_slice(args.C),
        "Z": parse_slice(args.Z),
        "Y": parse_slice(args.Y),
        "X": parse_slice(args.X),
    }
    # Load
    print(f"Loading {args.input}...", end="")
    img = BioImage(args.input)
    print(" done.")

    # Print Image Information
    print(f"Dimensions: {img.dims}")
    print(f"Shape: {img.shape}")
    print(f"Pixel Sizes: {img.physical_pixel_sizes}")
    print(f"Time Interval: {img.time_interval}")
    print(f"Metadata: {img.metadata}")
    print(f"Selection: {selection}")

    # Store
    print(f"Storing to {args.output}...", end="")
    img.save(args.output)
    print(" done.")


if __name__ == "__main__":
    main()
