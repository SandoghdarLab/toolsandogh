import os
import pathlib
import shutil
import subprocess
import tempfile
import urllib.parse

import numpy as np
import numpy.typing as npt
import xarray as xr
from bioio_imageio.writers import TimeseriesWriter
from bioio_ome_tiff.writers import OmeTiffWriter
from bioio_ome_zarr.writers import OMEZarrWriter

from ._canonicalize_video import canonicalize_video


def _check_ffmpeg_installed() -> None:
    """Ensure ffmpeg is installed and accessible."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not found in PATH. "
            "Please install ffmpeg to use video encoding features."
        )


def store_video(video: npt.ArrayLike, path: str | os.PathLike, **kwargs) -> None:
    """
    Store a given microscopy dataset at the supplied path.

    Parameters
    ----------
    video : npt.ArrayLike
        An array that is a suitable argument to :py:func:`~toolsandogh.canonicalize_video`.
    path : os.PathLike
        The name of a file, a URI, or a path.
    **kwargs : dict
        Extra arguments that are handled in a suffix-dependent way.
    """
    video = canonicalize_video(video)

    # Decode the path
    pathstr = str(path)

    # Handle Windows paths (e.g., C:\\...) correctly before URL parsing.
    # On Windows, a path like C:\\Users\\file.tiff might be parsed as scheme='C'.
    # We check if it looks like a valid absolute file path first.
    url_scheme = None
    if os.name == "nt" and len(pathstr) > 1 and pathstr[1] == ":":
        # Likely a Windows absolute path
        url_scheme = "file"
        path_obj = pathlib.Path(pathstr)
    else:
        url = urllib.parse.urlparse(pathstr)
        url_scheme = url.scheme or "file"
        path_obj = pathlib.Path(url.path)

    match (url_scheme, path_obj.suffix):
        case (scheme, ".bin" | ".raw"):
            store_raw_video(video, path_obj)
        case (scheme, ".mp4"):
            store_video_as_mp4(video, path_obj, **kwargs)
        case (scheme, ".avi"):
            data = video.stack(F=("T", "C", "Z")).transpose("F", "Y", "X").data
            TimeseriesWriter.save(data, pathstr, dimorder="TYX")
        case (scheme, ".tiff"):
            OmeTiffWriter.save(video.to_numpy(), pathstr)
        case (scheme, ".zarr"):
            writer = OMEZarrWriter(
                store=pathstr,
                level_shapes=video.shape,
                dtype=video.dtype,
                zarr_format=3,
            )
            writer.write_full_volume(video.data)
        case ("file", suffix):
            raise RuntimeError(f"Don't know how to store {suffix} data.")
        case (scheme, suffix):
            raise RuntimeError(f"Don't know how to store {suffix} data via {scheme}.")


def store_raw_video(video: xr.DataArray, path: str | os.PathLike) -> None:
    """
    Store a given microscopy dataset as a binary file.

    This method of storing discards all metadata, including the shape, so it is
    usually better to chose a different representation, e.g., as a .zarr file.

    Parameters
    ----------
    video : xarray.DataArray
        A canonical TCZYX array.
    path : os.PathLike
        The name of a file, a URI, or a path.
    """
    # We open the file in binary write mode and iterate over blocks.
    with open(path, "wb") as f:
        for block in video.data.blocks:
            # block is a view or array of the data.
            # We ensure it is contiguous before writing to avoid garbage bytes.
            np.ascontiguousarray(block).tofile(f)


def store_video_as_mp4(video: xr.DataArray, path: str | os.PathLike, fps: int = 30) -> None:
    """
    Store a given microscopy dataset as an MP4 file.

    This method stacks T, C, and Z dimensions into a single time dimension (F).
    It processes the data in chunks to avoid loading the entire video into memory.

    Parameters
    ----------
    video : xr.DataArray
        A canonical TCZYX array.
    path : os.PathLike
        The name of a file, a URI, or a path.
    fps : int
        The number of frames per second of the resulting mp4 video.
    """
    # Check for ffmpeg availability
    _check_ffmpeg_installed()

    # Stack T, C, and Z into a single Frame dimension
    data = video.stack(F=("T", "C", "Z")).transpose("F", "Y", "X")

    # Chunk the data to manage memory usage
    chunk_size = 1000
    data = data.chunk({"F": chunk_size})

    path_str = str(path)
    temp_dir = tempfile.TemporaryDirectory()
    chunk_files = []

    try:
        # Iterate over the chunks
        for i, chunk in enumerate(data.data.blocks):
            # Convert xarray block to numpy array
            chunk_np = np.array(chunk)

            # Define a temporary file path for this chunk
            chunk_path = os.path.join(temp_dir.name, f"chunk_{i:04d}.mp4")

            # Check if chunk already exists (allows resuming)
            if not os.path.exists(chunk_path):
                # Encode the chunk to MP4
                chunk_to_mp4(chunk_np, chunk_path, fps)

            chunk_files.append(chunk_path)

        # Concatenate all chunk files into the final video
        if len(chunk_files) > 0:
            # Create a list of files for ffmpeg concat demuxer
            list_file_path = os.path.join(temp_dir.name, "file_list.txt")
            with open(list_file_path, "w") as f:
                for chunk_file in chunk_files:
                    # Ensure paths are absolute or correctly relative for the concat demuxer
                    f.write(f"file '{chunk_file}'\n")

            # Run ffmpeg to concatenate
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file_path,
                "-c",
                "copy",
                path_str,
            ]

            proc = subprocess.Popen(concat_cmd)
            proc.wait()

            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed to concatenate chunks into {path_str}")

    finally:
        # Clean up temporary files
        temp_dir.cleanup()


def chunk_to_mp4(chunk: np.ndarray, path: str | os.PathLike, fps: int = 30) -> None:
    """
    Encode a (time, y, x) float32 chunk as an MP4 video.

    The chunk is assumed to contain grayscale intensities in the range [0, 1].
    It is scaled to uint8 and written to ffmpeg via a raw-video pipe.

    Parameters
    ----------
    chunk : np.ndarray
        Array of shape ``(T, Y, X)`` with ``dtype=float32``.
    path : str or os.PathLike
        Destination file for the encoded chunk.
    fps : int, optional
        Frames per second for the output video (default: 30).
    """
    # Determine width and height
    _, height, width = chunk.shape

    # Prepare data for encoding
    # If data is float, assume it needs scaling to 0-255.
    # If data is integer, we assume it is already in a displayable range or clip it.
    if np.issubdtype(chunk.dtype, np.floating):
        # Scale [0, 1] to [0, 255]
        data_to_encode = (np.clip(chunk, 0, 1) * 255).astype(np.uint8)
    elif chunk.dtype == np.uint8:
        data_to_encode = chunk
    else:
        # For other integer types (e.g. uint16), we normalize by max value to fit in uint8
        # This is a simplification; ideally, user provides min/max.
        max_val = np.iinfo(chunk.dtype).max
        data_to_encode = (chunk.astype(np.float32) / max_val * 255).astype(np.uint8)

    # Use gray8 pixel format since we are converting to uint8
    pix_fmt = "gray8"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",  # Input from stdin
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]

    # Use communicate to handle stdin/stderr automatically and avoid BrokenPipeError
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(input=data_to_encode.tobytes())

    if proc.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for chunk {path}:\n{error_msg}")
