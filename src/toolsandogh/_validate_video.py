import dask.array as da
import numpy as np
import ome_types.model as ome
import xarray as xr


def validate_video(video) -> None:
    """
    Ensure the supplied video is an xarray with axes T, C, Z, Y, and X.

    More precisely, raise an exception unless the supplied video adheres to the
    following rules:

    1. It is of type xarray.DataArray.

    2. Its data is represented as dask.array.Array.

    3. Its dims are ('T', 'C', 'Z', 'Y', 'X').

    4. Any T, Z, Y, or X coordinate has the element type np.float64.

    5. The video has metadata attached that is an ome_types.models.OME object.

    6. The attached metadata is consistent with the video itself.

    Parameters
    ----------
    video : Any
        The video to be validated.
    """
    if not isinstance(video, xr.DataArray):
        raise TypeError(f"Not an xarray.DataArray: {video}")

    if not isinstance(video.data, da.Array):
        raise TypeError(f"Expected dask.Array data, got {type(video.data)}.")

    # Ensure the video has the right dimensions in the right order.
    dims = ("T", "C", "Z", "Y", "X")
    if not video.dims == dims:
        raise ValueError(f"Dimension mismatch.  Expected {dims}, got {video.dims}.")

    # Ensure temporal and spatial coordinates are continuous.
    for dim in ("T", "Z", "Y", "X"):
        coord = video[dim]
        if coord.dtype != np.float64:
            raise TypeError(f"The {dim} coordinate is not continuous.")

    # Ensure metadata exists and is of the right format.
    metadata = video.attrs["processed"]
    if not isinstance(metadata, ome.OME):
        raise TypeError("Metadata is not of type ome_types.model.OME.")

    # Ensure the metadata is consistent.
    validate_video_metadata(video, metadata)


def validate_video_metadata(video: xr.DataArray, metadata: ome.OME):
    # Ensure metadata has exactly one image entry.
    if len(metadata.images) == 0:
        raise ValueError("Video has no image metadata.")
    if len(metadata.images) > 1:
        names = [image.name for image in metadata.images]
        raise ValueError(f"Video has metadata for multiple images: {names}")

    image: ome.Image = metadata.images[0]
    pixels: ome.Pixels = image.pixels
    expected_shape = (pixels.size_t, pixels.size_c, pixels.size_z, pixels.size_y, pixels.size_x)
    if video.shape != expected_shape:
        raise ValueError(f"Video has shape {video.shape}, but metadata shape {expected_shape}.")
