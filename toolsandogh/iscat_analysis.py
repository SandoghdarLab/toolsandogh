#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">3.12,<3.13"
# dependencies = [
#     "imgui-bundle<1.92.0",
#     "fastplotlib[imgui]",
#     "numpy",
#     "scipy",
#     "scikit-image",
#     "trackpy",
#     "imgrvt",
#     "tqdm",
#     "imageio[ffmpeg]",
#     "bioio",
#     "bioio-bioformats",
#     "xarray",
#     "dask",
# ]
# ///

import argparse
import itertools
import math
import multiprocessing
import multiprocessing.pool
import os
import time
import warnings
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Generic, Literal, TypeVar

import bioio
import bioio.writers
import bioio_bioformats
import dask.array as da
import imageio
import imgrvt
import numpy as np
import numpy.typing as npt
import trackpy
import xarray as xr
from imgui_bundle import imgui, portable_file_dialogs

if __name__ == "__main__":
    from fastplotlib.ui import EdgeWindow  # type: ignore
else:
    # If we are not in the GUI thread, define a dummy EdgeWindow
    class EdgeWindow:
        def __init__(self, figure, size, location, title):
            pass


_Dtype = TypeVar("_Dtype", bound=np.generic, covariant=True)


###############################################################################
###
###  Utilities


class SharedArray(Generic[_Dtype]):
    """
    A Numpy array wrapper to make it inter-process shareable via pickling.
    """

    _shape: tuple[int, ...]
    _dtype: np.dtype
    _shm: SharedMemory
    _array: npt.NDArray[_Dtype]

    def __init__(self, shape: tuple[int, ...], dtype: npt.DTypeLike):
        self._shape = shape
        self._dtype = np.dtype(dtype)
        size = math.prod(shape) * self._dtype.itemsize
        self._shm = SharedMemory(create=True, size=size)
        self._array = np.ndarray(shape, dtype, buffer=self._shm.buf)

    def __array__(self, dtype=None, copy=None):
        array = self._array
        if dtype:
            array = array.astype(dtype)
        if copy:
            array = array.copy()
        return array

    def __getitem__(self, key):
        return self._array.__getitem__(key)

    def __setitem__(self, key, value):
        return self._array.__setitem__(key, value)

    def __len__(self):
        return self._array.__len__()

    def __str__(self):
        return self._array.__str__()

    def __getstate__(self):
        return {
            "_shape": self._shape,
            "_dtype": self._dtype.name,
            "_shm": self._shm,
        }

    def __setstate__(self, state):
        self._shape = state["_shape"]
        self._dtype = np.dtype(state["_dtype"])
        self._shm = state["_shm"]
        self._array = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)

    @property
    def shape(self):
        return self._shape


###############################################################################
###
###  Analysis


@dataclass
class Task:
    """Base class for schedulable work items."""

    dependencies: list["Task"]
    status: Literal["unscheduled", "scheduled", "finished"]
    start: int
    end: int

    def is_unscheduled(self):
        return self.status == "unscheduled"

    def is_scheduled(self):
        return self.status == "scheduled"

    def is_finished(self):
        return self.status == "finished"

    def run(self):
        pass

    def schedule(self, pool: multiprocessing.pool.Pool):
        assert self.status == "unscheduled"
        self.status = "scheduled"

        def callback(_):
            assert self.status == "scheduled"
            self.status = "finished"

        def error_callback(exc):
            print(f"Caught a {type(exc)}!")
            raise exc

        pool.apply_async(
            run_task, [self], callback=callback, error_callback=error_callback
        )


def run_task(task: Task):
    task.run()
    return task


def filter_tasks(tasks: list[Task], start: int, end: int) -> list[Task]:
    return [task for task in tasks if not ((end < task.start) or (task.end < start))]


@dataclass
class CopyFrames(Task):
    src: SharedArray[np.float32]
    dst: SharedArray[np.float32]

    def run(self):
        self.dst[self.start : self.end] = self.src[self.start : self.end]


@dataclass
class ComputeFFT(Task):
    """Perform Fourier space corrections for the selected range of frames."""

    video: SharedArray[np.float32]
    fft: SharedArray[np.complex64]
    fft_log_abs: SharedArray[np.float32]
    corrected: SharedArray[np.float32]
    inner_radius: float
    outer_radius: float
    column_noise_threshold: float
    row_noise_threshold: float

    def run(self):
        # Compute the FFT mask
        nframes = self.end - self.start
        (_, nrows, ncols) = self.video.shape
        row, col = np.mgrid[:nrows, :ncols]
        distance = np.sqrt((row - nrows / 2) ** 2 + (col - ncols / 2) ** 2)
        dmax = np.max(distance)
        inner = (self.inner_radius * dmax) <= distance
        outer = distance <= (self.outer_radius * dmax)
        ring = np.logical_and(inner, outer)
        cross = np.zeros((nrows, ncols), dtype=np.bool_)
        row_offset = self.column_noise_threshold / 2
        row_start = round(nrows * (0.5 - row_offset))
        row_end = round(nrows * (0.5 + row_offset))
        cross[row_start:row_end, :] = True
        col_offset = self.row_noise_threshold / 2
        col_start = round(ncols * (0.5 - col_offset))
        col_end = round(ncols * (0.5 + col_offset))
        cross[:, col_start:col_end] = True
        pattern = np.logical_and(ring, ~cross)
        # Ensure the pattern is symmetric
        pattern = np.logical_and(pattern, np.flip(pattern, axis=0))
        pattern = np.logical_and(pattern, np.flip(pattern, axis=1))
        mask = np.broadcast_to(pattern, (nframes, nrows, ncols))

        # Compute the FFT
        raw_fft = np.fft.fft2(self.video[self.start : self.end], axes=(1, 2))
        shifted_fft = np.fft.fftshift(raw_fft, axes=(1, 2))
        filtered_fft = np.where(mask, shifted_fft, 0.0)
        ifft = np.fft.ifft2(np.fft.fftshift(filtered_fft, axes=(1, 2)), axes=(1, 2))
        self.corrected[self.start : self.end] = np.real(ifft)
        self.fft[self.start : self.end] = filtered_fft
        self.fft_log_abs[self.start : self.end] = np.log(np.abs(filtered_fft + 1))


@dataclass
class ComputeDRA(Task):
    """Compute the differential rolling average."""

    video: SharedArray[np.float32]
    dra: SharedArray[np.float32]
    window_size: int

    def run(self):
        # Compute the rolling average
        ndra = self.end - self.start
        vid_end = self.end + (2 * self.window_size) - 1
        padded = np.pad(
            self.video[self.start : vid_end],
            ((1, 0), (0, 0), (0, 0)),
        )
        cumsum = np.cumsum(padded, axis=0)
        ravg = cumsum[self.window_size :] - cumsum[0 : -self.window_size]
        if len(ravg) != (ndra + self.window_size):
            msg = f"DRA Error: {self.start=} {self.end=} {vid_end=} {len(ravg)=}"
            raise RuntimeError(msg)
        # Compute the differential rolling average
        left = ravg[0 : -self.window_size]
        right = ravg[self.window_size :]
        self.dra[self.start : self.end] = left - right


@dataclass
class ComputeRVT(Task):
    """Compute the radial variance transform (RVT)."""

    video: SharedArray[np.float32]
    rvt: SharedArray[np.float32]
    min_radius: int
    max_radius: int
    upsample: int
    start: int
    end: int

    def run(self):
        for frame in range(self.start, self.end):
            self.rvt[frame] = imgrvt.rvt(
                self.video[frame],
                rmin=self.min_radius,
                rmax=self.max_radius,
                upsample=self.upsample,
                pad_mode="edge",
                kind="normalized",
            )


@dataclass
class ComputeLOC(Task):
    """Localize particles using trackpy."""

    video: SharedArray[np.float32]
    loc: SharedArray
    radius: int
    min_mass: float
    percentile: int
    maxlocs: int
    start: int
    end: int

    def run(self):
        self.loc[self.start : self.end] = 0
        for frame in range(self.start, self.end):
            locs = trackpy.locate(
                self.video[frame],
                diameter=2 * self.radius + 1,
                minmass=self.min_mass,
                percentile=self.percentile,
                topn=self.maxlocs,
                preprocess=False,
            )
            index = -1
            for x, y, mass, size, _, signal, _, _ in locs.itertuples(
                name=None, index=False
            ):
                index += 1
                self.loc[frame, index, 0] = np.float32(y)
                self.loc[frame, index, 1] = np.float32(x)
                self.loc[frame, index, 2] = np.float32(frame)
                self.loc[frame, index, 3] = np.float32(mass)
                self.loc[frame, index, 4] = np.float32(size)
                self.loc[frame, index, 5] = np.float32(signal)


class Analysis:
    """All the data associated with one iSCAT Analysis run.

    Attributes:
        args (argparse.Namespace): The parsed command line arguments
        video (SharedArray[np.float32]): The original video
        pool (multiprocessing.pool.Pool): The worker pool
        work (list[multiprocessing.pool.AsyncResult]): List of pending calculations
        current_frame (int): The frame being viewed right now
        fft (SharedArray[np.complex64]): FFT of the original video
        fft_log_abs (SharedArray[np.float32]): log(abs(fft + 1))
        corrected (SharedArray[np.float32]): The inverse FFT of fft_part
        dra (SharedArray[np.float32]): Differential rolling average of corrected
        rvt (SharedArray[np.float32]): Radial variance transform of dra
        loc (SharedArray[np.float32]): nframes x particles x (x, y, z, mass, size, signal)
        fft_tasks (list[ComputeFFT]): List of FFT tasks
        dra_tasks (list[ComputeDRA]): List of DRA tasks
        rvt_tasks (list[ComputeRVT]): List of RVT tasks
        loc_tasks (list[ComputeLOC]): List of LOC tasks
    """

    args: argparse.Namespace
    pool: multiprocessing.pool.Pool
    video: SharedArray[np.float32]
    fft: SharedArray[np.complex64]
    fft_log_abs: SharedArray[np.float32]
    corrected: SharedArray[np.float32]
    dra: SharedArray[np.float32]
    rvt: SharedArray[np.float32]
    loc: SharedArray[np.float32]
    fft_tasks: list[Task]
    dra_tasks: list[Task]
    rvt_tasks: list[Task]
    loc_tasks: list[Task]

    def __init__(self, args: argparse.Namespace, video: xr.DataArray):
        assert video.dims == ("T", "Y", "X")
        self.args = args
        self.pool = multiprocessing.Pool(processes=args.processes)
        self.work = []
        self.current_frame = 0
        # initialize all shared arrays
        self.video = SharedArray(video.shape, np.float32)
        self.video[...] = video.astype(np.float32)
        self.fft = SharedArray(video.shape, dtype=np.complex64)
        self.fft_log_abs = SharedArray(video.shape, dtype=np.float32)
        self.corrected = SharedArray(video.shape, dtype=np.float32)
        self.dra = SharedArray(video.shape, dtype=np.float32)
        (f, h, w) = video.shape
        u = self.args.rvt_upsample
        self.rvt = SharedArray((f, h * u, w * u), dtype=np.float32)
        nframes = video.shape[0]
        nparticles = args.particles
        self.loc = SharedArray((nframes, nparticles, 6), dtype=np.float32)
        # Initialize tasks
        self.fft_tasks = []
        self.dra_tasks = []
        self.rvt_tasks = []
        self.loc_tasks = []
        self.invalidate_fft()
        self.invalidate_dra()
        self.invalidate_rvt()
        self.invalidate_loc()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()
        self.video._shm.unlink()
        self.fft._shm.unlink()
        self.fft_log_abs._shm.unlink()
        self.corrected._shm.unlink()
        self.rvt._shm.unlink()
        self.dra._shm.unlink()
        self.loc._shm.unlink()
        self.pool.close()
        self.pool.terminate()

    def change_current_frame(self, frame: int):
        if self.current_frame != frame:
            self.current_frame = frame

    def invalidate_fft(self):
        chunk_size = 16
        self.fft_tasks = [
            ComputeFFT(
                dependencies=[],
                status="unscheduled",
                video=self.video,
                fft=self.fft,
                fft_log_abs=self.fft_log_abs,
                corrected=self.corrected,
                inner_radius=self.args.fft_inner_radius,
                outer_radius=self.args.fft_outer_radius,
                column_noise_threshold=self.args.fft_column_noise_threshold,
                row_noise_threshold=self.args.fft_row_noise_threshold,
                start=start,
                end=min(start + chunk_size, len(self.video)),
            )
            for start in range(0, len(self.video), chunk_size)
        ]

    def invalidate_dra(self):
        nframes = len(self.video)
        window_size_limit = nframes // 2
        window_size = self.args.dra_window_size
        if window_size > window_size_limit:
            warnings.warn(f"""Truncating DRA window size to {window_size_limit}.""")
            self.args.dra_window_size = window_size_limit
            window_size = window_size_limit
        ndra = nframes if window_size == 0 else nframes - (2 * window_size) + 1
        chunk_size = max(16, window_size)
        nchunks = math.floor(ndra / chunk_size)
        rest = ndra % chunk_size
        position = 0
        bounds: list[tuple[int, int]] = []
        for size in [chunk_size] * nchunks + [rest]:
            bounds.append((position, position + size))
            position += size
        if window_size == 0:
            self.dra_tasks = [
                CopyFrames(
                    dependencies=filter_tasks(self.fft_tasks, start, end),
                    status="unscheduled",
                    start=start,
                    end=end,
                    src=self.corrected,
                    dst=self.dra,
                )
                for start, end in bounds
            ]
        else:
            self.dra_tasks = [
                ComputeDRA(
                    dependencies=filter_tasks(
                        self.fft_tasks, start, end + (2 * window_size) - 1
                    ),
                    status="unscheduled",
                    start=start,
                    end=end,
                    video=self.corrected,
                    dra=self.dra,
                    window_size=self.args.dra_window_size,
                )
                for start, end in bounds
            ]

    def invalidate_rvt(self):
        nframes = len(self.video)
        window_size = self.args.dra_window_size
        nrvt = nframes if window_size == 0 else nframes - (2 * window_size) + 1
        chunk_size = 8
        bounds: list[tuple[int, int]] = []
        for start in range(0, nrvt, chunk_size):
            end = min(nrvt, start + chunk_size)
            bounds.append((start, end))
        self.rvt_tasks = [
            ComputeRVT(
                dependencies=filter_tasks(self.dra_tasks, start, start + chunk_size),
                status="unscheduled",
                start=start,
                end=end,
                video=self.dra,
                rvt=self.rvt,
                min_radius=self.args.rvt_min_radius,
                max_radius=self.args.rvt_max_radius,
                upsample=self.args.rvt_upsample,
            )
            for start, end in bounds
        ]

    def invalidate_loc(self):
        nframes = len(self.video)
        window_size = self.args.dra_window_size
        nloc = nframes if window_size == 0 else nframes - (2 * window_size) + 1
        chunk_size = 8
        bounds: list[tuple[int, int]] = []
        for start in range(0, nloc, chunk_size):
            end = min(nloc, start + chunk_size)
            bounds.append((start, end))
        self.loc_tasks = [
            ComputeLOC(
                dependencies=filter_tasks(self.rvt_tasks, start, end),
                status="unscheduled",
                start=start,
                end=end,
                video=self.rvt,
                loc=self.loc,
                radius=self.args.localization_radius,
                min_mass=self.args.localization_min_mass,
                percentile=self.args.localization_percentile,
                maxlocs=self.args.particles,
            )
            for start, end in bounds
        ]

    def all_tasks(self):
        def task_distance(task) -> int:
            return abs(self.current_frame - task.start)

        # Determine which tasks are ready
        a = sorted(self.loc_tasks, key=task_distance)
        b = sorted(self.rvt_tasks, key=task_distance)
        c = sorted(self.dra_tasks, key=task_distance)
        d = sorted(self.fft_tasks, key=task_distance)
        return itertools.chain(a, b, c, d)

    def available_tasks(self):
        """
        Returns a generator over tasks whose dependencies are satisfied.
        """

        def task_distance(task) -> int:
            return abs(self.current_frame - task.start)

        # Determine which tasks are ready
        for task in self.all_tasks():
            if task.is_unscheduled():
                if all(dep.is_finished() for dep in task.dependencies):
                    yield task

    def next_batch_of_tasks(self, maxlength: int):
        """
        Returns up to maxlength tasks whose dependencies are satisfied.
        """
        if maxlength == 0:
            return []
        it = itertools.batched(self.available_tasks(), maxlength)
        try:
            return next(it)
        except StopIteration:
            return []

    def update_schedule(self):
        # Ensure there are at least twice as many pending calculations as there
        # are workers in the pool.
        goal = 2 * self.args.processes
        scheduled = []
        completed = []
        for task in self.work:
            if task.is_finished():
                completed.append(task)
            else:
                scheduled.append(task)
        max_tasks = max(0, goal - len(scheduled))
        tasks = self.next_batch_of_tasks(max_tasks)
        for task in tasks:
            task.schedule(self.pool)
            scheduled.append(task)
        # Remove all the completed tasks
        self.work = scheduled

    def finish(self):
        """Complete the Analysis run."""
        # Print the parameters of the analysis run
        print("Performing iSCAT Analysis with the following parameters:")
        self.print_args()
        # Complete the work
        while not all(task.is_finished() for task in self.loc_tasks):
            self.update_schedule()
            time.sleep(0.01)

    def print_args(self):
        flags = []
        for key, value in vars(self.args).items():
            arg = key.replace("_", "-")
            if isinstance(value, bool):
                if value is True:
                    flags.append(f"--{arg}")
                else:
                    flags.append(f"--no-{arg}")
            elif isinstance(value, str):
                flags.append(f"--{arg}='{value}'")
            else:
                flags.append(f"--{arg}={value}")
        print(f"{os.path.basename(__file__)} {' '.join(flags)}")


###############################################################################
###
### The iSCAT GUI


COLORMAPS = ["magma", "gray", "viridis", "plasma", "inferno", "cividis", "gnuplot2"]

CIRCLE_POINTS = 16


class SideBar(EdgeWindow):
    analysis: Analysis

    def __init__(self, figure, size, location, title, analysis):
        super().__init__(figure=figure, size=size, location=location, title=title)
        self.analysis = analysis

    def update(self):
        args = self.analysis.args
        # Widgets
        imgui.text("GUI Parameters")
        if imgui.begin_combo("##colormap_combo", args.colormap):
            for cmap in COLORMAPS:
                is_selected = cmap == args.colormap
                if imgui.selectable(cmap, is_selected)[0]:
                    args.colormap = cmap
            imgui.end_combo()
        imgui.separator()

        imgui.text("FFT Parameters")
        _, fft_i_r = imgui.slider_float(
            "fft-inner-radius", v=args.fft_inner_radius, v_min=0.0, v_max=0.25
        )
        _, fft_o_r = imgui.slider_float(
            "fft-outer-radius", v=args.fft_outer_radius, v_min=0.0, v_max=1.0
        )
        _, fft_rnt = imgui.slider_float(
            "fft-row-noise-threshold",
            v=args.fft_row_noise_threshold,
            v_min=0.0,
            v_max=0.125,
        )
        _, fft_cnt = imgui.slider_float(
            "fft-column-noise-threshold",
            v=args.fft_column_noise_threshold,
            v_min=0.0,
            v_max=0.125,
        )
        imgui.separator()

        imgui.text("DRA Parameters")
        _, dra_w_s = imgui.slider_int(
            "dra-window-size",
            v=args.dra_window_size,
            v_min=0,
            v_max=min(1000, args.frames // 2),
        )
        imgui.separator()

        imgui.text("RVT Parameters")
        _, rvt_mnr = imgui.slider_int(
            "rvt-min-radius", v=args.rvt_min_radius, v_min=0, v_max=20
        )
        _, rvt_mxr = imgui.slider_int(
            "rvt-max-radius", v=args.rvt_max_radius, v_min=0, v_max=50
        )
        imgui.separator()

        imgui.text("Localization Parameters")
        _, localization_radius = imgui.slider_int(
            "localization-radius", v=args.localization_radius, v_min=0, v_max=20
        )
        _, localization_min_mass = imgui.slider_float(
            "localization-min-mass", v=args.localization_min_mass, v_min=0.0, v_max=5.0
        )
        _, localization_percentile = imgui.slider_int(
            "localization-percentile",
            v=args.localization_percentile,
            v_min=0,
            v_max=100,
        )
        _, args.circle_alpha = imgui.slider_float(
            "circle-alpha", v=args.circle_alpha, v_min=0.0, v_max=1.0
        )

        imgui.separator()

        imgui.text("Save Files")

        def truncate_path(path: str, maxlen=35):
            if len(path) > (maxlen - 3):
                return "..." + path[-maxlen:]
            return path

        pfd = portable_file_dialogs
        filters = [
            "TIFF files",
            "*.tif *.tiffAVI Files",
            "*.avi",
            "MP4 Files",
            "*.mp4",
            "Numpy files",
            "*.npy",
        ]
        if imgui.button(f"FFT File: {truncate_path(args.fft_file)}"):
            filename = pfd.save_file("fft-file", args.fft_file, filters).result()
            if filename:
                args.fft_file = filename
        if imgui.button(f"DRA File: {truncate_path(args.dra_file)}"):
            filename = pfd.save_file("dra-file", args.dra_file, filters).result()
            if filename:
                args.dra_file = filename
        if imgui.button(f"RVT File: {truncate_path(args.rvt_file)}"):
            filename = pfd.save_file("rvt-file", args.rvt_file, filters).result()
            if filename:
                args.rvt_file = filename
        if imgui.button(f"LOC File: {truncate_path(args.loc_file)}"):
            filename = pfd.save_file("loc-file", args.loc_file, filters).result()
            if filename:
                args.loc_file = filename
        imgui.separator()

        nunscheduled = 0
        nscheduled = 0
        nfinished = 0
        for task in self.analysis.all_tasks():
            if task.is_finished():
                nfinished += 1
            elif task.is_scheduled():
                nscheduled += 1
            else:
                nunscheduled += 1
        ntasks = nfinished + nscheduled + nunscheduled
        imgui.text(f"Status: {nfinished}/{ntasks} computed ({nscheduled} scheduled)")

        # Ensure RVT min and max radius are consistent with each other
        if rvt_mxr <= rvt_mnr:
            rvt_mxr = rvt_mnr + 1

        # Changes
        fft_changes: bool = False
        dra_changes: bool = False
        rvt_changes: bool = False
        loc_changes: bool = False
        if fft_i_r != args.fft_inner_radius:
            args.fft_inner_radius = fft_i_r
            fft_changes = True
        if fft_o_r != args.fft_outer_radius:
            args.fft_outer_radius = fft_o_r
            fft_changes = True
        if fft_rnt != args.fft_row_noise_threshold:
            args.fft_row_noise_threshold = fft_rnt
            fft_changes = True
        if fft_cnt != args.fft_column_noise_threshold:
            args.fft_column_noise_threshold = fft_cnt
            fft_changes = True
        if dra_w_s != args.dra_window_size:
            args.dra_window_size = dra_w_s
            dra_changes = True
        if rvt_mnr != args.rvt_min_radius:
            args.rvt_min_radius = rvt_mnr
            rvt_changes = True
        if rvt_mxr != args.rvt_max_radius:
            args.rvt_max_radius = rvt_mxr
            rvt_changes = True
        if localization_radius != args.localization_radius:
            args.localization_radius = localization_radius
            loc_changes = True
        if localization_min_mass != args.localization_min_mass:
            args.localization_min_mass = localization_min_mass
            loc_changes = True
        if localization_percentile != args.localization_percentile:
            args.localization_percentile = localization_percentile
            loc_changes = True
        if fft_changes:
            dra_changes = True
            rvt_changes = True
            loc_changes = True
        if dra_changes:
            rvt_changes = True
            loc_changes = True
        if fft_changes:
            self.analysis.invalidate_fft()
        if dra_changes:
            self.analysis.invalidate_dra()
        if rvt_changes:
            self.analysis.invalidate_rvt()
        if loc_changes:
            self.analysis.invalidate_loc()


def iscat_gui(analysis: Analysis):
    import fastplotlib as fpl

    iw = fpl.widgets.image_widget._widget.ImageWidget(
        data=[
            analysis.video._array,
            analysis.fft_log_abs._array,
            analysis.corrected._array,
            analysis.dra._array,
            analysis.rvt._array,
            analysis.rvt._array,
        ],
        names=["original", "fft", "corrected", "dra", "rvt", "rvtcopy"],
        cmap="plasma",
        figure_kwargs={
            "size": (analysis.args.gui_width, analysis.args.gui_height),
            "controller_ids": None,
        },
    )
    # Hide the axes
    for subplot in iw.figure:
        subplot.axes.visible = False

    sidebar_width = min(0.3 * analysis.args.gui_width, 390)
    sidebar = SideBar(iw.figure, sidebar_width, "right", "Parameters", analysis)
    iw.figure.add_gui(sidebar)  # type: ignore

    # Draw circles around all localized particles
    localizations = analysis.loc[analysis.current_frame]
    data = circle_data(localizations)
    color = np.array([1.0, 1.0, 0.0, analysis.args.circle_alpha])
    ls10 = iw.figure[1, 0].add_line_collection(data, colors=color)
    ls11 = iw.figure[1, 1].add_line_collection(data, colors=color)

    def index_changed(index):
        new_frame = index["t"]
        if analysis.current_frame == new_frame:
            return
        analysis.current_frame = new_frame

    iw.add_event_handler(index_changed, event="current_index")

    def animation():
        analysis.update_schedule()
        iw.cmap = analysis.args.colormap
        # Update the Circle Plot
        localizations = analysis.loc[analysis.current_frame]
        data = circle_data(localizations)
        color = f"#ffff00{math.floor(255 * analysis.args.circle_alpha):02x}"
        ls10.data = data
        ls11.data = data
        ls10.colors = color
        ls11.colors = color
        # Update the ImageWidget
        iw.current_index = iw.current_index

    iw.figure.add_animations(animation)
    iw.show()
    fpl.loop.run()


def circle_data(localizations):
    """
    Turn a nparticles x (x, y, z, mass, size, signal) array into a
    nparticles x CIRCLE_POINTS x 2 array of line segments."""
    # determine circle centers and radii
    xs = localizations[:, 0]
    ys = localizations[:, 1]
    rs = localizations[:, 4]
    # determine the X and Y offsets
    nparticles = len(localizations)
    theta = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
    oxs = rs.reshape((nparticles, 1)) * np.sin(theta)
    oys = rs.reshape((nparticles, 1)) * np.cos(theta)
    # determine and return the line segments
    lxs = xs.reshape((nparticles, 1)) + oxs
    lys = ys.reshape((nparticles, 1)) + oys
    lzs = np.zeros((nparticles, CIRCLE_POINTS))
    return np.stack([lxs, lys, lzs], axis=2)


###############################################################################
###
###  Parameter Handling


def main():
    """Parse command-line arguments and start iSCAT processing."""
    parser = argparse.ArgumentParser(description="Analyze iSCAT recordings.")

    parser.add_argument(
        "-i", "--input-file", type=str, required=True, help="Input file path"
    )

    parser.add_argument(
        "--initial-frame",
        type=int,
        default=0,
        help="The first frame of the raw video to load.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=-1,
        help="The number of frames of the input video to load.",
    )

    parser.add_argument("--fft-file", type=str, default="", help="FFT output file path")

    parser.add_argument("--dra-file", type=str, default="", help="DRA output file path")

    parser.add_argument("--rvt-file", type=str, default="", help="RVT output file path")

    parser.add_argument("--loc-file", type=str, default="", help="LOC output file path")

    parser.add_argument(
        "--raw-dtype",
        type=str,
        default="",
        help="The dtype of the RAW file to load, or an empty string.",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=-1,
        help="The number of rows of the raw video to load.",
    )

    parser.add_argument(
        "--columns",
        type=int,
        default=-1,
        help="The number of columns of the raw video to load.",
    )

    parser.add_argument(
        "--channel", type=int, default=0, help="What channel of the video to load."
    )

    parser.add_argument(
        "--zstack", type=int, default=0, help="What slice of the Z stack to load."
    )

    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to rescale the input video to the [0, 1] interval.",
    )

    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to open a GUI to preview the parameters (default: True).",
    )

    parser.add_argument(
        "--gui-width",
        type=int,
        default=1920,
        help="The width of the GUI window in Pixels.",
    )

    parser.add_argument(
        "--gui-height",
        type=int,
        default=1080,
        help="The height of the GUI window in Pixels.",
    )

    parser.add_argument(
        "--colormap",
        type=str,
        default=COLORMAPS[0],
        help="The colormap used for visualization in the GUI.",
    )

    parser.add_argument(
        "--fft-inner-radius",
        type=float,
        default=0.0,
        help="The inner circle to cut out in FFT space to filter high-frequencies.",
    )

    parser.add_argument(
        "--fft-outer-radius",
        type=float,
        default=1.0,
        help="The outer circle to cut out in FFT space to filter low-frequencies.",
    )

    parser.add_argument(
        "--fft-row-noise-threshold",
        type=float,
        default=0.00,
        help="The percentage of row noise frequencies to cut out in FFT space.",
    )

    parser.add_argument(
        "--fft-column-noise-threshold",
        type=float,
        default=0.00,
        help="The percentage of column noise frequencies to cut out in FFT space.",
    )

    parser.add_argument(
        "--dra-window-size",
        type=int,
        default=20,
        help="The number of frames to compute the differential rolling average over.",
    )

    parser.add_argument(
        "--rvt-min-radius",
        type=int,
        default=1,
        help="The minimum radius (in pixels) to consider for radial variance transform.",
    )

    parser.add_argument(
        "--rvt-max-radius",
        type=int,
        default=8,
        help="The maximum radius (in pixels) to consider for radial variance transform.",
    )

    parser.add_argument(
        "--rvt-upsample",
        type=int,
        default=1,
        help="The degree of upsampling during radial variance transform.",
    )

    parser.add_argument(
        "--localization-radius",
        type=int,
        default=4,
        help="The radius (in pixels) of structures to locate in the RVT image.",
    )

    parser.add_argument(
        "--localization-min-mass",
        type=float,
        default=0.0,
        help="The minimum mass of structures to locate in the RVT image.",
    )

    parser.add_argument(
        "--localization-percentile",
        type=int,
        default=80,
        help="Features must be brighter than this percentile to be considered for localization.",
    )

    parser.add_argument(
        "--particles",
        type=int,
        default=200,
        help="Track only the specified number of brightest particles in each frame.",
    )

    parser.add_argument(
        "--circle-alpha",
        type=float,
        default=1.0,
        help="Alpha channel of the circles drawn around each localized particle.",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="The number of background processes to use for computing.",
    )

    args = parser.parse_args()

    # Validate that input file exists
    if not os.path.isfile(args.input_file):
        parser.error(f"Input file '{args.input_file}' does not exist.")

    # Ensure that output files don't exist
    def check_output_file(output_file):
        if output_file:
            if os.path.exists(output_file):
                parser.error(f"Cannot write to existing file '{output_file}'.")

    check_output_file(args.fft_file)
    check_output_file(args.dra_file)
    check_output_file(args.rvt_file)
    check_output_file(args.loc_file)

    # Validate the video parameters
    if args.initial_frame < 0:
        parser.error("The --initial-frame argument must be non-negative.")
    if args.frames < -1:
        parser.error("The --frames argument must be non-negative or -1.")
    if args.raw_dtype:
        dtype = np.dtype(args.raw_dtype)
        if not args.rows > 0:
            parser.error("The --rows argument must be given when reading raw files.")
        if not args.columns > 0:
            parser.error("The --columns argument must be given when reading raw files.")
    else:
        if args.rows != -1:
            parser.error(
                "The --rows argument must only be given when reading raw files."
            )
        if args.columns != -1:
            parser.error(
                "The --columns argument must only be given when reading raw files."
            )

    # Validate the GUI parameters
    if args.gui_width < 1:
        parser.error(f"Videos must have at least one row, got {args.gui_width}.")
    if args.gui_height < 1:
        parser.error(f"Videos must have at least one row, got {args.gui_height}.")

    # Validate the FFT parameters
    if not 0 <= args.fft_inner_radius <= 1:
        parser.error(
            f"The --fft-inner-radius must be between 0 and 1, got {args.fft_inner_radius}"
        )
    if not 0 <= args.fft_outer_radius <= 1:
        parser.error(
            f"The --fft-outer-radius must be between 0 and 1, got {args.fft_outer_radius}"
        )
    if not 0 <= args.fft_row_noise_threshold <= 1:
        parser.error(
            f"The --fft-row-noise-threshold must be between 0 and 1, got {args.fft_row_noise_threshold}"
        )
    if not 0 <= args.fft_column_noise_threshold <= 1:
        parser.error(
            f"The --fft-column-noise-threshold must be between 0 and 1, got {args.fft_column_noise_threshold}"
        )

    # Validate the DRA parameters
    if args.dra_window_size < 0:
        parser.error(
            f"The --dra-window-size must be non-negative, got {args.dra_window_size}"
        )

    # Validate the RVT parameters
    if args.rvt_min_radius < 0:
        parser.error(
            f"The --rvt-min-radius must be non-negative, got {args.rvt_min_radius}"
        )
    if not args.rvt_min_radius < args.rvt_max_radius:
        parser.error(
            f"The --rvt-max-radius must be larger than {args.rvt_min_radius}, got {args.rvt_max_radius}"
        )
    if not args.rvt_upsample > 0:
        parser.error(
            f"The --rvt-upsample must be non-negative, got {args.rvt_upsample}"
        )

    # Validate the localization parameters
    if not (0 <= args.circle_alpha <= 1):
        parser.error("The --circle-alpha must be between zero and one.")

    # Load the input file, either from raw data or from a suitable image or
    # video file.
    video: xr.DataArray
    if args.raw_dtype:
        dtype = np.dtype(args.raw_dtype)
        # Derive the number of frames when it is not set
        if args.frames == -1:
            bytes_per_frame = dtype.itemsize * args.rows * args.columns
            args.frames = os.path.getsize(args.input_file) // bytes_per_frame

        # Load the raw video
        shape = (args.frames, args.rows, args.columns)
        bytes_per_frame = args.rows * args.columns * dtype.itemsize
        offset = args.initial_frame * bytes_per_frame
        mmap_array = np.memmap(
            args.input_file, dtype=dtype, mode="r", shape=shape, offset=offset
        )
        dask_array = da.from_array(mmap_array)
        video = xr.DataArray(dask_array, dims=("T", "Y", "X"))
    else:
        # Read video using bioio
        img = bioio.BioImage(args.input_file, reader=bioio_bioformats.Reader)
        T = slice(
            args.initial_frame,
            None if args.frames == -1 else args.initial_frame + args.frames,
            1,
        )
        xarr = img.get_xarray_stack()
        video = xarr.isel(I=0, T=T, C=args.channel, Z=args.zstack)
    # Check that the video matches the prescribed parameters
    assert video.dims == ("T", "Y", "X")
    if args.frames == -1:
        args.frames = video.shape[0]
    if args.rows == -1:
        args.rows = video.shape[1]
    if args.columns == -1:
        args.columns = video.shape[2]
    assert video.shape == (args.frames, args.rows, args.columns)
    # Normalize the video if desired.
    if args.normalize:
        minimum = np.min(video)
        maximum = np.max(video)
        video = (video - minimum) / (maximum - minimum)
    # Start the iSCAT analysis
    with Analysis(args, video) as analysis:
        # Unless we have --no-gui, open a GUI for tuning the args
        if args.gui:
            # Compute the first batch of items before creating the GUI, so that we
            # have sane bounds for each histogram.
            while not analysis.loc_tasks[0].is_finished():
                analysis.update_schedule()
                time.sleep(0.01)
            # Start the GUI
            iscat_gui(analysis)
        # Save the results

        def maybe_save(filename: str, array: npt.ArrayLike):
            if not filename:
                return
            imageio.mimwrite(filename, [np.array(array)])

        maybe_save(args.fft_file, analysis.corrected)
        maybe_save(args.dra_file, analysis.dra)
        maybe_save(args.rvt_file, analysis.rvt)
        # TODO loc file


if __name__ == "__main__":
    # Spawn new children rather then forking them off, to avoid that shared
    # arrays are pickled and then never freed.
    multiprocessing.set_start_method("spawn")
    # Run the actual main function.
    main()
