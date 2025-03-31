#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "imgui[sdl2]",
#     "fastplotlib[imgui] @ git+https://github.com/fastplotlib/fastplotlib.git@a057faa#egg=fastplotlib[imgui]",
#     "imageio[ffmpeg]",
#     "numpy",
#     "scipy",
#     "scikit-image",
#     "trackpy",
#     "imgrvt",
#     "tqdm",
# ]
# ///

import argparse
import os
import fastplotlib as fpl
import imgrvt as rvt
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import tqdm
import math
import trackpy
import pandas
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui

COLORMAPS = ["gray", "viridis", "plasma", "inferno", "magma", "cividis", "gnuplot2"]

###############################################################################
###
###  Analysis


class Analysis:
    """All the data associated with one iSCAT Analysis run.

    Attributes:
        args (argparse.Namespace): The parsed command line arguments
        video (npt.NDArray[np.float32]): The original video
        fft (npt.NDArray[np.complex64]): FFT of the original video
        fft_log_abs (npt.NDArray[np.float32]): log(abs(fft + 1))
        corrected (npt.NDArray[np.float32]): The inverse FFT of fft_part
        dra (npt.NDArray[np.float32]): Differential rolling average of corrected
        rvt (npt.NDArray[np.float32]): Radial variance transform of dra
        bits (npt.NDArray[np.bool_]): One bit per dra_window_size frames
        worklist (list[slice]): A list of dra slices that have yet to be computed
    """

    args: argparse.Namespace
    video: npt.NDArray[np.float32]
    fft: npt.NDArray[np.complex64]
    fft_log_abs: npt.NDArray[np.float32]
    corrected: npt.NDArray[np.float32]
    dra: npt.NDArray[np.float32]
    rvt: npt.NDArray[np.float32]
    locs: list[pandas.DataFrame]
    worklist: list[slice]

    def __init__(self, args: argparse.Namespace, video: npt.NDArray[np.float32]):
        assert video.ndim == 3
        self.args = args
        # initialize all buffers
        self.video = video.astype(np.float32)
        self.fft = np.zeros(video.shape, dtype=np.complex64)
        self.fft_log_abs = np.zeros(video.shape, dtype=np.float32)
        self.corrected = np.zeros(video.shape, dtype=np.float32)
        self.dra = np.zeros(video.shape, dtype=np.float32)
        self.rvt = np.zeros(video.shape, dtype=np.float32)
        self.locs = [pandas.DataFrame()] * video.shape[0]
        # initialize the worklist
        self.worklist = []
        self.reset(clear_buffers=False)

    def sort_worklist(self, current_frame=0):
        # Reorder the worklist
        def slice_distance(s: slice) -> int:
            if s.start <= current_frame < s.stop:
                return 0
            else:
                return min(abs(s.start - current_frame),
                           abs(s.stop - 1 - current_frame))

        self.worklist = sorted(self.worklist, key=slice_distance, reverse=True)

    def reset(self, clear_buffers=True):
        """Clear all buffers and reinitialize the worklist."""
        # Clear all buffers
        if clear_buffers:
            self.fft[...] = 0
            self.fft_log_abs[...] = 0
            self.corrected[...] = 0
            self.dra[...] = 0
            self.rvt[...] = 0
            for index in range(len(self.locs)):
                self.locs[index] = pandas.DataFrame()
        # Reinitialize the worklist
        self.worklist.clear()
        window_size = self.args.dra_window_size
        nframes = self.video.shape[0] - 2 * window_size + 1
        nchunks = (nframes // window_size)
        chunk_size = math.ceil(nframes / nchunks)
        for start in range(0, nframes - chunk_size, chunk_size):
            if (nframes - start) < (2 * chunk_size):
                end = nframes
            else:
                end = start + chunk_size
            self.worklist.append(slice(start, end, 1))
        self.sort_worklist()

    def advance(self):
        """(Re)compute parts of the Analysis run.  Prefer those close to the current_frame."""
        if len(self.worklist) == 0:
            return
        window_size = self.args.dra_window_size
        dra_frames = self.worklist.pop()
        # print(f"Computing DRA/RVT frames from {dra_frames.start} to {dra_frames.stop - 1}.")
        frames = slice(dra_frames.start, dra_frames.stop + 2 * window_size - 1)
        (nframes, nrows, ncols) = self.video[frames].shape

        # Determine the FFT mask
        row, col = np.mgrid[:nrows, :ncols]
        distance = np.sqrt((row - nrows / 2) ** 2 + (col - ncols / 2) ** 2)
        dmax = np.max(distance)
        inner = (self.args.fft_inner_radius * dmax) <= distance
        outer = distance <= (self.args.fft_outer_radius * dmax)
        ring = np.logical_and(inner, outer)
        cross = np.zeros((nrows, ncols), dtype=np.bool_)
        row_offset = self.args.fft_column_noise_threshold / 2
        row_start = round(nrows * (0.5 - row_offset))
        row_end = round(nrows * (0.5 + row_offset))
        cross[row_start:row_end, :] = True
        col_offset = self.args.fft_row_noise_threshold / 2
        col_start = round(ncols * (0.5 - col_offset))
        col_end = round(ncols * (0.5 + col_offset))
        cross[:, col_start:col_end] = True
        mask = np.broadcast_to(np.logical_and(ring, ~cross), (nframes, nrows, ncols))

        # Compute the FFT
        raw_fft = np.fft.fft2(self.video[frames], axes=(1, 2))
        shifted_fft = np.fft.fftshift(raw_fft, axes=(1, 2))
        filtered_fft = np.where(mask, shifted_fft, 0.0)
        ifft = np.fft.ifft2(np.fft.fftshift(filtered_fft, axes=(1,2)), axes=(1,2))
        corrected = np.real(ifft)
        self.fft[frames] = filtered_fft
        self.fft_log_abs[frames] = np.log(np.abs(filtered_fft + 1))
        self.corrected[frames] = corrected

        # Compute the rolling average
        padded = np.pad(corrected, ((1, 0), (0, 0), (0, 0)),)
        cumsum = np.cumsum(padded, axis=0)
        ravg = cumsum[window_size:] - cumsum[0:-window_size]

        # Compute the differential rolling average
        left = ravg[0:-window_size]
        right = ravg[window_size:]
        self.dra[dra_frames] = (left - right)

        # Compute the RVT
        for frame in range(dra_frames.start, dra_frames.stop):
            self.rvt[frame] = rvt.rvt(self.dra[frame],
                                      rmin=self.args.rvt_min_radius,
                                      rmax=self.args.rvt_max_radius,
                                      upsample=self.args.rvt_upsample,
                                      kind="normalized")

        # Track Particles
        for frame in range(dra_frames.start, dra_frames.stop):
            self.locs[frame] = trackpy.locate(self.rvt[frame],
                                              diameter=2*self.args.tracking_radius+1,
                                              minmass=self.args.tracking_min_mass,
                                              percentile=self.args.tracking_percentile,
                                              preprocess=False,
                                              characterize=False)


    def finish(self):
        """Complete the Analysis run."""
        print("Performing iSCAT Analysis with the following parameters:")
        self.print_args()
        if len(self.worklist) > 0:
            for _ in tqdm.tqdm(range(len(self.worklist)), unit="chunks"):
                self.advance()
        assert len(self.worklist) == 0

    def print_args(self):
        print(f"{os.path.basename(__file__)}", end="")
        for key, value in vars(self.args).items():
            print(f" --{key}={value}", end="")
        print("")


###############################################################################
###
### The iSCAT GUI


class SideBar(EdgeWindow):
    analysis: Analysis
    rvt_changes: bool
    loc_changes: bool

    def __init__(self, figure, size, location, title, analysis):
        super().__init__(figure=figure, size=size, location=location, title=title)
        self.rvt_changes = False
        self.loc_changes = True
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
        _, gui_sync = imgui.checkbox("Apply Settings Immediately", args.gui_sync)
        imgui.separator()

        imgui.text("FFT Parameters")
        _, fft_i_r = imgui.slider_float("fft-inner-radius", v=args.fft_inner_radius, v_min=0.0, v_max=0.25)
        _, fft_o_r = imgui.slider_float("fft-outer-radius", v=args.fft_outer_radius, v_min=0.0, v_max=1.0)
        _, fft_rnt = imgui.slider_float("fft-row-noise-threshold", v=args.fft_row_noise_threshold, v_min=0.0, v_max=0.125)
        _, fft_cnt = imgui.slider_float("fft-column-noise-threshold", v=args.fft_column_noise_threshold, v_min=0.0, v_max=0.125)
        imgui.separator()

        imgui.text("DRA Parameters")
        _, dra_w_s = imgui.slider_int("dra-window-size", v=args.dra_window_size, v_min=0, v_max=min(500, args.frames // 3))
        imgui.separator()

        imgui.text("RVT Parameters")
        _, rvt_mnr = imgui.slider_int("rvt-min-radius", v=args.rvt_min_radius, v_min=0, v_max=20)
        _, rvt_mxr = imgui.slider_int("rvt-max-radius", v=args.rvt_max_radius, v_min=0, v_max=50)
        _, rvt_ups = imgui.slider_int("rvt-upsample", v=args.rvt_upsample, v_min=0, v_max=4)
        imgui.separator()

        imgui.text("Tracking Parameters")
        _, tracking_radius = imgui.slider_int("tracking-radius", v=args.tracking_radius, v_min=0, v_max=20)
        _, tracking_min_mass = imgui.slider_float("tracking-min-mass", v=args.tracking_min_mass, v_min=0.0, v_max=400.0)
        _, tracking_percentile = imgui.slider_float("tracking-percentile", v=args.tracking_percentile, v_min=0.0, v_max=100.0)
        imgui.separator()

        # Changes
        if gui_sync != args.gui_sync:
            args.gui_sync = gui_sync
        if fft_i_r != args.fft_inner_radius:
            args.fft_inner_radius = fft_i_r
            self.rvt_changes = True
        if fft_o_r != args.fft_outer_radius:
            args.fft_outer_radius = fft_o_r
            self.rvt_changes = True
        if fft_rnt != args.fft_row_noise_threshold:
            args.fft_row_noise_threshold = fft_rnt
            self.rvt_changes = True
        if fft_cnt != args.fft_column_noise_threshold:
            args.fft_column_noise_threshold = fft_cnt
            self.rvt_changes = True
        if dra_w_s != args.dra_window_size:
            args.dra_window_size = dra_w_s
            self.rvt_changes = True
        if rvt_mnr != args.rvt_min_radius:
            args.rvt_min_radius = rvt_mnr
            self.rvt_changes = True
        if rvt_mxr != args.rvt_max_radius:
            args.rvt_max_radius = rvt_mxr
            self.rvt_changes = True
        if rvt_ups != args.rvt_upsample:
            args.rvt_upsample = rvt_ups
            self.rvt_changes = True
        if tracking_radius != args.tracking_radius:
            args.tracking_radius = tracking_radius
            self.loc_changes = True
        if tracking_min_mass != args.tracking_min_mass:
            args.tracking_min_mass = tracking_min_mass
            self.loc_changes = True
        if tracking_percentile != args.tracking_percentile:
            args.tracking_percentile = tracking_percentile
            self.loc_changes = True
        if self.rvt_changes:
            self.loc_changes = True
        if args.gui_sync:
            if self.rvt_changes:
                self.analysis.reset()
                self.rvt_changes = False

def iscat_gui(analysis: Analysis):
    iw = fpl.widgets.image_widget._widget.ImageWidget(
        data=[analysis.video, analysis.fft_log_abs, analysis.corrected, analysis.dra, analysis.rvt, analysis.rvt],
        names=["original", "fft", "corrected", "dra", "rvt", "rvt"],
        cmap="plasma",
        figure_kwargs={"size": (analysis.args.gui_width, analysis.args.gui_height),
                       "controller_ids": None})
    # Hide the axes
    for subplot in iw.figure:
        subplot.axes.visible = False

    sidebar_width = min(0.3*analysis.args.gui_width, 400)
    sidebar = SideBar(iw.figure, sidebar_width, "right", "Parameters", analysis)
    iw.figure.add_gui(sidebar)
    frame = 0
    lc1 = iw.figure[1,0].add_line_collection([], thickness=2, colors="yellow")
    lc2 = iw.figure[1,1].add_line_collection([], thickness=2, colors="yellow")

    def index_changed(index):
        nonlocal frame
        new_frame = index["t"]
        if frame == new_frame:
            return
        frame = new_frame
        analysis.sort_worklist(frame)
        analysis.advance()
        redraw_circles()

    iw.add_event_handler(index_changed, event="current_index")

    def animation():
        analysis.advance()
        iw.cmap = analysis.args.colormap
        # Update the ImageWidget
        iw.current_index = iw.current_index
        if sidebar.loc_changes:
            sidebar.loc_changes = False
            redraw_circles()

    def redraw_circles():
        circles = list()
        for _, row in analysis.locs[frame].iterrows():
            x, y = float(row["x"]), float(row["y"])
            circles.append(make_circle(x, y, analysis.args.tracking_radius))
        nonlocal lc1, lc2
        iw.figure[1,0].delete_graphic(lc1)
        iw.figure[1,1].delete_graphic(lc2)
        lc1 = iw.figure[1,0].add_line_collection(circles, thickness=2, colors="yellow")
        lc2 = iw.figure[1,1].add_line_collection(circles, thickness=2, colors="yellow")

    iw.figure.add_animations(animation)
    iw.show()


def make_circle(x: float, y: float, radius: float, n_points: int = 20) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n_points)
    xs = radius * np.sin(theta)
    ys = radius * np.cos(theta)

    return np.column_stack([xs, ys]) + np.array([x, y])


###############################################################################
###
###  Parameter Handling


def main():
    parser = argparse.ArgumentParser(description="Analyze iSCAT recordings.")

    parser.add_argument("-i", "--input-file", type=str, required=True,
                        help="Input file path")

    parser.add_argument("-o", "--output-file", type=str, required=True,
                        help="Output file path")

    parser.add_argument("--dtype", type=str, default="",
                        help="The dtype of the RAW file to load, or an empty string.")

    parser.add_argument("--rows", type=int, default=-1,
                        help="The number of rows of the raw video to load.")

    parser.add_argument("--columns", type=int, default=-1,
                        help="The number of columns of the raw video to load.")

    parser.add_argument("--frames", type=int, default=-1,
                        help="The number of frames of the input video to load.")

    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to rescale the input video to the [0, 1] interval.")

    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to open a GUI to preview the parameters (default: True).")

    parser.add_argument("--gui-width", type=int, default=1920,
                        help="The width of the GUI window in Pixels.")

    parser.add_argument("--gui-height", type=int, default=1080,
                        help="The height of the GUI window in Pixels.")

    parser.add_argument("--gui-sync", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to keep the GUI in sync with the chosen parameters (default: True).")

    parser.add_argument("--colormap", type=str, default=COLORMAPS[0],
                        help="The colormap used for visualization in the GUI.")

    parser.add_argument("--fft-inner-radius", type=float, default=0.0,
                        help="The inner circle to cut out in FFT space to filter high-frequencies.")

    parser.add_argument("--fft-outer-radius", type=float, default=1.0,
                        help="The outer circle to cut out in FFT space to filter low-frequencies.")

    parser.add_argument("--fft-row-noise-threshold", type=float, default=0.00,
                        help="The percentage of row noise frequencies to cut out in FFT space.")

    parser.add_argument("--fft-column-noise-threshold", type=float, default=0.00,
                        help="The percentage of column noise frequencies to cut out in FFT space.")

    parser.add_argument("--dra-window-size", type=int, default=20,
                        help="The number of frames to compute the differential rolling average over.")

    parser.add_argument("--rvt-min-radius", type=int, default=1,
                        help="The minimum radius (in pixels) to consider for radial variance transform.")

    parser.add_argument("--rvt-max-radius", type=int, default=20,
                        help="The maximum radius (in pixels) to consider for radial variance transform.")

    parser.add_argument("--rvt-upsample", type=int, default=1,
                        help="The degree of upsampling during radial variance transform.")

    parser.add_argument("--tracking-radius", type=int, default=4,
                        help="The radius (in pixels) of structures to locate in the RVT image.")

    parser.add_argument("--tracking-min-mass", type=float, default=0.0,
                        help="The minimum mass of structures to locate in the RVT image.")

    parser.add_argument("--tracking-percentile", type=float, default=90.0,
                        help="Features must be brighter than this percentile to be considered for tracking.")

    args = parser.parse_args()

    # Validate that input file exists
    if not os.path.isfile(args.input_file):
        parser.error(f"Input file '{args.input_file}' does not exist.")

    # Validate that output file doesn't exist
    if os.path.exists(args.output_file):
        parser.error(f"Output file '{args.output_file}' already exists.")

    # Validate the video parameters
    if args.dtype:
        dtype = np.dtype(args.dtype)
        if not args.rows > 0:
            parser.error(f"The --rows argument must be given when reading raw files.")
        if not args.columns > 0:
            parser.error(f"The --columns argument must be given when reading raw files.")
    else:
        if args.rows != -1:
            parser.error(f"The --rows argument must only be given when reading raw files.")
        if args.columns != -1:
            parser.error(f"The --columns argument must only be given when reading raw files.")

    # Validate the GUI parameters
    if args.gui_width < 1:
        parser.error(f"Videos must have at least one row, got {args.gui_width}.")
    if args.gui_height < 1:
        parser.error(f"Videos must have at least one row, got {args.gui_height}.")

    # Validate the FFT parameters
    if not 0 <= args.fft_inner_radius <= 1:
        parser.error(f"The --fft-inner-radius must be between 0 and 1, got {args.fft_inner_radius}")
    if not 0 <= args.fft_outer_radius <= 1:
        parser.error(f"The --fft-outer-radius must be between 0 and 1, got {args.fft_outer_radius}")
    if not 0 <= args.fft_row_noise_threshold <= 1:
        parser.error(f"The --fft-row-noise-threshold must be between 0 and 1, got {args.fft_row_noise_threshold}")
    if not 0 <= args.fft_column_noise_threshold <= 1:
        parser.error(f"The --fft-column-noise-threshold must be between 0 and 1, got {args.fft_column_noise_threshold}")

    # Validate the DRA parameters
    if args.dra_window_size < 1:
        parser.error(f"The --dra-window-size must be at least one, got {args.dra_window_size}")

    # Validate the RVT parameters
    if args.rvt_min_radius < 0:
        parser.error(f"The --rvt-min-radius must be non-negative, got {args.rvt_min_radius}")
    if not args.rvt_min_radius < args.rvt_max_radius:
        parser.error(f"The --rvt-max-radius must be larger than {args.rvt_min_radius}, got {args.rvt_max_radius}")
    if not args.rvt_upsample > 0:
        parser.error(f"The --rvt-upsample must be non-negative, got {args.rvt_upsample}")

    if args.dtype:
        dtype = np.dtype(args.dtype)
        # Derive the number of frames when it is not set.
        if args.frames == -1:
            bytes_per_frame = dtype.itemsize * args.rows * args.columns
            args.frames = os.path.getsize(args.input_file) // bytes_per_frame

        # Load the raw video
        shape = (args.frames, args.rows, args.columns)
        count = shape[0] * shape[1] * shape[2]
        pixels = np.fromfile(args.input_file, dtype=dtype, count=count)
        video = np.reshape(pixels, shape)
    else:
        # Read video using imageio
        data = iio.imread(args.input_file)
        if not 0 < data.ndim <= 3:
            raise ValueError(f"Unexpected input video shape: {data.shape}")
        video = np.reshape(data, ( (1,) * (3 - data.ndim) + data.shape))
    # Normalize the video if desired.
    if args.normalize:
        minimum = np.min(video)
        maximum = np.max(video)
        video = (video - minimum) / (maximum - minimum)
    # Start the iSCAT analysis
    analysis = Analysis(args, video)
    # Unless we have --no-gui, open a GUI for tuning the args
    if args.gui:
        # Compute the first batch of items before creating the GUI, so that we
        # have sane bounds for each histogram.
        analysis.advance()
        iscat_gui(analysis)
        fpl.loop.run()
    # Finish the analysis and save the results.
    analysis.finish()


if __name__ == "__main__":
    main()
