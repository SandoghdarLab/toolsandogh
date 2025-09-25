"""Tests input and output of videos."""

import pathlib

from toolsandogh import generate_video, load_video


def test_generate_video():
    """Unit test for :func:`toolsandogh.generate_video`."""
    generate_video()


def test_load_video():
    """Unit test for :func:`toolsandogh.load_video`."""
    parent = pathlib.Path(__file__).resolve().parent
    testfile = str(parent / "testfile.tiff")
    load_video(testfile)
