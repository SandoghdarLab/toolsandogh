import pathlib

from toolsandogh.io import generate_video, load_video


def test_generate_video():
    generate_video()


def test_load_video():
    parent = pathlib.Path(__file__).resolve().parent
    testfile = str(parent / "testfile.tiff")
    load_video(testfile)
