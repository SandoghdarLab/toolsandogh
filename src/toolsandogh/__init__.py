"""
A collection of tools for working with large-scale microscopy data.

Provided to you by the Sandoghdar Division of the Max Planck Institute for the Physics of Light.
"""

from ._generate_video import generate_video
from ._load_video import load_video

__all__ = [
    "load_video",
    "generate_video",
]
