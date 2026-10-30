from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__)
except PackageNotFoundError:
    __version__ = "0+unknown"

from satlib import (
    backtracking,
    diffuse_optimization,
    discretization,
    geometry,
    limits,
    shading,
    tracking,
    twist,
    _trigd,
    _util,
)
