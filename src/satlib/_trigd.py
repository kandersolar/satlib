import numpy as np

def sind(x):
    return np.sin(np.radians(x))


def cosd(x):
    return np.cos(np.radians(x))


def tand(x):
    return np.tan(np.radians(x))


def asind(x, **kwargs):
    return np.degrees(np.asin(x, **kwargs))


def acosd(x, **kwargs):
    return np.degrees(np.acos(x, **kwargs))


def atand(x, **kwargs):
    return np.degrees(np.atan(x, **kwargs))


def atan2d(x, y, **kwargs):
    return np.degrees(np.arctan2(x, y, **kwargs))
