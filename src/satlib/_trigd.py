import numpy as np

def sind(x):
    return np.sin(np.radians(x))


def cosd(x):
    return np.cos(np.radians(x))


def tand(x):
    return np.tan(np.radians(x))


def asind(x):
    return np.degrees(np.asin(x))


def acosd(x):
    return np.degrees(np.acos(x))


def atand(x):
    return np.degrees(np.atan(x))


def atan2d(x, y):
    return np.degrees(np.arctan2(x, y))
