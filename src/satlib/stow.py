import numpy as np
from satlib._util import fixtype

def night_stow(psza, stow_angle=0):
    """
    
    Parameters
    ----------
    psza : numeric
        Projected solar zenith angle. [degrees]
    stow_angle : numeric, default 0
        The stow position. [degrees]

    Returns
    -------
    stow : numeric
        stow_angle where abs(psza) > 90, and nan otherwise
    """
    psza = ((psza + 180) % 360) - 180
    out = np.where(np.abs(psza) > 90, stow_angle, np.nan)
    return fixtype(out)
