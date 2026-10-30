import inspect
import pandas as pd
import numpy as np


def fixtype(x):
    frame = inspect.currentframe()
    arginfo = inspect.getargvalues(frame.f_back)
    locals = arginfo.locals
    param_names = arginfo.args  # ignores kwargs

    pd_index = None
    np_array = False
    for name in param_names:
        val = locals[name]
        if isinstance(val, pd.Series):
            pd_index = val.index
            break
        elif isinstance(val, np.ndarray):
            np_array = True
            break

    if pd_index is not None:
        x = pd.Series(x, index=pd_index)
        return x
    if not np_array:
        return x.item()
    return x
