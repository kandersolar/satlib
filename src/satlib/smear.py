import pandas as pd


def _smear(x, limit):
    ts = x.index[1] - x.index[0]
    limit = limit * (ts / pd.Timedelta("1min"))
    diff = x.diff()
    carryover = 0
    for i in range(1, len(diff)):
        d = diff.iloc[i]
        d = d + carryover
        if d > limit:
            diff.iloc[i] = limit
            carryover = d - limit
        elif d < -limit:
            diff.iloc[i] = -limit
            carryover = d + limit
        else:
            diff.iloc[i] = d
            carryover = 0
    x = x.copy()
    x.iloc[1:] = x.iloc[0] + diff.cumsum().iloc[1:]
    return x

def _reverse(x):
    return pd.Series(x.values[::-1], index=x.index)

def smear(x, limit, how):
    if how == "catch-up":
        return _smear(x, limit)
    if how == "anticipate":
        return _reverse(_smear(_reverse(x), limit))
    if how == "steeper":
        x1 = x[x < 0]
        x2 = x[x > 0]
        y1 = _smear(x1, limit)
        y2 = _reverse(_smear(_reverse(x2), limit))
        out = x.copy()
        out[x < 0] = y1
        out[x > 0] = y2
        return out
    if how == "shallower":
        x1 = x[x < 0]
        x2 = x[x > 0]
        y1 = _reverse(_smear(_reverse(x1), limit))
        y2 = _smear(x2, limit)
        out = x.copy()
        out[x < 0] = y1
        out[x > 0] = y2
        return out
    raise
