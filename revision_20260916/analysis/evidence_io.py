"""Read archived floating-point evidence without avoidable decimal conversion drift."""
import pandas as pd

def read_csv(*args, **kwargs):
    kwargs.setdefault('float_precision', 'round_trip')
    return pd.read_csv(*args, **kwargs)
