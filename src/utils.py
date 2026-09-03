"""Shared helper functions reused across the project notebooks."""
import ast
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def parse_price(series: pd.Series) -> pd.Series:
    """Convert a '$1,234.00'-style string column to float."""
    return (
        series.astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )


def parse_list_col(x):
    """Parse a stringified Python list (amenities / host_verifications) into a real list."""
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        try:
            return json.loads(str(x).replace("'", '"'))
        except Exception:
            return []


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
