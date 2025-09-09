import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple


def reconstruct_to_original(
    df_encoded: pd.DataFrame,
    preprocesser: Any,
    info: Dict[str, Any],
) -> pd.DataFrame:
    """Decode an encoded dataframe back to original numeric/categorical spaces.

    - df_encoded: dataframe of integer-coded values (after synthesis)
    - preprocesser: instance of preprocess_common.load_data_common.data_preporcesser_common
                    already fitted during preprocessing
    - info: dict with 'num_columns' and 'cat_columns' listing original column names
    """
    x_num_rev, x_cat_rev = preprocesser.reverse_data(df_encoded)
    num_cols = info.get("num_columns", []) or []
    cat_cols = info.get("cat_columns", []) or []

    if x_num_rev is not None and x_cat_rev is not None:
        out = pd.DataFrame(
            np.concatenate((x_num_rev, x_cat_rev), axis=1),
            columns=num_cols + cat_cols,
        )
    elif x_num_rev is not None:
        out = pd.DataFrame(x_num_rev, columns=num_cols)
    elif x_cat_rev is not None:
        out = pd.DataFrame(x_cat_rev, columns=cat_cols)
    else:
        # If preprocesser cannot decode, pass through columns
        out = df_encoded.copy()
        out.columns = num_cols + cat_cols
    return out

