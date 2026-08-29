import numpy as np
import pandas as pd
from typing import Tuple


def parse_urine_score(label: str) -> float:
    """Parse the numeric prefix from a radio label like '2: Medium Green'.

    Returns 0.0 on parsing failure.
    """
    try:
        return float(label.split(":")[0])
    except Exception:
        return 0.0


def extract_positive_probability(prob_array) -> float:
    """Safely extract the positive-class probability from predict_proba output.

    Handles numpy arrays, lists, and unexpected shapes.
    """
    try:
        return float(prob_array[0, 1])
    except Exception:
        try:
            return float(prob_array[0][1])
        except Exception:
            # Fallback to first element
            return float(prob_array[0][0])


def validate_input_vector_columns(input_vector: pd.DataFrame, model) -> Tuple[bool, str]:
    """Validate that the input_vector columns match the trained model's expected features.

    Returns (True, "") if valid, otherwise (False, message).
    """
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        present = list(input_vector.columns)
        if set(expected) != set(present):
            missing = [c for c in expected if c not in present]
            extra = [c for c in present if c not in expected]
            msg_parts = []
            if missing:
                msg_parts.append(f"missing columns: {missing}")
            if extra:
                msg_parts.append(f"unexpected columns: {extra}")
            return False, "; ".join(msg_parts)
        # Optionally check order
        if expected != present:
            return True, f"column order differs. expected order: {expected}"
        return True, ""
    else:
        # If the model doesn't expose feature names, do a best-effort check for count
        if input_vector.shape[1] == getattr(model, 'n_features_in_', input_vector.shape[1]):
            return True, ""
        return False, f"expected {getattr(model, 'n_features_in_', '?')} features, got {input_vector.shape[1]}"
