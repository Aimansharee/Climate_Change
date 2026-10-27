import pytest
import pandas as pd
import numpy as np
from main import handle_outliers


def test_handle_outliers_removes_extreme_values():
    """Verify that handle_outliers filters rows where z-score exceeds threshold."""
    # Create normal distribution with 2 clear outliers (> 3 sigma)
    np.random.seed(42)
    normal_vals = np.random.normal(loc=20.0, scale=2.0, size=100)
    data = pd.DataFrame({"Temperature": np.append(normal_vals, [100.0, -50.0])})

    cleaned = handle_outliers(data, "Temperature", threshold=3.0)
    assert len(cleaned) < len(data)
    assert 100.0 not in cleaned["Temperature"].values
    assert -50.0 not in cleaned["Temperature"].values


def test_handle_outliers_non_existent_column():
    """Verify that handle_outliers returns unchanged dataframe if column does not exist."""
    df = pd.DataFrame({"Temperature": [10.0, 20.0, 30.0]})
    result = handle_outliers(df, "NonExistentColumn")
    assert len(result) == len(df)
    assert list(result.columns) == list(df.columns)


def test_handle_outliers_custom_threshold():
    """Verify handle_outliers behaves correctly with tighter or looser thresholds."""
    df = pd.DataFrame({"Val": [10, 10, 10, 10, 10, 10, 10, 10, 10, 50]})
    # With strict threshold
    strict = handle_outliers(df, "Val", threshold=1.5)
    assert 50 not in strict["Val"].values


def test_preprocessing_reduces_raw_outliers(csv_file_path):
    """Verify that raw dataset has more rows than preprocessed dataset due to outlier removal."""
    raw_df = pd.read_csv(csv_file_path)
    from main import load_and_preprocess_data

    cleaned_df = load_and_preprocess_data(csv_file_path)
    assert len(raw_df) == 10000
    assert len(cleaned_df) < len(raw_df)
    assert len(cleaned_df) == 9943  # Confirmed 57 outlier rows filtered
