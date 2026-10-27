import os
import pytest
import pandas as pd
from main import load_and_preprocess_data


def test_dataset_file_exists():
    """Verify that climate_change_data.csv exists in the project root."""
    assert os.path.exists("climate_change_data.csv"), "climate_change_data.csv must exist."
    assert os.path.getsize("climate_change_data.csv") > 0, "Dataset file must not be empty."


def test_load_and_preprocess_returns_dataframe(csv_file_path):
    """Verify that load_and_preprocess_data returns a valid, non-empty pandas DataFrame."""
    df = load_and_preprocess_data(csv_file_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 9000, f"Expected >9000 rows after cleaning, got {len(df)}"


def test_load_missing_file_raises_error():
    """Verify that attempting to load a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data("non_existent_climate_file.csv")


def test_datetime_and_year_extraction(real_cleaned_data):
    """Verify that 'Date' is converted to datetime and 'Year' column is properly extracted."""
    assert "Date" in real_cleaned_data.columns
    assert "Year" in real_cleaned_data.columns
    assert pd.api.types.is_datetime64_any_dtype(real_cleaned_data["Date"])
    assert pd.api.types.is_integer_dtype(real_cleaned_data["Year"])
    assert real_cleaned_data["Year"].min() >= 2000
    assert real_cleaned_data["Year"].max() <= 2025


def test_data_is_sorted_chronologically(real_cleaned_data):
    """Verify that rows are sorted in ascending order by Date."""
    dates = real_cleaned_data["Date"].reset_index(drop=True)
    assert dates.is_monotonic_increasing, "Dataset should be sorted chronologically by Date."
