import os
import pytest
import pandas as pd
import numpy as np
from main import load_and_preprocess_data, NUMERICAL_FEATURES

CSV_PATH = "climate_change_data.csv"


@pytest.fixture(scope="session")
def csv_file_path():
    """Return the absolute or relative path to the project dataset."""
    assert os.path.exists(CSV_PATH), f"Dataset file {CSV_PATH} must exist for testing."
    return CSV_PATH


@pytest.fixture(scope="session")
def real_cleaned_data(csv_file_path):
    """Return the preprocessed and outlier-cleaned dataset."""
    df = load_and_preprocess_data(csv_file_path)
    return df


@pytest.fixture
def sample_climate_data():
    """Return a synthetic small DataFrame mimicking the real dataset schema for fast unit tests."""
    data = {
        "Date": [
            "2020-01-01 00:00:00",
            "2020-02-01 00:00:00",
            "2020-03-01 00:00:00",
            "2020-04-01 00:00:00",
            "2020-05-01 00:00:00",
            "2020-06-01 00:00:00",
            "2020-07-01 00:00:00",
            "2020-08-01 00:00:00",
            "2020-09-01 00:00:00",
            "2020-10-01 00:00:00",
        ],
        "Location": [f"Station_{i}" for i in range(10)],
        "Country": ["USA", "Germany", "France", "Japan", "Egypt", "Brazil", "China", "India", "UK", "Canada"],
        "Temperature": [15.2, 16.1, 14.8, 18.2, 22.1, 25.4, 28.0, 26.5, 20.1, 17.3],
        "CO2 Emissions": [410.5, 412.3, 408.9, 415.0, 418.2, 420.1, 417.8, 419.4, 413.2, 411.0],
        "Sea Level Rise": [0.2, 0.3, 0.1, 0.4, 0.5, 0.4, 0.6, 0.5, 0.3, 0.2],
        "Precipitation": [50.1, 62.4, 45.0, 80.2, 30.5, 120.4, 15.2, 22.8, 70.1, 55.4],
        "Humidity": [60.0, 65.5, 58.2, 72.1, 45.3, 85.0, 40.2, 50.1, 68.4, 62.0],
        "Wind Speed": [12.5, 15.0, 10.2, 20.4, 8.5, 25.1, 14.2, 11.0, 18.3, 13.7],
    }
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    return df


@pytest.fixture
def feature_names():
    """Return the standard numerical feature names."""
    return NUMERICAL_FEATURES
