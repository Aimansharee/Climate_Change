import pytest
import pandas as pd
import numpy as np


def test_required_columns_presence(real_cleaned_data):
    """Verify that all expected climate feature columns are present."""
    expected_columns = [
        "Date",
        "Location",
        "Country",
        "Temperature",
        "CO2 Emissions",
        "Sea Level Rise",
        "Precipitation",
        "Humidity",
        "Wind Speed",
        "Year",
    ]
    for col in expected_columns:
        assert col in real_cleaned_data.columns, f"Missing required column: {col}"


def test_no_null_values_in_cleaned_dataset(real_cleaned_data):
    """Verify that no NaN or null values remain in the cleaned dataset."""
    null_counts = real_cleaned_data.isnull().sum()
    assert null_counts.sum() == 0, f"Cleaned dataset contains unexpected null values:\n{null_counts}"


def test_temperature_values_range(real_cleaned_data):
    """Verify that Temperature values fall within realistic planetary bounds."""
    min_temp = real_cleaned_data["Temperature"].min()
    max_temp = real_cleaned_data["Temperature"].max()
    assert min_temp >= -50.0, f"Unrealistic minimum temperature: {min_temp}°C"
    assert max_temp <= 65.0, f"Unrealistic maximum temperature: {max_temp}°C"


def test_co2_emissions_range(real_cleaned_data):
    """Verify that CO2 emissions fall within atmospheric range (ppm)."""
    min_co2 = real_cleaned_data["CO2 Emissions"].min()
    max_co2 = real_cleaned_data["CO2 Emissions"].max()
    assert min_co2 > 100.0, f"CO2 concentration too low: {min_co2} ppm"
    assert max_co2 < 1000.0, f"CO2 concentration too high: {max_co2} ppm"


def test_bounded_environmental_indicators(real_cleaned_data):
    """Verify bounded physical indicators (Humidity 0-100%, Precipitation >= 0, Wind Speed >= 0)."""
    assert (real_cleaned_data["Humidity"] >= 0).all()
    assert (real_cleaned_data["Humidity"] <= 100.0).all()

    assert (real_cleaned_data["Precipitation"] >= 0).all()
    assert (real_cleaned_data["Wind Speed"] >= 0).all()


def test_geographical_features(real_cleaned_data):
    """Verify that Country and Location contain valid non-empty string entries."""
    assert real_cleaned_data["Country"].nunique() > 50, "Should contain multiple unique countries."
    assert real_cleaned_data["Location"].nunique() > 100, "Should contain multiple unique locations."
    assert (real_cleaned_data["Country"].str.strip() != "").all()
    assert (real_cleaned_data["Location"].str.strip() != "").all()
