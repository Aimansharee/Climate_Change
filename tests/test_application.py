import os
import pytest
import pandas as pd
from main import load_custom_css


def test_custom_css_file_exists():
    """Verify that style.css file exists in root."""
    assert os.path.exists("style.css"), "style.css should exist in the repository."
    with open("style.css", "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 0
    assert "metric-container" in content


def test_filtering_logic_by_location_and_year(real_cleaned_data):
    """Verify filtering behavior given user selections for Location and Year range."""
    selected_locations = list(real_cleaned_data["Location"].unique()[:3])
    start_year = 2005
    end_year = 2015

    filtered = real_cleaned_data[
        (real_cleaned_data["Location"].isin(selected_locations))
        & (real_cleaned_data["Year"] >= start_year)
        & (real_cleaned_data["Year"] <= end_year)
    ]

    assert isinstance(filtered, pd.DataFrame)
    if not filtered.empty:
        assert set(filtered["Location"].unique()).issubset(set(selected_locations))
        assert (filtered["Year"] >= start_year).all()
        assert (filtered["Year"] <= end_year).all()


def test_table_columns_selection(real_cleaned_data):
    """Verify that columns_to_display handles various climate event types correctly."""
    event_types = ["Temperature", "CO2 Emissions"]
    columns_to_display = ["Date", "Location", "Country", "Year"] + event_types

    table = real_cleaned_data[[c for c in columns_to_display if c in real_cleaned_data.columns]]
    assert list(table.columns) == columns_to_display
    assert not table.empty
