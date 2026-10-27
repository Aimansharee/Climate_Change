import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def test_dual_axis_time_series_figure(sample_climate_data):
    """Verify creation of dual-axis Plotly figure for temperature and CO2."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_climate_data["Date"],
            y=sample_climate_data["Temperature"],
            mode="lines+markers",
            name="Temperature (°C)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_climate_data["Date"],
            y=sample_climate_data["CO2 Emissions"],
            mode="lines+markers",
            name="CO2 Emissions (ppm)",
            yaxis="y2",
        )
    )
    fig.update_layout(yaxis2=dict(anchor="x", overlaying="y", side="right"))

    assert len(fig.data) == 2
    assert fig.data[0].name == "Temperature (°C)"
    assert fig.data[1].name == "CO2 Emissions (ppm)"


def test_choropleth_map_figure(sample_climate_data):
    """Verify generation of global precipitation choropleth map."""
    agg_df = sample_climate_data.groupby("Country").agg({"Precipitation": "mean"}).reset_index()
    fig = px.choropleth(
        agg_df,
        locations="Country",
        locationmode="country names",
        color="Precipitation",
        hover_name="Country",
    )
    assert fig is not None
    assert len(fig.data) > 0


def test_top_countries_bar_chart(sample_climate_data):
    """Verify top countries by CO2 emissions bar chart creation."""
    co2_agg = sample_climate_data.groupby("Country")["CO2 Emissions"].mean().reset_index()
    fig = px.bar(co2_agg, x="Country", y="CO2 Emissions")
    assert fig is not None
    assert len(fig.data) > 0


def test_correlation_heatmap_rendering(sample_climate_data, feature_names):
    """Verify correlation heatmap generation with matplotlib and seaborn."""
    corr = sample_climate_data[feature_names].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    assert fig is not None
    assert ax is not None
    plt.close(fig)
