import pytest


def test_core_library_imports():
    """Verify that all core libraries specified in requirements.txt can be imported."""
    import streamlit
    import pandas
    import numpy
    import scipy
    import sklearn
    import plotly
    import matplotlib
    import seaborn
    import streamlit_option_menu
    import tensorflow

    assert streamlit.__name__ == "streamlit"
    assert pandas.__name__ == "pandas"
    assert numpy.__name__ == "numpy"
    assert scipy.__name__ == "scipy"
    assert sklearn.__name__ == "sklearn"
    assert plotly.__name__ == "plotly"
    assert matplotlib.__name__ == "matplotlib"
    assert seaborn.__name__ == "seaborn"
    assert streamlit_option_menu.__name__ == "streamlit_option_menu"
    assert tensorflow.__name__ == "tensorflow"


def test_tensorflow_keras_public_api():
    """Verify that standard Keras public API is imported without deprecated internal paths."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense

    assert Model is not None
    assert Input is not None
    assert Dense is not None


def test_main_module_exports():
    """Verify that main.py exports expected pipeline functions and constants."""
    import main

    expected_callables = [
        "handle_outliers",
        "load_and_preprocess_data",
        "load_custom_css",
        "run_kmeans",
        "compute_knee_distances",
        "run_dbscan",
        "build_autoencoder",
        "HomePage",
        "GraphsPage",
        "EDAPage",
        "ClusteringPage",
        "FinalPage",
        "main",
    ]

    for func_name in expected_callables:
        assert hasattr(main, func_name), f"main.py should define function '{func_name}'"
        assert callable(getattr(main, func_name)), f"'{func_name}' should be callable"

    assert hasattr(main, "NUMERICAL_FEATURES")
    assert isinstance(main.NUMERICAL_FEATURES, list)
    assert len(main.NUMERICAL_FEATURES) == 5
