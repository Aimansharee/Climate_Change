import ast
import os
import pytest
import numpy as np
import pandas as pd
from main import (
    run_kmeans,
    run_dbscan,
    build_autoencoder,
    load_custom_css,
    load_and_preprocess_data,
    NUMERICAL_FEATURES,
)


def test_regression_bug_001_clustering_not_in_global_scope():
    """BUG-001 Regression Test: Ensure clustering and AI models do not execute at global module level."""
    with open("main.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    # Check top-level statements: no KMeans, DBSCAN, or Autoencoder fit calls allowed at module root
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call_name = ""
            if isinstance(node.value.func, ast.Name):
                call_name = node.value.func.id
            elif isinstance(node.value.func, ast.Attribute):
                call_name = node.value.func.attr
            assert call_name not in ["fit", "fit_predict"], (
                f"Regression BUG-001: Model fitting '{call_name}' must not occur in global module scope."
            )


def test_regression_bug_002_autoencoder_independent_of_dbscan(sample_climate_data):
    """BUG-002 Regression Test: Autoencoder must execute successfully even if DBSCAN produces 100% noise."""
    # Force DBSCAN to produce 100% noise
    _, dbscan_labels, dbscan_sil = run_dbscan(
        sample_climate_data, NUMERICAL_FEATURES, eps=0.0001, min_samples=10
    )
    assert set(dbscan_labels) == {-1}, "DBSCAN should classify all as noise with tiny eps."
    assert dbscan_sil is None

    # Autoencoder must build and run independently without dependency on DBSCAN results
    autoencoder, encoder_model = build_autoencoder(input_dim=5, encoding_dim=2)
    assert autoencoder is not None
    assert encoder_model is not None

    test_input = np.random.uniform(0.0, 1.0, size=(5, 5))
    encoded = encoder_model.predict(test_input, verbose=0)
    assert encoded.shape == (5, 2)
    assert not np.isnan(encoded).any()


def test_regression_bug_003_dbscan_eps_handles_noise_and_clusters(sample_climate_data):
    """BUG-003 Regression Test: DBSCAN handles valid clusters and edge-case noise safely."""
    # Test valid clustering with appropriate scale
    _, labels_clustered, sil_clustered = run_dbscan(
        sample_climate_data, NUMERICAL_FEATURES, eps=15.0, min_samples=2
    )
    assert len(labels_clustered) == len(sample_climate_data)

    # Test pure noise does not cause division by zero or crash in silhouette calculation
    _, labels_noise, sil_noise = run_dbscan(
        sample_climate_data, NUMERICAL_FEATURES, eps=0.0001, min_samples=10
    )
    assert sil_noise is None


def test_regression_bug_004_autoencoder_sigmoid_target_scaling(sample_climate_data):
    """BUG-004 Regression Test: Verify Autoencoder receives normalized [0, 1] inputs for sigmoid activation."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sample_climate_data[NUMERICAL_FEATURES])

    assert np.all(scaled_data >= -1e-7) and np.all(scaled_data <= 1.0 + 1e-7)

    autoencoder, _ = build_autoencoder(input_dim=5, encoding_dim=2)
    history = autoencoder.fit(scaled_data, scaled_data, epochs=10, batch_size=4, verbose=0)

    # Sigmoid outputs stay in [0, 1], so MSE against [0, 1] targets must be reasonably low (< 0.25)
    final_mse = history.history["loss"][-1]
    assert final_mse < 0.25, f"Regression BUG-004: Reconstruction MSE too high ({final_mse})."


def test_regression_bug_005_no_unresolved_placeholders_in_code_or_report():
    """BUG-005 Regression Test: Ensure no unreplaced placeholders remain in main.py."""
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_placeholders = [
        "[number of rows]",
        "[number of columns]",
        "{silhouette_kmeans}",
        "{silhouette_dbscan}",
        "[Kaggle/UCI/Google Dataset Search]",
    ]

    for placeholder in forbidden_placeholders:
        assert placeholder not in content, (
            f"Regression BUG-005: Unresolved placeholder '{placeholder}' found in main.py."
        )


def test_regression_bug_006_no_deprecated_tensorflow_python_imports():
    """BUG-006 Regression Test: Ensure tensorflow.python.keras internal module is never imported."""
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()

    assert "tensorflow.python.keras" not in content, (
        "Regression BUG-006: Deprecated import 'tensorflow.python.keras' found in main.py."
    )
    assert "tensorflow.keras" in content, (
        "Standard public API 'tensorflow.keras' should be used instead."
    )


def test_regression_bug_007_no_redundant_imports():
    """BUG-007 Regression Test: Ensure no duplicate or unused heavy imports."""
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()

    # silhouette_score must not be imported multiple times
    count = content.count("silhouette_score")
    # Must only appear once in import statements
    import_lines = [line for line in content.splitlines() if "import" in line and "silhouette_score" in line]
    assert len(import_lines) == 1, (
        f"Regression BUG-007: Multiple import lines for silhouette_score: {import_lines}"
    )

    # Unused modules must not be imported
    assert "from sklearn.decomposition import PCA" not in content
    assert "from sklearn.manifold import TSNE" not in content


def test_regression_bug_008_style_css_is_loaded(tmp_path):
    """BUG-008 Regression Test: Ensure style.css loader executes safely and handles missing files gracefully."""
    # Existing style.css in project
    assert os.path.exists("style.css"), "style.css must exist in project root."
    load_custom_css("style.css")  # Should not raise exception

    # Non-existent CSS file should gracefully do nothing without crashing
    non_existent = str(tmp_path / "missing.css")
    load_custom_css(non_existent)  # Should not raise exception


def test_regression_bug_009_autoencoder_caching_and_deterministic_output():
    """BUG-009 Regression Test: Verify Autoencoder caching and deterministic latent dimensionality."""
    from main import get_trained_autoencoder

    test_tuples = tuple(map(tuple, np.random.uniform(0.0, 1.0, size=(10, 5))))

    # First call (trains and caches)
    ae1, enc1, lat1 = get_trained_autoencoder(test_tuples, input_dim=5, encoding_dim=2, epochs=5)
    # Second call (returns cached instance)
    ae2, enc2, lat2 = get_trained_autoencoder(test_tuples, input_dim=5, encoding_dim=2, epochs=5)

    assert lat1.shape == (10, 2)
    assert np.array_equal(lat1, lat2), "Cached Autoencoder should yield deterministic representations."
