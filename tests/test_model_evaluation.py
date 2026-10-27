import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from main import run_kmeans, run_dbscan, NUMERICAL_FEATURES


def test_kmeans_cluster_evaluation_metrics(real_cleaned_data):
    """Verify standard clustering evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)."""
    # Use subset for fast test execution
    subset = real_cleaned_data.iloc[:500]
    X = subset[NUMERICAL_FEATURES]

    for k in [2, 3, 4]:
        model, labels, sil = run_kmeans(subset, NUMERICAL_FEATURES, n_clusters=k, random_state=42)

        assert sil is not None
        assert -1.0 <= sil <= 1.0

        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)

        assert db_score >= 0.0, "Davies-Bouldin index must be non-negative."
        assert ch_score > 0.0, "Calinski-Harabasz index must be strictly positive."


def test_dbscan_evaluation_with_noise_handling(sample_climate_data):
    """Verify that DBSCAN returns None for silhouette score when no clusters or single cluster is found."""
    # When eps is extremely small (e.g. 0.001), all points become noise (-1)
    _, labels, sil = run_dbscan(sample_climate_data, NUMERICAL_FEATURES, eps=0.001, min_samples=5)
    assert sil is None, "Silhouette score must be None when all points are classified as noise."


def test_autoencoder_reconstruction_error(sample_climate_data):
    """Verify that Autoencoder Mean Squared Error reconstruction loss decreases or is finite."""
    from main import build_autoencoder
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(sample_climate_data[NUMERICAL_FEATURES])

    autoencoder, _ = build_autoencoder(input_dim=5, encoding_dim=2)
    history = autoencoder.fit(X_scaled, X_scaled, epochs=5, batch_size=4, verbose=0)

    loss_values = history.history["loss"]
    assert len(loss_values) == 5
    assert all(np.isfinite(l) for l in loss_values)
    assert loss_values[-1] < 1.0, f"Reconstruction MSE should be reasonable, got {loss_values[-1]}"
