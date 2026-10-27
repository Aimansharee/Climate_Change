import pytest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from main import (
    run_kmeans,
    compute_knee_distances,
    run_dbscan,
    build_autoencoder,
    NUMERICAL_FEATURES,
)


def test_kmeans_training_and_silhouette(sample_climate_data):
    """Verify K-Means model training, output labels, and silhouette score calculation."""
    n_clusters = 3
    model, labels, sil_score = run_kmeans(
        sample_climate_data,
        features=NUMERICAL_FEATURES,
        n_clusters=n_clusters,
        random_state=42,
    )

    assert isinstance(model, KMeans)
    assert len(labels) == len(sample_climate_data)
    assert set(labels).issubset(set(range(n_clusters)))
    assert len(set(labels)) == n_clusters
    assert isinstance(sil_score, float)
    assert -1.0 <= sil_score <= 1.0


def test_knee_method_distances(sample_climate_data):
    """Verify that compute_knee_distances returns sorted positive distances."""
    distances = compute_knee_distances(sample_climate_data, NUMERICAL_FEATURES, n_neighbors=3)

    assert isinstance(distances, np.ndarray)
    assert len(distances) == len(sample_climate_data)
    assert (distances >= 0).all()
    # Check monotonicity
    assert np.all(np.diff(distances) >= 0), "Distances for knee method must be sorted ascending."


def test_dbscan_clustering_execution(sample_climate_data):
    """Verify DBSCAN model execution, label generation, and cluster count."""
    model, labels, sil_score = run_dbscan(
        sample_climate_data,
        features=NUMERICAL_FEATURES,
        eps=15.0,
        min_samples=2,
    )

    assert isinstance(model, DBSCAN)
    assert len(labels) == len(sample_climate_data)
    if sil_score is not None:
        assert isinstance(sil_score, float)
        assert -1.0 <= sil_score <= 1.0


def test_autoencoder_architecture():
    """Verify Autoencoder layer dimensions, input shape, and encoder latent dimension."""
    input_dim = 5
    encoding_dim = 2
    autoencoder, encoder_model = build_autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)

    assert autoencoder.input_shape == (None, input_dim)
    assert autoencoder.output_shape == (None, input_dim)
    assert encoder_model.output_shape == (None, encoding_dim)


def test_autoencoder_training_and_latent_representation():
    """Verify that Autoencoder trains on normalized data and produces 2D embeddings without NaNs."""
    np.random.seed(42)
    X = np.random.uniform(0.0, 1.0, size=(64, 5))

    autoencoder, encoder_model = build_autoencoder(input_dim=5, encoding_dim=2)
    history = autoencoder.fit(X, X, epochs=3, batch_size=16, verbose=0)

    # Check loss is finite and valid
    final_loss = history.history["loss"][-1]
    assert np.isfinite(final_loss)
    assert final_loss >= 0.0

    # Predict latent encoding
    encoded = encoder_model.predict(X, verbose=0)
    assert encoded.shape == (64, 2)
    assert not np.isnan(encoded).any()
    assert not np.isinf(encoded).any()
