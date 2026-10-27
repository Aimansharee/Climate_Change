import os
import tempfile
import pytest
import numpy as np
from main import build_autoencoder, run_kmeans, NUMERICAL_FEATURES


def test_kmeans_hyperparameters(sample_climate_data):
    """Verify K-Means model configuration and initialization parameters."""
    model, _, _ = run_kmeans(sample_climate_data, NUMERICAL_FEATURES, n_clusters=4, random_state=123)
    assert model.n_clusters == 4
    assert model.random_state == 123
    assert hasattr(model, "cluster_centers_")
    assert model.cluster_centers_.shape == (4, 5)


def test_autoencoder_model_save_and_load(tmp_path):
    """Verify that the Autoencoder model can be safely saved to and loaded from disk."""
    import tensorflow as tf

    autoencoder, _ = build_autoencoder(input_dim=5, encoding_dim=2)
    test_input = np.ones((1, 5), dtype=np.float32)
    initial_pred = autoencoder.predict(test_input, verbose=0)

    save_path = str(tmp_path / "test_autoencoder.keras")
    autoencoder.save(save_path)
    assert os.path.exists(save_path)

    loaded_model = tf.keras.models.load_model(save_path, compile=False)
    loaded_pred = loaded_model.predict(test_input, verbose=0)

    np.testing.assert_allclose(initial_pred, loaded_pred, rtol=1e-5, atol=1e-5)
