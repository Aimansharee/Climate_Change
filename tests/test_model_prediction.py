import pytest
import numpy as np
import pandas as pd
from main import run_kmeans, build_autoencoder, NUMERICAL_FEATURES


def test_kmeans_batch_and_single_prediction(sample_climate_data):
    """Verify K-Means model prediction on batch and single observation."""
    model, _, _ = run_kmeans(sample_climate_data, NUMERICAL_FEATURES, n_clusters=3, random_state=42)

    # Batch prediction
    batch_features = sample_climate_data[NUMERICAL_FEATURES]
    batch_preds = model.predict(batch_features)
    assert len(batch_preds) == len(sample_climate_data)
    assert batch_preds.dtype in [np.int32, np.int64]

    # Single prediction
    single_obs = batch_features.iloc[[0]]
    single_pred = model.predict(single_obs)
    assert len(single_pred) == 1
    assert single_pred[0] == batch_preds[0]


def test_kmeans_deterministic_predictions(sample_climate_data):
    """Verify that predictions on identical inputs yield identical cluster assignments."""
    model, _, _ = run_kmeans(sample_climate_data, NUMERICAL_FEATURES, n_clusters=3, random_state=42)
    features = sample_climate_data[NUMERICAL_FEATURES]

    pred_run_1 = model.predict(features)
    pred_run_2 = model.predict(features)
    assert np.array_equal(pred_run_1, pred_run_2)


def test_autoencoder_inference_dimensions_and_finite_values():
    """Verify Autoencoder inference output dimensions and absence of NaNs or infinities."""
    input_dim = 5
    encoding_dim = 2
    autoencoder, encoder_model = build_autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)

    test_inputs = np.random.uniform(0.0, 1.0, size=(10, input_dim))

    # Encoder inference
    latent = encoder_model.predict(test_inputs, verbose=0)
    assert latent.shape == (10, encoding_dim)
    assert not np.isnan(latent).any()
    assert not np.isinf(latent).any()

    # Full autoencoder reconstruction
    reconstruction = autoencoder.predict(test_inputs, verbose=0)
    assert reconstruction.shape == (10, input_dim)
    assert not np.isnan(reconstruction).any()
    assert (reconstruction >= 0.0).all() and (reconstruction <= 1.0).all()
