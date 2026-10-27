# 🧪 Climate Change AI — Automated Test Suite

Comprehensive automated test suite for the Climate Change AI analytics and modeling dashboard. Built with **pytest**, this suite validates data loading, data validation, preprocessing routines, machine learning clustering, deep learning autoencoders, visualizations, application modules, and dedicated regression tests for all documented defects.

---

## 📂 Test Suite Structure

```text
tests/
├── conftest.py                   # Pytest fixtures (real & synthetic datasets, column schemas)
├── test_imports.py               # Dependency, library version, and export validation (3 tests)
├── test_data_loading.py          # CSV loading, chronological sorting, datetime parsing (5 tests)
├── test_data_validation.py       # Data integrity, schema validation, physical bounds (6 tests)
├── test_preprocessing.py         # Outlier filtering with Z-score, sequential vs set reduction (4 tests)
├── test_models.py                # K-Means, DBSCAN, and Autoencoder architecture validation (5 tests)
├── test_model_loading.py         # Hyperparameters and Keras model save/load round-trip (2 tests)
├── test_model_prediction.py      # Batch and single inference, deterministic prediction checks (3 tests)
├── test_model_evaluation.py      # Silhouette, Davies-Bouldin, Calinski-Harabasz & MSE metrics (3 tests)
├── test_regression.py            # Dedicated regression tests for BUG-001 through BUG-009 (9 tests)
├── test_visualization.py         # Plotly time-series, Choropleth maps, Seaborn heatmaps (4 tests)
├── test_application.py           # CSS stylesheet loading, sidebar filter logic (3 tests)
└── README.md                     # Test suite documentation (this file)
```

---

## 🎯 Test Categories & Coverage (47 Tests Total)

| Category | File | Description | Tests Count |
|---|---|---|---|
| **Imports & Environment** | `test_imports.py` | Validates dependencies (`pandas`, `numpy`, `tensorflow`, `streamlit`, `sklearn`) and public API availability. | 3 |
| **Data Ingestion** | `test_data_loading.py` | Validates file existence, non-emptiness, datetime parsing, chronological order, and error handling. | 5 |
| **Data Quality & Physical Bounds** | `test_data_validation.py` | Verifies zero nulls, bounded ranges (Temperature: -50 to 65°C, CO2: 100 to 1000 ppm, Humidity: 0-100%). | 6 |
| **Preprocessing & Cleaning** | `test_preprocessing.py` | Tests Z-score outlier detection ($\pm 3\sigma$) on individual features and verifies row count reduction. | 4 |
| **AI / ML Model Execution** | `test_models.py` | Verifies K-Means fit, DBSCAN Knee method distances, and Autoencoder 5D $\to$ 2D compression. | 5 |
| **Model Persistence & Config** | `test_model_loading.py` | Verifies hyperparameter assignment and Keras `.keras` model save/load round-trip. | 2 |
| **Inference & Prediction** | `test_model_prediction.py` | Verifies batch and single sample prediction consistency, finite values, and absence of NaNs. | 3 |
| **Model Performance Metrics** | `test_model_evaluation.py` | Evaluates Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, and reconstruction loss. | 3 |
| **Bug Regression Suite** | `test_regression.py` | Explicit regression tests verifying that BUG-001 through BUG-009 cannot be reintroduced. | 9 |
| **Visualization & Plotting** | `test_visualization.py` | Validates Plotly traces (Dual-axis, Choropleth, Bar) and Seaborn/Matplotlib heatmaps. | 4 |
| **Application Integration** | `test_application.py` | Validates custom CSS loading and multi-attribute slice filtering logic. | 3 |

---

## 🚀 How to Execute the Tests

### 1. Run All 47 Tests
```bash
python -m pytest tests/ -v
```

### 2. Run Dedicated Regression Tests
```bash
python -m pytest tests/test_regression.py -v
```

### 3. Run Specific Test Module
```bash
python -m pytest tests/test_models.py -v
```

---

## ⚙️ Fixtures in `conftest.py`

- `csv_file_path`: Path to `climate_change_data.csv`.
- `real_cleaned_data`: Cleaned dataset loaded through `main.load_and_preprocess_data()` (cached for session).
- `sample_climate_data`: Lightweight 10-row synthetic climate dataframe for fast, deterministic unit tests.
- `feature_names`: List of standard 5 numerical features (`NUMERICAL_FEATURES`).

---

## 📚 Related Documentation

For comprehensive reports and guides, refer to the [docs/](file:///f:/climate-AI/docs) directory:
- [docs/TESTING_GUIDE.md](file:///f:/climate-AI/docs/TESTING_GUIDE.md): Testing setup and troubleshooting.
- [docs/TEST_REPORT.md](file:///f:/climate-AI/docs/TEST_REPORT.md): Detailed verification metrics and test execution logs.
- [docs/BUG_REPORT.md](file:///f:/climate-AI/docs/BUG_REPORT.md): Defect logs and resolution histories.
