# 🐛 Climate Change AI — Bug Report & Remediation Log

This document records all defects identified during comprehensive static analysis, execution testing, and AI model verification of the **Climate Change AI** project.

---

## Summary of Defects & Remediation Status

| Bug ID | Severity | File Affected | Summary | Regression Test | Final Status |
|---|---|---|---|---|---|
| **BUG-001** | Critical | `main.py` | Clustering and Autoencoder code executed globally on every page | `tests/test_regression.py::test_regression_bug_001_clustering_not_in_global_scope` | **FIXED & VERIFIED** |
| **BUG-002** | High | `main.py` | Autoencoder trapped inside DBSCAN `else:` branch | `tests/test_regression.py::test_regression_bug_002_autoencoder_independent_of_dbscan` | **FIXED & VERIFIED** |
| **BUG-003** | Medium | `main.py` | DBSCAN `eps=2.0` default too small for unscaled 5D features (100% noise) | `tests/test_regression.py::test_regression_bug_003_dbscan_eps_handles_noise_and_clusters` | **FIXED & VERIFIED** |
| **BUG-004** | High | `main.py` | Autoencoder trained on unnormalized features with `sigmoid` activation | `tests/test_regression.py::test_regression_bug_004_autoencoder_sigmoid_target_scaling` | **FIXED & VERIFIED** |
| **BUG-005** | Medium | `main.py` | Hardcoded string placeholders and unformatted metric expressions in Final Report | `tests/test_regression.py::test_regression_bug_005_no_unresolved_placeholders_in_code_or_report` | **FIXED & VERIFIED** |
| **BUG-006** | Medium | `main.py` | Deprecated internal TensorFlow import `tensorflow.python.keras` | `tests/test_regression.py::test_regression_bug_006_no_deprecated_tensorflow_python_imports` | **FIXED & VERIFIED** |
| **BUG-007** | Low | `main.py` | Duplicate and unused imports (`PCA`, `TSNE`, `silhouette_score`) | `tests/test_regression.py::test_regression_bug_007_no_redundant_imports` | **FIXED & VERIFIED** |
| **BUG-008** | Low | `style.css` / `main.py` | `style.css` present in root but never injected or loaded | `tests/test_regression.py::test_regression_bug_008_style_css_is_loaded` | **FIXED & VERIFIED** |
| **BUG-009** | Medium | `main.py` | Autoencoder trained synchronously on every interaction without caching | `tests/test_regression.py::test_regression_bug_009_autoencoder_caching_and_deterministic_output` | **FIXED & VERIFIED** |

---

## Detailed Bug Reports & Verification Evidence

### BUG-001: Clustering and Autoencoder Code Executed Globally on Every Page
- **Bug ID:** `BUG-001`
- **Severity:** Critical
- **Affected File:** `main.py` (originally lines 316–407)
- **Original Problem:** The code performing K-Means, DBSCAN, and Autoencoder training was located at the root level of `main.py` after the `elif selected == "EDA":` block instead of inside the `elif selected == "Clustering Analysis":` branch.
- **Root Cause:** Incomplete block indentation and missing encapsulation in a dedicated page function.
- **Applied Fix:** Encapsulated clustering logic into `ClusteringPage(data, event_type)` and connected it exclusively to `selected == "Clustering Analysis"`.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_001_clustering_not_in_global_scope`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_001_clustering_not_in_global_scope -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**

---

### BUG-002: Autoencoder Trapped Inside DBSCAN `else:` Branch
- **Bug ID:** `BUG-002`
- **Severity:** High
- **Affected File:** `main.py` (originally lines 376–407)
- **Original Problem:** The Autoencoder model definition, compilation, training, and visualization were indented inside the `else:` clause of `if len(set(data['DBSCAN_Cluster'])) > 1:`.
- **Root Cause:** Faulty code block indentation.
- **Applied Fix:** Un-indented the Autoencoder logic into an independent model section within `ClusteringPage`.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_002_autoencoder_independent_of_dbscan`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_002_autoencoder_independent_of_dbscan -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**

---

### BUG-003: DBSCAN `eps=2.0` Default Too Small for Unscaled 5D Features
- **Bug ID:** `BUG-003`
- **Severity:** Medium
- **Affected File:** `main.py` (originally lines 350–355)
- **Original Problem:** The default distance threshold for DBSCAN was set to `eps = 2.0` with a max of `10.0`. However, Euclidean distances in the 5-dimensional feature space (where $CO_2$ has a standard deviation of 49.7) have a 5th-nearest-neighbor distance starting at 5.7 and reaching 31.9.
- **Root Cause:** Slider range `[0.1, 10.0]` did not match the physical scale of the unscaled dataset.
- **Applied Fix:** Adjusted the `eps` slider range to `[0.1, 20.0]` with default `10.0`, and added robust checks for Silhouette score calculation when only noise or single clusters exist.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_003_dbscan_eps_handles_noise_and_clusters`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_003_dbscan_eps_handles_noise_and_clusters -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**

---

### BUG-004: Autoencoder Trained on Unnormalized Features with Sigmoid Activation
- **Bug ID:** `BUG-004`
- **Severity:** High
- **Affected File:** `main.py` (originally lines 388–394)
- **Original Problem:** The Autoencoder decoder layer was configured with `activation="sigmoid"` (output range $[0, 1]$), while input and reconstruction targets were raw climate features ranging up to $582.9$ ($CO_2$).
- **Root Cause:** Omission of Min-Max scaling despite the project documentation claiming features were normalized.
- **Applied Fix:** Integrated `MinMaxScaler` in the Autoencoder pipeline before training and latent representation extraction.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_004_autoencoder_sigmoid_target_scaling`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_004_autoencoder_sigmoid_target_scaling -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**

---

### BUG-005: Hardcoded String Placeholders in Final Report
- **Bug ID:** `BUG-005`
- **Severity:** Medium
- **Affected File:** `main.py` (originally lines 201, 247–248, 255)
- **Original Problem:** The Final Report page contained unreplaced placeholder strings: `[number of rows]`, `[number of columns]`, `{silhouette_kmeans}`, and `{silhouette_dbscan}`.
- **Root Cause:** Strings were written as regular string literals rather than dynamic f-strings or format calls.
- **Applied Fix:** Converted to dynamic f-strings passing actual dataset dimensions (`len(data):,`) and calculated model metrics.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_005_no_unresolved_placeholders_in_code_or_report`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_005_no_unresolved_placeholders_in_code_or_report -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**

---

### BUG-006: Deprecated Internal TensorFlow Import
- **Bug ID:** `BUG-006`
- **Severity:** Medium
- **Affected File:** `main.py` (originally lines 13–14)
- **Original Problem:** Used `from tensorflow.python.keras.models import Model` instead of public API `from tensorflow.keras.models import Model`.
- **Root Cause:** Reliance on deprecated private submodules.
- **Applied Fix:** Updated imports to standard `from tensorflow.keras.models import Model` and `from tensorflow.keras.layers import Input, Dense`.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_006_no_deprecated_tensorflow_python_imports`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_006_no_deprecated_tensorflow_python_imports -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**

---

### BUG-007: Duplicate and Unused Imports
- **Bug ID:** `BUG-007`
- **Severity:** Low
- **Affected File:** `main.py` (lines 15–18)
- **Original Problem:** `silhouette_score` was imported twice. `PCA`, `TSNE`, `davies_bouldin_score`, and `calinski_harabasz_score` were imported but never used anywhere in the script.
- **Root Cause:** Incomplete cleanup of experimental imports during development.
- **Applied Fix:** Cleaned up unused imports and consolidated the `sklearn.metrics` import.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_007_no_redundant_imports`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_007_no_redundant_imports -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**

---

### BUG-008: `style.css` File Exists but Never Injected
- **Bug ID:** `BUG-008`
- **Severity:** Low
- **Affected File:** `style.css` / `main.py`
- **Original Problem:** `style.css` contains custom shadow and metric box styles, but `main.py` never opened or injected it into the Streamlit app.
- **Root Cause:** Missing CSS file reading and injection routine.
- **Applied Fix:** Added `load_custom_css("style.css")` which reads the stylesheet and injects it via `st.markdown("<style>...</style>", unsafe_allow_html=True)`.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_008_style_css_is_loaded`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_008_style_css_is_loaded -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**

---

### BUG-009: Autoencoder Trained Synchronously on Every Interaction
- **Bug ID:** `BUG-009`
- **Severity:** Medium
- **Affected File:** `main.py` (line 394)
- **Original Problem:** On every UI interaction, the Autoencoder trained from scratch for 50 epochs synchronously, causing a 15–30 second freeze on every click.
- **Root Cause:** Missing caching of trained neural network weights.
- **Applied Fix:** Added `@st.cache_resource` wrapper function `get_trained_autoencoder` and reduced epochs to 25.
- **Changed Files:** `main.py`
- **Regression Test:** `tests/test_regression.py::test_regression_bug_009_autoencoder_caching_and_deterministic_output`
- **Test Command:** `python -m pytest tests/test_regression.py::test_regression_bug_009_autoencoder_caching_and_deterministic_output -v`
- **Actual Test Result:** `PASSED`
- **Final Status:** **FIXED & VERIFIED**
