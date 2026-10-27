# 🌍 Climate Change Analysis & Machine Learning Dashboard

A comprehensive engineering project for analyzing and exploring global climate change data using **Machine Learning** and **Deep Learning** techniques. The project features an interactive dashboard built with **Streamlit** and is supported by a comprehensive automated testing suite using **pytest**.

---

## 📌 Project Overview

This project aims to investigate and analyze major environmental and climate indicators, including temperature, carbon dioxide (`CO2`) emissions, precipitation, humidity, and wind speed, using observations collected from monitoring stations across **243 countries over more than 20 years**.

The application enables users to:

- Explore climate patterns and trends across time and geographical locations.
- Identify and group climate regions and monitoring stations using clustering algorithms (**K-Means** and **DBSCAN**).
- Reduce dimensionality and visualize complex multidimensional data in a two-dimensional space using a **Deep Learning Autoencoder**.

---

## 🧹 Data Preprocessing & Cleaning

The dataset was comprehensively processed, cleaned, and prepared using **Pandas** and **NumPy**, following these steps:

### 1. DateTime Parsing & Sorting

- Converted the `Date` column into a standardized datetime format using `pd.to_datetime()`.
- Extracted the `Year` column to facilitate temporal filtering and annual statistical analysis.
- Sorted records chronologically in ascending order using `sort_values` to ensure accurate time-series analysis.

### 2. Handling Missing Values

- Inspected records and validated data integrity.
- Removed records containing invalid dates using `dropna(subset=['Date'])`.

### 3. Outlier Detection & Removal

- Applied the statistical **Z-Score** method using NumPy arrays and SciPy to calculate each observation's deviation from the mean.
- Filtered and removed extreme values exceeding the `±3σ` threshold across all numerical features:
  - Temperature
  - CO2 Emissions
  - Precipitation
  - Humidity
  - Wind Speed
- Successfully removed **57 outliers**, resulting in a cleaned dataset containing **9,943 records**.

### 4. Feature Scaling for Deep Learning

- Applied `MinMaxScaler` to normalize the five numerical features to the `[0, 1]` range before feeding them into the Autoencoder.
- This ensures mathematical compatibility with the `sigmoid` activation function used in the decoder.

---

## ✨ Key Features

### 1. Home Dashboard

- An advanced sidebar for filtering climate data by country/location, climate variable, and year range.
- An interactive data table displaying filtered observations in real time.

### 2. Advanced Data Visualizations (Graphs)

- **Dual-Axis Chart:** An interactive time-series visualization illustrating the relationship between temperature and `CO2` emissions.
- **Global Choropleth Map:** A geographical heat map showing precipitation distribution across countries worldwide.
- **Top 10 Emitters:** A bar chart displaying the ten countries with the highest carbon dioxide emissions.

### 3. Exploratory Data Analysis (EDA)

- Comprehensive descriptive statistics for all numerical variables using `describe()`.
- Distribution histograms and Kernel Density Estimation (**KDE**) plots.
- A correlation heatmap illustrating relationships between environmental indicators.
- Boxplots for examining statistical distributions, dispersion, and potential outliers.

### 4. Clustering Analysis & AI Models

**K-Means Clustering**

- Dynamic clustering with an interactive slider for selecting the number of clusters (`k`).
- Cluster quality evaluation using the **Silhouette Score**.

**DBSCAN Clustering**

- Density-based clustering supported by the **Knee Method**, using `k`-Nearest Neighbors (k-NN) to help determine an appropriate epsilon (`ϵ`) distance threshold.
- Handles noise points and outliers during clustering analysis.

**Deep Learning Autoencoder**

- A neural network built using **TensorFlow / Keras**.
- Compresses five-dimensional climate data into a **two-dimensional latent space (2D Latent Space)**.
- Provides interactive visualizations of the compressed representations.

**Optimized Performance**

- Uses `@st.cache_resource` to cache trained neural network models and prevent computationally expensive retraining whenever dashboard filters are changed.

### 5. Interactive Final Report

- A comprehensive summary that dynamically displays actual dataset statistics, evaluation metrics, and model results.
- Automatically integrates calculated performance indicators and real-time data values.

---

## 📁 Project Structure

```text
climate-AI/
│
├── climate_change_data.csv       # Original climate dataset (10,000 records)
├── main.py                       # Main application and modular Streamlit interface
├── style.css                     # Custom styles integrated into the application
├── requirements.txt              # Core application dependencies
├── requirements-dev.txt          # Development and automated testing dependencies
├── .gitignore                    # Excludes caches, virtual environments, and logs
├── README.md                     # Main project documentation
│
├── docs/                         # Technical documentation and reports
│   ├── BUG_REPORT.md             # Documented bugs and remediation history
│   ├── TEST_REPORT.md            # Comprehensive testing and validation report
│   └── TESTING_GUIDE.md          # Detailed instructions for running tests
│
└── tests/                        # Automated test suite (47 passing tests)
    ├── conftest.py               # Test fixtures and sample datasets
    ├── test_imports.py           # Dependency and import validation
    ├── test_data_loading.py      # Data loading and chronological sorting tests
    ├── test_data_validation.py   # Dataset schema and physical bounds validation
    ├── test_preprocessing.py     # Outlier removal and Z-score validation
    ├── test_models.py            # K-Means, DBSCAN, and Autoencoder tests
    ├── test_model_loading.py     # Model persistence and configuration tests
    ├── test_model_prediction.py  # Prediction and inference tests
    ├── test_model_evaluation.py  # Evaluation metrics (Silhouette, MSE)
    ├── test_regression.py        # Regression tests for BUG-001 through BUG-009
    ├── test_visualization.py     # Chart and map generation tests
    ├── test_application.py       # Application CSS and filtering integration
    └── README.md                 # Test suite documentation
```

---

## 🛠️ Requirements

### Python Version

- **Python 3.9+**
- Tested and validated using **Python 3.11.5**

### Core Dependencies (`requirements.txt`)

```text
streamlit>=1.25.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit-option-menu>=0.3.6
```

### Development & Testing Dependencies (`requirements-dev.txt`)

```text
pytest>=7.4.0
pytest-cov>=4.1.0
flake8>=6.1.0
```

---

## 🚀 Installation & Setup

Follow these steps to install and run the application locally.

### 1. Navigate to the Project Directory

Open a terminal in the project directory.

For the local Windows installation:

```bash
cd /f/climate-AI
```

This example uses Git Bash. Replace the path with your local project directory if necessary.

### 2. Create and Activate a Virtual Environment (Recommended)

**Windows — PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows — Git Bash:**

```bash
python -m venv venv
source venv/Scripts/activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

Install the core application dependencies:

```bash
pip install -r requirements.txt
```

Optionally, install development and testing tools:

```bash
pip install -r requirements-dev.txt
```

### 4. Run the Streamlit Application

Start the application using:

```bash
streamlit run main.py
```

Once the application starts successfully, it should automatically open in your default web browser.

Alternatively, open the following URL:

**http://localhost:8501**

---

## 🧪 Running Automated Tests

The project includes a comprehensive automated testing suite containing **47 tests**, covering data processing, AI models, application functionality, and regression validation.

### Run All Tests

Execute the following command from the project root:

```bash
python -m pytest tests/ -v
```

### Test Results

The reported automated test execution completed successfully:

```text
======================== 47 passed in 6.92s ========================
```

All 47 automated test cases passed successfully during the reported validation run.

### Additional Testing Documentation

For more information about running tests in VS Code, examining test coverage, and reviewing technical validation results, refer to the following files:

- **[Testing Guide](docs/TESTING_GUIDE.md)** — Step-by-step instructions for running automated tests and troubleshooting common issues.
- **[Test Report](docs/TEST_REPORT.md)** — Comprehensive technical testing and project validation results.
- **[Bug Report](docs/BUG_REPORT.md)** — Documented defects, implemented fixes, and regression verification history.

---
