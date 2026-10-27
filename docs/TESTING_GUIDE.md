# 📖 Climate Change AI — Complete Testing Guide

A step-by-step practical guide for developers, QA engineers, and evaluators to set up, run, test, and troubleshoot the **Climate Change AI** repository on Windows, macOS, or Linux.

---

## 💻 1. Prerequisites

- **Python 3.9 to 3.11** installed. (Recommended: Python 3.11).
- **Git** (for version control).
- **VS Code** or another IDE (optional, but recommended).

---

## 🛠️ 2. Environment Setup

### 2.1 Clone or Navigate to the Repository
Open **PowerShell**, **Command Prompt**, or **Git Bash**:
```bash
cd f:/climate-AI
```

### 2.2 Create and Activate a Python Virtual Environment

- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If execution of scripts is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On Windows (Git Bash):**
  ```bash
  python -m venv venv
  source venv/Scripts/activate
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

## 📦 3. Installing Dependencies

### 3.1 Install Production Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Install Testing and Development Dependencies
```bash
pip install -r requirements-dev.txt
```

---

## 🚀 4. Running the Application

To launch the interactive dashboard:
```bash
streamlit run main.py
```
The browser will automatically open at:
`http://localhost:8501`

---

## 🧪 5. Running Automated Tests

### 5.1 Run the Full Test Suite
```bash
python -m pytest tests/ -v
```

### 5.2 Run Specific Test Categories
- **Test Core Imports & Dependencies:**
  ```bash
  python -m pytest tests/test_imports.py -v
  ```
- **Test Dataset Loading & Integrity:**
  ```bash
  python -m pytest tests/test_data_loading.py tests/test_data_validation.py -v
  ```
- **Test Preprocessing & Outlier Cleaning:**
  ```bash
  python -m pytest tests/test_preprocessing.py -v
  ```
- **Test Machine Learning & Autoencoder Models:**
  ```bash
  python -m pytest tests/test_models.py tests/test_model_prediction.py tests/test_model_evaluation.py -v
  ```
- **Test Visualizations & UI Integration:**
  ```bash
  python -m pytest tests/test_visualization.py tests/test_application.py -v
  ```

### 5.3 Run with Code Coverage Report (if `pytest-cov` installed)
```bash
python -m pytest tests/ --cov=main --cov-report=term-missing
```

---

## 🔍 6. Testing in VS Code

1. Open the project folder in VS Code:
   ```bash
   code f:/climate-AI
   ```
2. Select your Python interpreter:
   - Press `Ctrl + Shift + P`
   - Type `Python: Select Interpreter`
   - Choose `.\venv\Scripts\python.exe`
3. Open the **Testing Tab** (flask icon in the left sidebar):
   - Click **Configure Tests** $\to$ **pytest**.
   - Select the `tests` directory.
   - Run tests individually or click **Run All Tests**.

---

## ❓ 7. Troubleshooting Common Issues

### Issue 1: `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`
- **Cause:** Known cosmetic warning during Protobuf import in some TensorFlow 2.18 builds.
- **Solution:** This is harmless and does not affect model execution or test outcomes. Can be suppressed by setting environment variable:
  ```powershell
  $env:TF_ENABLE_ONEDNN_OPTS="0"
  ```

### Issue 2: Matplotlib GUI Window Conflicts During Tests
- **Cause:** Headless testing environments cannot open GUI windows.
- **Solution:** Our test suite automatically configures the non-interactive backend (`matplotlib.use("Agg")`) in [tests/test_visualization.py](file:///f:/climate-AI/tests/test_visualization.py).

### Issue 3: Streamlit ScriptRunContext Warning in Bare Python Mode
- **Cause:** When calling functions from `main.py` directly in unit tests without running `streamlit run`, Streamlit logs an informational warning.
- **Solution:** Harmless; tests pass without error.
