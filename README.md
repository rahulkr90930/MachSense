# MachSense — Predictive Maintenance & RUL Intelligence

MachSense is an AI-powered predictive-maintenance platform combining a condition-aware temporal model, a continuous Health Index, probabilistic RUL, reconstruction-based anomaly evidence, lifecycle staging, and a maintenance signal.

## 🚀 Quickstart: Clone & Run

```bash
# 1. Clone the repository
git clone https://github.com/rahulkr90930/MachSense.git
cd MachSense

# 2. Create and activate a virtual environment (Python 3.11 recommended)
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the interactive Streamlit Dashboard or Notebooks
python run.py app           # Launch Streamlit Web UI
python run.py paderborn     # Run Primary Paderborn 17-Bearing Workflow
python run.py femto         # Run FEMTO / PRONOSTIA Bearing Benchmark
python run.py cmapss        # Run NASA C-MAPSS FD001 Turbofan Benchmark
```

## Dataset strategy

### 1. Paderborn — primary dataset

The Paderborn run-to-failure release contains 17 bearing experiments under time-varying speed and load. The complete official raw release is 152 GB. For the main experiment, MachSense uses the dataset authors' public extracted-feature repository so the project remains practical on a student PC. An optional cell downloads official raw B01 (~966 MB compressed) so you can demonstrate direct raw-data provenance without downloading the complete 152 GB archive.

### 2. FEMTO / PRONOSTIA — raw bearing benchmark

This is the independent bearing benchmark. The notebook downloads the raw archive and extracts a deliberately compact **two-feature representation per measurement: horizontal RMS and vertical RMS**. This keeps the raw-data pipeline lightweight. We can expand to richer vibration features later if anomaly detection needs them.

### 3. NASA C-MAPSS FD001 — optional cross-domain benchmark

C-MAPSS is a famous small turbofan-engine RUL benchmark. NASA describes the release as multivariate time series with 21 sensor measurements plus 3 operational settings; FD001 contains 100 training trajectories and 100 test trajectories under one operating condition and one fault mode. It is small enough for a laptop and gives MachSense a non-bearing stress test.

**The datasets are NOT concatenated.** Paderborn/FEMTO are bearing vibration systems; C-MAPSS is a simulated turbofan-engine system. Pooling their raw rows would mix incompatible sensor semantics and units. Instead, we train the same MachSense formulation separately and compare its behavior.

## 📓 Notebooks-First Architecture

Each dataset has its own **100% self-contained, interactive Jupyter Notebook**. Anyone working on a dataset can train models, tweak hyperparameters, modify neural network layers, adjust loss functions, and visualize results entirely inside that notebook without navigating or editing external `.py` files:

### [`01_Paderborn_End_to_End.ipynb`](file:///notebooks/01_Paderborn_End_to_End.ipynb)
- **Primary Bearing Workflow**: 17 run-to-failure bearing experiments under time-varying speed and load.
- **In-Notebook Implementation**: Inline temporal sequence dataset, Condition-FiLM TCN network, multi-task pinball quantile loss, training loop with epoch convergence, and 3-panel prognostic trajectory plots.

### [`02_FEMTO_End_to_End.ipynb`](file:///notebooks/02_FEMTO_End_to_End.ipynb)
- **Independent Bearing Benchmark**: Accelerated run-to-failure bearing testing under dynamic radial load (IEEE PHM 2012).
- **In-Notebook Implementation**: Dedicated train/validation splits, inline PyTorch model, complete training & anomaly calibration loop, and uncertainty degradation fan charts.

### [`03_CMAPSS_End_to_End.ipynb`](file:///notebooks/03_CMAPSS_End_to_End.ipynb)
- **Turbofan Degradation Benchmark**: NASA C-MAPSS FD001 (100 run-to-failure training engines, 100 test engines).
- **In-Notebook Implementation**: Piecewise linear RUL targets, inline PyTorch training loop, and **full benchmark evaluation on all 100 test engines** against ground-truth RUL (computing Test RMSE and NASA Scoring function $S$).

*(The `src/` directory serves as an underlying library powering the Streamlit web dashboard and automated smoke tests).*

## Why C-MAPSS is a benchmark, not a combined training source

The scientific value is **cross-domain validation**, not an artificial mega-dataset. FEMTO asks whether MachSense behaves on another bearing rig and raw vibration archive; C-MAPSS asks whether the high-level RUL/health formulation survives a different machine type and sensor modality.

## Running

Use Python 3.11.

```bash
pip install -r requirements.txt
```

Then open the notebook you want. The optional launcher is:

```bash
python run.py paderborn
python run.py benchmark
python run.py app
python run.py test
```

## Streamlit

The dashboard has three dataset choices:

- Paderborn
- PRONOSTIA / FEMTO
- NASA C-MAPSS FD001

Every choice loads a separate checkpoint. There is also an **Upload CSV** mode.

After every training run the notebook creates a real held-out input file:

```text
artifacts/check_samples/<dataset>_validation_input.csv
```

That CSV is the recommended way to test the trained model in Streamlit.

## Error and leakage checks

- disk-space checks before downloading
- download retries
- ZIP integrity test
- safe ZIP extraction against path traversal
- dataset schema validation
- missing/NaN/Inf detection
- duplicate run/time detection
- minimum independent-run checks
- bearing/engine-wise validation split
- train-only normalization statistics
- target/ID leakage checks
- checkpoint dimension/schema checks
- minimum sequence-window checks

## Sources

Paderborn: https://zenodo.org/records/10868257
Paderborn authors' features/code: https://github.com/alireza-javanmardi/bearing-RUL
NASA C-MAPSS portal: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
C-MAPSS public mirror used for file download: https://github.com/PunVas/nasa-c-mapss
FEMTO / PRONOSTIA public archive: https://phm-datasets.s3.amazonaws.com/NASA/10.%20FEMTO%20Bearing.zip
