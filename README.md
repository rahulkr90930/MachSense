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
python run.py benchmark     # Run FEMTO & C-MAPSS Benchmarks
```

## Dataset strategy

### 1. Paderborn — primary dataset

The Paderborn run-to-failure release contains 17 bearing experiments under time-varying speed and load. The complete official raw release is 152 GB. For the main experiment, MachSense uses the dataset authors' public extracted-feature repository so the project remains practical on a student PC. An optional cell downloads official raw B01 (~966 MB compressed) so you can demonstrate direct raw-data provenance without downloading the complete 152 GB archive.

### 2. FEMTO / PRONOSTIA — raw bearing benchmark

This is the independent bearing benchmark. The notebook downloads the raw archive and extracts a deliberately compact **two-feature representation per measurement: horizontal RMS and vertical RMS**. This keeps the raw-data pipeline lightweight. We can expand to richer vibration features later if anomaly detection needs them.

### 3. NASA C-MAPSS FD001 — optional cross-domain benchmark

C-MAPSS is a famous small turbofan-engine RUL benchmark. NASA describes the release as multivariate time series with 21 sensor measurements plus 3 operational settings; FD001 contains 100 training trajectories and 100 test trajectories under one operating condition and one fault mode. It is small enough for a laptop and gives MachSense a non-bearing stress test.

**The datasets are NOT concatenated.** Paderborn/FEMTO are bearing vibration systems; C-MAPSS is a simulated turbofan-engine system. Pooling their raw rows would mix incompatible sensor semantics and units. Instead, we train the same MachSense formulation separately and compare its behavior.

## Exactly two notebooks

### `01_Paderborn_End_to_End.ipynb`

One notebook from start to finish:

`download → preprocess → audit → train → evaluate → export check CSV`

It also contains the optional raw B01 audit.

### `02_Benchmark_End_to_End.ipynb`

One benchmark notebook with a single switch:

```python
BENCHMARK = "femto"
# or
BENCHMARK = "cmapss"
```

It performs the complete workflow for the selected benchmark.

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
