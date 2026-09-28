from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import load_checkpoint, predict_series, check_checkpoint

st.set_page_config(
    page_title="MachSense — Machine Health & Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    .badge-stage-normal {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-stage-degrading {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-stage-wearout {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

DATASETS = {
    "Paderborn (17 Bearings — Primary)": ("paderborn", ROOT / "data/processed/paderborn_features.parquet"),
    "NASA C-MAPSS FD001 (Turbofan RUL Benchmark)": ("cmapss", ROOT / "data/processed/cmapss_fd001_train.parquet"),
    "PRONOSTIA / FEMTO (Raw Bearing Benchmark)": ("femto", ROOT / "data/processed/femto_features.parquet")
}

# Sidebar Navigation & Selection
st.sidebar.title("⚙️ MachSense")
st.sidebar.caption("Multi-Task Neural Predictive Maintenance")

# Checkpoint availability scan
status_dict = {}
for label, (key, _) in DATASETS.items():
    ckpt_file = ROOT / "artifacts/checkpoints" / f"machsense_{key}.pt"
    status_dict[label] = ckpt_file.exists()

def format_dataset_label(label):
    return f"{label} {'[Ready]' if status_dict[label] else '[Untrained]'}"

dataset_options = list(DATASETS.keys())
selected_label = st.sidebar.selectbox(
    "Benchmark Dataset",
    dataset_options,
    format_func=format_dataset_label
)

key, data_path = DATASETS[selected_label]
ckpt_path = ROOT / "artifacts/checkpoints" / f"machsense_{key}.pt"

st.sidebar.markdown("---")
st.sidebar.subheader("Data Input Mode")
mode = st.sidebar.radio(
    "Source",
    ["Saved validation sample", "Processed dataset", "Upload custom CSV"]
)

df = None
try:
    if mode == "Saved validation sample":
        sample_file = ROOT / "artifacts/check_samples" / f"{key}_validation_input.csv"
        if not sample_file.exists():
            sample_file = ROOT / "sample_inputs" / f"{key}_demo_input.csv"
        if not sample_file.exists():
            st.sidebar.warning(f"Validation sample for {key} not found. Please upload a custom CSV or run training.")
        else:
            df = pd.read_csv(sample_file)
    elif mode == "Processed dataset":
        if not data_path.exists():
            st.sidebar.warning(f"Processed dataset not found at `{data_path.name}`. Run feature extraction or choose 'Saved validation sample'.")
        else:
            df = pd.read_parquet(data_path)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload model input CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("👆 Please upload a CSV file with sensor/operating features in the sidebar to begin inference, or choose 'Saved validation sample'.")
except Exception as exc:
    st.error(f"Failed to load dataset: {exc}")

st.title("⚙️ MachSense — Predictive Health & RUL Intelligence")
st.markdown("Unified physics-informed multi-task deep learning architecture evaluating machine health, remaining useful life quantiles, and anomaly detection.")

if not ckpt_path.exists():
    st.warning(f"### Model checkpoint for **{selected_label}** not found.")
    st.info(f"""
    To generate the trained checkpoint for **{key}**:
    - **Paderborn**: Run `notebooks/01_Paderborn_End_to_End.ipynb`
    - **FEMTO**: Run `notebooks/02_FEMTO_End_to_End.ipynb`
    - **NASA C-MAPSS**: Run `notebooks/03_CMAPSS_End_to_End.ipynb`
    
    *Alternatively, select an already trained dataset from the sidebar dropdown.*
    """)
    custom_ckpt = st.file_uploader("Or upload a custom PyTorch model checkpoint (.pt)", type=["pt"])
    if custom_ckpt:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path, "wb") as f:
            f.write(custom_ckpt.read())
        st.success("Uploaded checkpoint successfully! Reloading...")
        st.rerun()
    st.stop()

try:
    checkpoint = check_checkpoint(ckpt_path)
except Exception as e:
    st.error(f"Error loading model checkpoint: {e}")
    st.stop()

if df is None:
    st.info("👆 Please select or upload input data from the sidebar to view inference predictions.")
    st.stop()

if "bearing_id" not in df.columns:
    df["bearing_id"] = "run_01"

available_runs = sorted(df["bearing_id"].astype(str).unique())
selected_run = st.sidebar.selectbox("Machine / Bearing Unit", available_runs)

run_df = df[df["bearing_id"].astype(str) == str(selected_run)].copy()

# Run predictions
with st.spinner("Executing neural inference across temporal trajectory..."):
    try:
        pred = predict_series(run_df, checkpoint)
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        st.stop()

if pred.empty:
    st.warning(f"No predictions generated. The selected sequence has {len(run_df)} observations; at least 1 ordered observation is required.")
    st.stop()

latest = pred.iloc[-1]

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Health Index</div>
        <div class="metric-value">{latest.health:.1f}%</div>
        <div class="metric-sub">Normalized system vitality</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    native_val = f"{latest.rul_p50_native:.1f}" if "rul_p50_native" in latest else "N/A"
    unit_name = "cycles" if "cmapss" in key.lower() else "steps"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Median RUL (P50)</div>
        <div class="metric-value">{latest.rul_p50:.1f}%</div>
        <div class="metric-sub">{native_val} native {unit_name}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    risk_color = "#ef4444" if latest.anomaly > 50 else ("#f59e0b" if latest.anomaly > 20 else "#10b981")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Anomaly Risk</div>
        <div class="metric-value" style="color: {risk_color};">{latest.anomaly:.1f}%</div>
        <div class="metric-sub">Autoencoder reconstruction err</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    stage_class = "badge-stage-normal"
    if "degrad" in str(latest.stage).lower():
        stage_class = "badge-stage-degrading"
    elif "wear" in str(latest.stage).lower() or "critical" in str(latest.stage).lower():
        stage_class = "badge-stage-wearout"
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Lifecycle State</div>
        <div class="metric-value" style="font-size: 1.35rem; margin-top: 4px;">
            <span class="{stage_class}">{latest.stage}</span>
        </div>
        <div class="metric-sub">Operational phase assessment</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Recommended Maintenance Action Banner
action_str = str(latest.action).upper()
if "SHUTDOWN" in action_str or "URGENT" in action_str:
    st.error(f"🚨 **MAINTENANCE RECOMMENDATION:** {latest.action}")
elif "INSPECT" in action_str or "MONITOR" in action_str or "PLAN" in action_str:
    st.warning(f"⚠️ **MAINTENANCE RECOMMENDATION:** {latest.action}")
else:
    st.success(f"[OK] **MAINTENANCE RECOMMENDATION:** {latest.action}")

# Tabs for visual inspection
tab_traj, tab_data, tab_model = st.tabs([
    "📈 Trajectory Analytics",
    "📋 Detailed Observations Feed",
    "🧠 Model Diagnostics & Spec"
])

with tab_traj:
    st.subheader("Degradation & RUL Forecast Trajectory")
    st.caption("Normalized scores over the observed operating history.")
    
    chart_data = pred.set_index("time_idx")[["health", "rul_p50", "anomaly"]]
    chart_data.columns = ["Health Index (%)", "Median RUL (%)", "Anomaly Risk (%)"]
    st.line_chart(chart_data, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("RUL Prediction Uncertainty Bounds (P10 - P50 - P90)")
        rul_bounds = pred.set_index("time_idx")[["rul_p10", "rul_p50", "rul_p90"]]
        rul_bounds.columns = ["P10 (Pessimistic)", "P50 (Median)", "P90 (Optimistic)"]
        st.line_chart(rul_bounds, use_container_width=True)
        
    with col_b:
        st.caption("Native Remaining Useful Life (Physical Scale)")
        native_chart = pred.set_index("time_idx")[["rul_p50_native"]]
        native_chart.columns = [f"RUL ({unit_name})"]
        st.line_chart(native_chart, use_container_width=True)

with tab_data:
    st.subheader("Temporal Inference Records")
    st.dataframe(
        pred.tail(50).style.format({
            "health": "{:.2f}%",
            "rul_p10": "{:.2f}%",
            "rul_p50": "{:.2f}%",
            "rul_p90": "{:.2f}%",
            "anomaly": "{:.2f}%",
            "rul_p50_native": "{:.1f}"
        }),
        use_container_width=True
    )
    
    csv_bytes = pred.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_bytes,
        file_name=f"machsense_predictions_{key}_{selected_run}.csv",
        mime="text/csv"
    )

with tab_model:
    st.subheader("Model Specifications & Architecture")
    mcol1, mcol2 = st.columns(2)
    
    with mcol1:
        st.markdown("**Configuration**")
        st.json({
            "Dataset Domain": key,
            "Trained Validation Bearing": checkpoint.get("validation_bearing"),
            "Sliding Window Size": checkpoint["config"].get("window"),
            "Encoder Layers": checkpoint["config"].get("layers"),
            "Hidden Dimensions": checkpoint["config"].get("hidden_dim"),
            "Condition Embedding Dim": checkpoint["config"].get("condition_dim"),
            "Anomaly Detection Threshold": checkpoint.get("anomaly_threshold"),
            "RUL Scale Factor": checkpoint.get("rul_scale")
        })
        
    with mcol2:
        st.markdown(f"**Input Sensors & Features ({len(checkpoint['features'])})**")
        st.write(checkpoint["features"])
        st.markdown(f"**Operating Conditions ({len(checkpoint['condition_cols'])})**")
        st.write(checkpoint["condition_cols"])
