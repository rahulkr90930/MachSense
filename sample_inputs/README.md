# MachSense check CSVs

After training, each notebook automatically creates a **real held-out validation CSV** under `artifacts/check_samples/`.

Those files are the recommended way to verify the trained model in Streamlit because they contain exactly the feature and condition columns saved in the checkpoint.

Synthetic demo CSVs in this folder are only schema examples; they are not benchmark data.
