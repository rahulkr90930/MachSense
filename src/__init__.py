"""
MachSense Package Root — Exposes core engine routines and data pipelines.
"""

from .engine import (
    MachSenseNet,
    ConditionFiLM,
    TemporalBlock,
    SequenceDataset,
    build_targets,
    quantile_loss,
    machsense_loss,
    standardize,
    choose_numeric_features,
    condition_columns,
    audit_table,
    assert_no_leakage,
    check_checkpoint,
    predict_series,
    load_checkpoint,
    train_machsense,
    seed_everything,
    require_free_space,
    save_json,
    load_config,
    anomaly_probability,
    lifecycle_stage,
    maintenance_action,
)

from .data_pipeline import (
    build_paderborn_features,
    build_femto_features,
    build_cmapss_features,
    build_cmapss_fd001,
)
