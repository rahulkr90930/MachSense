from __future__ import annotations


def anomaly_probability(reconstruction_error: float, threshold: float, scale: float) -> float:
    # Logistic conversion keeps the score in [0,1] while preserving a calibrated threshold.
    scale = max(scale, 1e-8)
    z = (reconstruction_error - threshold) / scale
    # stable-ish scalar sigmoid
    if z >= 0:
        e = 2.718281828 ** (-z)
        return 1.0 / (1.0 + e)
    e = 2.718281828 ** z
    return e / (1.0 + e)


def lifecycle_stage(health: float, anomaly: float) -> str:
    if health >= 0.78 and anomaly < 0.25:
        return "Infant"
    if health >= 0.42 and anomaly < 0.65:
        return "Useful Life"
    return "Wear-out"


def maintenance_action(health: float, rul_p10: float, rul_p50: float, anomaly: float) -> str:
    if anomaly >= 0.80 or rul_p10 <= 0.08 or health <= 0.20:
        return "URGENT INSPECTION / PLAN SHUTDOWN"
    if anomaly >= 0.55 or rul_p50 <= 0.20 or health <= 0.40:
        return "SCHEDULE MAINTENANCE SOON"
    if anomaly >= 0.30 or rul_p50 <= 0.40 or health <= 0.60:
        return "INCREASE MONITORING"
    return "CONTINUE NORMAL OPERATION"
