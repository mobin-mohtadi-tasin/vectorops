"""
VectorOps - Failure Prediction (inference)

Loads the trained Random Forest classifier + regressor and scores live
node telemetry, attaching fail_risk_pct and minutes_to_bottleneck to each
NodeTelemetry object. Falls back to a simple heuristic if models haven't
been trained yet (e.g. fresh clone, before running data/train_model.py).
"""
import os
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
from typing import List
from app.models import NodeTelemetry

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(os.path.dirname(HERE), "models")
CLF_PATH = os.path.join(MODEL_DIR, "failure_classifier.joblib")
REG_PATH = os.path.join(MODEL_DIR, "bottleneck_regressor.joblib")

FEATURES = ["gpu_core_util_pct", "vram_util_pct", "temp_c", "power_draw_w"]

_clf = None
_reg = None
_models_loaded = False


def _try_load():
    global _clf, _reg, _models_loaded
    if _models_loaded:
        return
    if os.path.exists(CLF_PATH) and os.path.exists(REG_PATH):
        _clf = joblib.load(CLF_PATH)
        _reg = joblib.load(REG_PATH)
    _models_loaded = True


def _heuristic_risk(node: NodeTelemetry):
    """Fallback used only if the RF models haven't been trained yet."""
    risk = 0.0
    risk += max(0, node.temp_c - 70) * 2.2
    risk += max(0, node.gpu_core_util_pct - 70) * 1.1
    risk += max(0, node.vram_util_pct - 70) * 1.3
    risk = min(99.0, risk)
    minutes = max(0.0, 60 - risk * 0.55)
    return round(risk, 1), round(minutes, 1)


def score_nodes(nodes: List[NodeTelemetry]) -> List[NodeTelemetry]:
    if not nodes:
        return []
    _try_load()

    if _clf is None or _reg is None:
        for n in nodes:
            n.fail_risk_pct, n.minutes_to_bottleneck = _heuristic_risk(n)
        return nodes

    df = pd.DataFrame([{
        "gpu_core_util_pct": n.gpu_core_util_pct,
        "vram_util_pct": n.vram_util_pct,
        "temp_c": n.temp_c,
        "power_draw_w": n.power_draw_w,
    } for n in nodes])

    risk_probs = _clf.predict_proba(df[FEATURES])[:, 1] * 100
    minutes_preds = _reg.predict(df[FEATURES])

    for n, risk, minutes in zip(nodes, risk_probs, minutes_preds):
        n.fail_risk_pct = round(float(risk), 1)
        n.minutes_to_bottleneck = round(float(max(0, minutes)), 1)

    return nodes


def models_ready() -> bool:
    _try_load()
    return _clf is not None and _reg is not None
