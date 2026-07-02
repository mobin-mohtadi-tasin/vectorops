"""
VectorOps - Failure Prediction Model Training

Trains two Random Forest models on the synthetic telemetry dataset:
  1. RandomForestClassifier  -> P(bottleneck within 30 min)   [fail_risk_pct]
  2. RandomForestRegressor   -> predicted minutes to bottleneck

Both are saved as joblib artifacts consumed by app/failure_model.py at
inference time.
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
import joblib

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "telemetry_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(HERE), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["gpu_core_util_pct", "vram_util_pct", "temp_c", "power_draw_w"]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y_cls = df["will_fail_30min"]
    y_reg = df["minutes_to_bottleneck"]

    X_train, X_test, ycls_train, ycls_test, yreg_train, yreg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    clf.fit(X_train, ycls_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    print(f"[classifier] accuracy={accuracy_score(ycls_test, preds):.4f}  auc={roc_auc_score(ycls_test, probs):.4f}")

    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    reg.fit(X_train, yreg_train)
    reg_preds = reg.predict(X_test)
    print(f"[regressor]  MAE={mean_absolute_error(yreg_test, reg_preds):.2f} minutes")

    joblib.dump(clf, os.path.join(MODEL_DIR, "failure_classifier.joblib"))
    joblib.dump(reg, os.path.join(MODEL_DIR, "bottleneck_regressor.joblib"))
    print(f"Saved models to {MODEL_DIR}")

    print("\nFeature importances (classifier):")
    for f, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f:20s} {imp:.3f}")


if __name__ == "__main__":
    main()
