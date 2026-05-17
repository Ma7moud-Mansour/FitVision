"""
FitVision — Multi-Model Comparison Training Pipeline
═════════════════════════════════════════════════════
Trains 3 models from 2 model families, evaluates with advanced metrics,
exports confusion matrices, a unified comparison table, and saves the
best performer for production use.

Models:
  1. LogisticRegression   (Linear family — Baseline)
  2. RandomForestClassifier (Tree family — Current winner)
  3. XGBClassifier          (Boosted-tree family — Alternative)

Usage:
    python -m app.ml.training.train_models_comparison
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ── Optional: XGBoost (graceful fallback to LinearSVC if unavailable) ──
try:
    from xgboost import XGBClassifier
    _USE_XGBOOST = True
except ImportError:
    from sklearn.svm import LinearSVC
    _USE_XGBOOST = False

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ───
_BASE_DIR      = os.path.dirname(__file__)
DATASET_DIR    = os.path.join(_BASE_DIR, "..", "..", "..", "..", "Dataset")
XYZ_PATH       = os.path.join(DATASET_DIR, "xyz_distances.csv")
LABELS_PATH    = os.path.join(DATASET_DIR, "labels.csv")

MODEL_OUTPUT_DIR   = os.path.join(_BASE_DIR, "..", "models")
ARTIFACTS_DIR      = os.path.join(_BASE_DIR, "artifacts")

MODEL_PATH         = os.path.join(MODEL_OUTPUT_DIR, "workout_classifier.pkl")
SCALER_PATH        = os.path.join(MODEL_OUTPUT_DIR, "scaler.pkl")
ENCODER_PATH       = os.path.join(MODEL_OUTPUT_DIR, "label_encoder.pkl")
FEATURE_COLS_PATH  = os.path.join(MODEL_OUTPUT_DIR, "feature_columns.pkl")

# ─── Fixed seed for reproducibility ───
RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════
# STEP 1A — Data Loading & Merging
# ═══════════════════════════════════════════════════════════════════════
def load_and_merge() -> tuple:
    """
    Load xyz_distances.csv and labels.csv, merge on ``vid_id``.

    Returns
    -------
    X : np.ndarray          — feature matrix (n_samples, n_features)
    y : np.ndarray          — string class labels
    groups : np.ndarray     — video IDs for grouping
    feature_cols : list[str] — ordered column names
    """
    logger.info(f"Loading xyz_distances from {XYZ_PATH}")
    xyz = pd.read_csv(XYZ_PATH)
    logger.info(f"  Shape: {xyz.shape} | Unique vid_ids: {xyz['vid_id'].nunique()}")

    logger.info(f"Loading labels from {LABELS_PATH}")
    labels = pd.read_csv(LABELS_PATH)
    logger.info(f"  Shape: {labels.shape}")
    logger.info(f"  Class distribution:\n{labels['class'].value_counts().to_string()}")

    # Merge on vid_id — each frame inherits its video's label
    merged = xyz.merge(labels, on="vid_id", how="inner")
    logger.info(f"Merged shape: {merged.shape}")

    # Identify feature columns (everything except IDs and target)
    id_cols = ["vid_id", "frame_order"]
    feature_cols = [c for c in merged.columns if c not in id_cols + ["class"]]

    X = merged[feature_cols].values.astype(np.float64)
    y = merged["class"].values
    groups = merged["vid_id"].values

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Target classes: {np.unique(y)}")
    logger.info(f"NaN count: {np.isnan(X).sum()}")

    return X, y, groups, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# STEP 1B — Build Model Registry
# ═══════════════════════════════════════════════════════════════════════
def _build_models() -> dict:
    """Return an ordered dict of {name: estimator} for comparison."""
    models = {
        "LogisticRegression (Baseline)": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if _USE_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    else:
        logger.info("XGBoost not installed — falling back to LinearSVC")
        models["LinearSVC"] = LinearSVC(
            random_state=RANDOM_STATE,
            max_iter=2000,
        )

    return models


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Training + Evaluation
# ═══════════════════════════════════════════════════════════════════════
def _save_confusion_matrix(
    y_true, y_pred, class_names, model_name: str, out_dir: str
):
    """Save a confusion-matrix heatmap as PNG and a text log."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # ── Text log
    txt_path = os.path.join(out_dir, f"cm_{model_name}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Confusion Matrix — {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(cm_df.to_string() + "\n")
    logger.info(f"  Saved text CM → {txt_path}")

    # ── Visual heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"cm_{model_name}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved heatmap  → {png_path}")


def train_and_evaluate(X, y, groups, feature_cols):
    """
    Full multi-model pipeline:
      1. Encode labels
      2. Video-level stratified split (StratifiedGroupKFold) to prevent leakage
      3. StandardScaler — fit on train only (no leakage)
      4. Train each model, collect advanced metrics & 5-fold CV
      5. Export confusion matrices + comparison table
      6. Save the best model for production
    """
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. Encode labels ──
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── 2. Train / Test split — Video-Level (No Leakage) ──
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(cv.split(X, y_encoded, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    logger.info(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── 3. Robust preprocessing — no leakage ──
    from sklearn.preprocessing import StandardScaler, Normalizer
    from sklearn.pipeline import make_pipeline
    
    # L2 Normalization makes features scale-invariant (fixes video resolution mismatch)
    scaler = make_pipeline(Normalizer(norm='l2'), StandardScaler())
    X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
    X_test_scaled  = scaler.transform(X_test)         # transform test
    logger.info("Normalizer + StandardScaler fitted on training set only (no leakage)")

    # ── 4. Model training & evaluation ──
    models = _build_models()
    results = []  # list of dicts for the comparison DataFrame

    trained_models = {}  # name → fitted estimator

    for name, estimator in models.items():
        logger.info("\n" + "═" * 60)
        logger.info(f"Training: {name}")
        logger.info("═" * 60)

        estimator.fit(X_train_scaled, y_train)
        trained_models[name] = estimator

        y_pred = estimator.predict(X_test_scaled)

        # String labels for readable reports
        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)

        # Full classification report
        report = classification_report(y_test_labels, y_pred_labels)
        logger.info(f"\nClassification Report — {name}\n{report}")

        # Save report text
        report_path = os.path.join(ARTIFACTS_DIR, f"report_{name.replace(' ', '_')}.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report — {name}\n")
            f.write("=" * 60 + "\n")
            f.write(report + "\n")

        # Confusion matrix (text + PNG)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        _save_confusion_matrix(
            y_test_labels, y_pred_labels, class_names, safe_name, ARTIFACTS_DIR
        )

        # Advanced metrics
        acc   = accuracy_score(y_test, y_pred)
        p_mac = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_mac = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        p_wt  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        r_wt  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ── Cross-Validation to evaluate generalization ──
        logger.info(f"Running 5-Fold StratifiedGroupKFold CV for {name}...")
        pipeline = make_pipeline(StandardScaler(), estimator)
        cv_scores = cross_val_score(pipeline, X, y_encoded, groups=groups, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_f1_macro = cv_scores.mean()
        logger.info(f"CV Macro F1: {cv_f1_macro:.4f} (+/- {cv_scores.std():.4f})")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro Precision": round(p_mac, 4),
            "Macro Recall": round(r_mac, 4),
            "Macro F1-Score": round(f1_mac, 4),
            "CV Macro F1": round(cv_f1_macro, 4),
            "Weighted Precision": round(p_wt, 4),
            "Weighted Recall": round(r_wt, 4),
            "Weighted F1-Score": round(f1_wt, 4),
        })

    # ── 5. Unified comparison table ──
    comparison_df = pd.DataFrame(results)
    md_table = comparison_df.to_markdown(index=False)

    logger.info("\n" + "═" * 60)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("═" * 60)
    print("\n" + md_table + "\n")

    # Save to artifacts
    table_path = os.path.join(ARTIFACTS_DIR, "model_comparison.md")
    with open(table_path, "w") as f:
        f.write("# Model Comparison — FitVision Exercise Classifier\n\n")
        f.write(md_table + "\n")
    logger.info(f"Comparison table saved → {table_path}")

    # ── 6. Determine the winner and export for production ──
    best_idx = comparison_df["CV Macro F1"].idxmax()
    best_name = comparison_df.loc[best_idx, "Model"]
    best_f1   = comparison_df.loc[best_idx, "CV Macro F1"]

    logger.info("\n" + "═" * 60)
    logger.info(f"🏆  WINNER: {best_name}  (CV Macro F1 = {best_f1})")
    logger.info("═" * 60)

    best_estimator = trained_models[best_name]

    # Save production artifacts
    joblib.dump(best_estimator, MODEL_PATH)
    logger.info(f"Model saved   → {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved  → {SCALER_PATH}")

    joblib.dump(le, ENCODER_PATH)
    logger.info(f"Encoder saved → {ENCODER_PATH}")

    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    logger.info(f"Feature cols  → {FEATURE_COLS_PATH}")

    return best_estimator, le, scaler, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# Entry-point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    X, y, groups, feature_cols = load_and_merge()
    best_model, le, scaler, feature_cols = train_and_evaluate(X, y, groups, feature_cols)
    logger.info("\n✅ Multi-model comparison pipeline complete!")
