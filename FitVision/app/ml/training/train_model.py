"""
FitVision ML Training Pipeline
──────────────────────────────
Merges xyz_distances.csv and labels.csv, trains a classifier, and saves the model.

Usage:
    python -m app.ml.training.train_model
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ───
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Dataset")
XYZ_PATH = os.path.join(DATASET_DIR, "xyz_distances.csv")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.csv")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "workout_classifier.pkl")
ENCODER_PATH = os.path.join(MODEL_OUTPUT_DIR, "label_encoder.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_OUTPUT_DIR, "feature_columns.pkl")


def load_and_merge():
    """Load xyz_distances and labels, merge on vid_id."""
    logger.info(f"Loading xyz_distances from {XYZ_PATH}")
    xyz = pd.read_csv(XYZ_PATH)
    logger.info(f"  Shape: {xyz.shape} | Unique vid_ids: {xyz['vid_id'].nunique()}")

    logger.info(f"Loading labels from {LABELS_PATH}")
    labels = pd.read_csv(LABELS_PATH)
    logger.info(f"  Shape: {labels.shape}")
    logger.info(f"  Class distribution:\n{labels['class'].value_counts().to_string()}")

    # Merge on vid_id — each frame gets its video's label
    merged = xyz.merge(labels, on="vid_id", how="inner")
    logger.info(f"Merged shape: {merged.shape}")

    # Separate features and target
    id_cols = ["vid_id", "frame_order"]
    feature_cols = [c for c in merged.columns if c not in id_cols + ["class"]]
    X = merged[feature_cols].values
    y = merged["class"].values

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Target classes: {np.unique(y)}")
    logger.info(f"NaN check: {np.isnan(X).sum()} NaN values")

    return X, y, feature_cols


def train_and_evaluate(X, y, feature_cols):
    """Train GradientBoosting, evaluate, and save model + artifacts."""

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 80/20 split — stratified to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── RandomForest — fast (parallel training), interpretable, great for tabular data
    # n_jobs=-1 uses all CPU cores for parallel training
    logger.info("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation
    y_pred = clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    report = classification_report(y_test_labels, y_pred_labels)
    logger.info("\n" + report)

    logger.info("CONFUSION MATRIX")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    logger.info("\n" + cm_df.to_string())

    # ── Cross-validation score
    cv_scores = cross_val_score(clf, X, y_encoded, cv=5, scoring="accuracy")
    logger.info(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ── Feature importance (top 10)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    logger.info("\nTop 10 Features:")
    for i, idx in enumerate(top_idx):
        logger.info(f"  {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

    # ── Save artifacts
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    joblib.dump(clf, MODEL_PATH)
    logger.info(f"\nModel saved: {MODEL_PATH}")

    joblib.dump(le, ENCODER_PATH)
    logger.info(f"Label encoder saved: {ENCODER_PATH}")

    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    logger.info(f"Feature columns saved: {FEATURE_COLS_PATH}")

    return clf, le, feature_cols


if __name__ == "__main__":
    X, y, feature_cols = load_and_merge()
    clf, le, feature_cols = train_and_evaluate(X, y, feature_cols)
    logger.info("\nTraining pipeline complete!")
