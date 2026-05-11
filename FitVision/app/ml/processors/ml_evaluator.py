"""
MLEvaluatorMixin — Hybrid ML + Heuristic pattern for exercise evaluators.
─────────────────────────────────────────────────────────────────────────

Loads a trained model and provides `predict_exercise()` for frame-level
classification. Evaluators inherit this alongside ExerciseEvaluator to
get ML predictions with automatic heuristic fallback.
"""
import os
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

# ─── Paths to saved model artifacts ───
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_MODEL_PATH = os.path.join(_MODELS_DIR, "workout_classifier.pkl")
_ENCODER_PATH = os.path.join(_MODELS_DIR, "label_encoder.pkl")
_FEATURE_COLS_PATH = os.path.join(_MODELS_DIR, "feature_columns.pkl")

# ─── Module-level cache (loaded once, shared across all evaluators) ───
_model_cache = {}


def _load_artifacts():
    """Load model artifacts once into module-level cache."""
    if "model" not in _model_cache:
        if not os.path.exists(_MODEL_PATH):
            logger.warning(f"Model not found at {_MODEL_PATH}. ML predictions disabled.")
            _model_cache["model"] = None
            _model_cache["encoder"] = None
            _model_cache["feature_cols"] = None
            return

        logger.info(f"Loading ML model from {_MODEL_PATH}")
        _model_cache["model"] = joblib.load(_MODEL_PATH)
        _model_cache["encoder"] = joblib.load(_ENCODER_PATH)
        _model_cache["feature_cols"] = joblib.load(_FEATURE_COLS_PATH)
        logger.info(f"Model loaded — classes: {list(_model_cache['encoder'].classes_)}")


class MLEvaluatorMixin:
    """
    Mixin that adds ML-based exercise classification to any ExerciseEvaluator.

    Usage (MRO: Mixin first, then ABC):
        class SmartSquatEvaluator(MLEvaluatorMixin, ExerciseEvaluator):
            ...

    Provides:
        predict_exercise(features_dict) -> (label, confidence)
        extract_distance_features(landmarks, engine) -> dict
    """

    # ─── MediaPipe landmark pairs matching xyz_distances.csv columns ───
    # Each entry: (joint_a_name, joint_b_name, landmark_idx_a, landmark_idx_b)
    _JOINT_PAIRS = [
        ("left_shoulder", "left_wrist", 11, 15),
        ("right_shoulder", "right_wrist", 12, 16),
        ("left_hip", "left_ankle", 23, 27),
        ("right_hip", "right_ankle", 24, 28),
        ("left_hip", "left_wrist", 23, 15),
        ("right_hip", "right_wrist", 24, 16),
        ("left_shoulder", "left_ankle", 11, 27),
        ("right_shoulder", "right_ankle", 12, 28),
        ("left_hip", "right_wrist", 23, 16),
        ("right_hip", "left_wrist", 24, 15),
        ("left_elbow", "right_elbow", 13, 14),
        ("left_knee", "right_knee", 25, 26),
        ("left_wrist", "right_wrist", 15, 16),
        ("left_ankle", "right_ankle", 27, 28),
    ]

    # Average-based features: (name_suffix, center_idx, point_a_idx, point_b_idx)
    _AVG_FEATURES = [
        ("left_hip_avg_left_wrist_left_ankle", 23, 15, 27),
        ("right_hip_avg_right_wrist_right_ankle", 24, 16, 28),
    ]

    def __init__(self):
        super().__init__()
        _load_artifacts()  # Ensure model is loaded

    def extract_distance_features(self, landmarks) -> dict:
        """
        Extract the same xyz distance features used during training.
        Returns a dict matching the feature_columns order.
        """
        features = {}

        # Pairwise xyz distances
        for name_a, name_b, idx_a, idx_b in self._JOINT_PAIRS:
            la = landmarks[idx_a]
            lb = landmarks[idx_b]
            prefix = f"{name_a}_{name_b}"
            features[f"x_{prefix}"] = la.x - lb.x
            features[f"y_{prefix}"] = la.y - lb.y
            features[f"z_{prefix}"] = la.z - lb.z

        # Average-based features (distance from center to midpoint of two joints)
        for name, center_idx, pa_idx, pb_idx in self._AVG_FEATURES:
            lc = landmarks[center_idx]
            la = landmarks[pa_idx]
            lb = landmarks[pb_idx]
            avg_x = (la.x + lb.x) / 2
            avg_y = (la.y + lb.y) / 2
            avg_z = (la.z + lb.z) / 2
            features[f"x_{name}"] = lc.x - avg_x
            features[f"y_{name}"] = lc.y - avg_y
            features[f"z_{name}"] = lc.z - avg_z

        return features

    def predict_exercise(self, landmarks) -> tuple:
        """
        Predict exercise class from landmarks using the ML model.

        Returns:
            (predicted_label: str, confidence: float)
            Returns ("unknown", 0.0) if model is not available.
        """
        model = _model_cache.get("model")
        encoder = _model_cache.get("encoder")
        feature_cols = _model_cache.get("feature_cols")

        if model is None or encoder is None:
            return "unknown", 0.0

        # Extract features in the correct column order
        feat_dict = self.extract_distance_features(landmarks)
        feature_vector = np.array([feat_dict.get(col, 0.0) for col in feature_cols]).reshape(1, -1)

        # Predict
        pred_encoded = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        confidence = float(probabilities.max())
        pred_label = encoder.inverse_transform([pred_encoded])[0]

        return pred_label, confidence
