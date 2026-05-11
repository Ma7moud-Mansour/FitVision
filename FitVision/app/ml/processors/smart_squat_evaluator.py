"""
SmartSquatEvaluator — Hybrid ML + Heuristic Squat Analysis
──────────────────────────────────────────────────────────
Uses ML model to classify exercise type and confirm it's a squat,
then applies heuristic angle-based logic for rep counting and form feedback.
Falls back to pure heuristics if ML confidence is below threshold.
"""
import numpy as np
from .base_evaluator import ExerciseEvaluator
from .ml_evaluator import MLEvaluatorMixin

# Confidence threshold: below this, ML prediction is ignored
ML_CONFIDENCE_THRESHOLD = 0.6


class SmartSquatEvaluator(MLEvaluatorMixin, ExerciseEvaluator):
    """
    ML-Enhanced Squat Evaluator.

    Pipeline per frame:
        1. ML model classifies exercise type + confidence
        2. If confident squat → use ML-confirmed feedback
        3. Heuristic angle logic always runs for rep counting
        4. If ML says "not squat" with high confidence → flag as wrong exercise
    """

    def evaluate(self, landmarks, w, h, engine):
        # ─── ML Prediction (exercise classification) ───
        ml_label, ml_confidence = self.predict_exercise(landmarks)
        ml_info = {"ml_label": ml_label, "ml_confidence": round(ml_confidence, 3)}

        # ─── Heuristic: Angle-based rep counting (always runs) ───
        hip = [landmarks[engine.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[engine.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[engine.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[engine.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[engine.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[engine.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

        angle = engine.calculate_angle(hip, knee, ankle)

        # Rep counting logic (same as original SquatEvaluator)
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1

        # ─── Hybrid Feedback ───
        if ml_confidence >= ML_CONFIDENCE_THRESHOLD:
            if ml_label == "squat":
                if angle < 90 and self.stage == "down":
                    self.feedback = f"✅ ML-Confirmed Perfect Rep #{self.counter} ({ml_confidence:.0%})"
                elif angle > 90 and self.stage == "up":
                    self.feedback = "Go lower for better ROM!"
                else:
                    self.feedback = f"Squat detected ({ml_confidence:.0%})"
            else:
                # ML thinks it's a different exercise
                self.feedback = f"⚠️ Expected squat, detected {ml_label} ({ml_confidence:.0%})"
        else:
            # Low ML confidence — pure heuristic fallback
            if angle < 90 and self.stage == "down":
                self.feedback = f"Perfect! Rep: {self.counter}"
            elif angle > 90 and self.stage == "up":
                self.feedback = "Go lower for better ROM!"

        return {
            "angle": angle,
            "counter": self.counter,
            "feedback": self.feedback,
            **ml_info,
        }

    def get_ml_prediction(self, landmarks):
        """Expose ML prediction for the VideoProcessor UI overlay."""
        return self.predict_exercise(landmarks)
