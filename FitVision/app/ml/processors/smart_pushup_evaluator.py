"""
SmartPushupEvaluator — Hybrid ML + Heuristic Pushup Analysis
"""
import numpy as np
from .base_evaluator import ExerciseEvaluator
from .ml_evaluator import MLEvaluatorMixin

ML_CONFIDENCE_THRESHOLD = 0.6


class SmartPushupEvaluator(MLEvaluatorMixin, ExerciseEvaluator):
    """
    ML-Enhanced Pushup Evaluator.
    ML classifies exercise type; heuristics handle rep counting + alignment check.
    """

    def evaluate(self, landmarks, w, h, engine):
        # ─── ML Prediction ───
        ml_label, ml_confidence = self.predict_exercise(landmarks)
        ml_info = {"ml_label": ml_label, "ml_confidence": round(ml_confidence, 3)}

        # ─── Heuristic: Angle + alignment ───
        shoulder = [landmarks[engine.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[engine.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        elbow = [landmarks[engine.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                 landmarks[engine.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        wrist = [landmarks[engine.mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                 landmarks[engine.mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

        angle = engine.calculate_angle(shoulder, elbow, wrist)

        # Alignment check (hip vs shoulder Y)
        hip_y = landmarks[engine.mp_pose.PoseLandmark.LEFT_HIP.value].y
        shoulder_y = landmarks[engine.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        alignment_error = abs(hip_y - shoulder_y)

        # Rep counting
        if angle < 70:
            self.stage = "down"
        if angle > 160 and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        # ─── Hybrid Feedback ───
        form_ok = alignment_error <= 0.15

        if ml_confidence >= ML_CONFIDENCE_THRESHOLD:
            if ml_label == "push_up":
                if self.stage == "up" and angle > 160:
                    form_msg = "Good Form" if form_ok else "Keep your back straight!"
                    self.feedback = f"✅ Rep #{self.counter} — {form_msg} ({ml_confidence:.0%})"
                else:
                    self.feedback = f"Pushup detected ({ml_confidence:.0%})"
            else:
                self.feedback = f"⚠️ Expected pushup, detected {ml_label} ({ml_confidence:.0%})"
        else:
            self.feedback = "Good Form" if form_ok else "Keep your back straight!"

        return {
            "angle": angle,
            "counter": self.counter,
            "feedback": self.feedback,
            **ml_info,
        }

    def get_ml_prediction(self, landmarks):
        return self.predict_exercise(landmarks)
