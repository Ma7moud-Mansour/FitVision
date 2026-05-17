"""
Generic ML-only Evaluator for exercises without heuristic logic.
Used for pull_up, situp, jumping_jack — relies purely on ML classification.
Uses percentage-based wrong exercise detection from BaseEvaluator.
"""
from .base_evaluator import ExerciseEvaluator
from .ml_evaluator import MLEvaluatorMixin


class GenericMLEvaluator(MLEvaluatorMixin, ExerciseEvaluator):
    """
    ML-driven evaluator for any exercise type.
    Uses the trained model for classification and basic rep counting
    via state transitions detected by the ML model.
    """

    def __init__(self, expected_exercise: str):
        super().__init__()
        self.expected_exercise = expected_exercise
        self._prev_confidence = 0.0

    def evaluate(self, results, w, h, engine):
        landmarks = results.pose_landmarks.landmark
        world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else landmarks
        
        ml_label, ml_confidence = self.predict_exercise(world_landmarks, 1, 1)
        ml_info = {"ml_label": ml_label, "ml_confidence": round(ml_confidence, 3)}

        # ─── Wrong exercise tracking (percentage-based) ───
        self._update_wrong_exercise_tracking(
            ml_label, ml_confidence, self.expected_exercise
        )

        # Simple state-machine rep counting based on ML confidence oscillation
        # When confidence for the expected exercise drops and rises, it signals a rep cycle
        is_match = (ml_label == self.expected_exercise and ml_confidence > 0.5)

        if is_match and self._prev_confidence < 0.4:
            self.stage = "active"
            self.counter += 1
            self.feedback = f"✅ Rep #{self.counter} ({ml_confidence:.0%})"

            duration_ms = (self.current_timestamp_ms - self._current_rep_start_ms
                           if self._current_rep_start_ms else None)
            start_frame = (self._current_rep_start_frame
                           if self._current_rep_start_frame else None)

            self.rep_data.append({
                "rep_number": int(self.counter),
                "min_angle": None,
                "max_angle": None,
                "rom": None,
                "is_valid": True,
                "feedback": str(self.feedback),
                "form_score": float(round(ml_confidence, 2)),
                "start_frame": int(start_frame) if start_frame is not None else None,
                "end_frame": int(self.current_frame),
                "duration_ms": int(duration_ms) if duration_ms is not None else None,
            })
            self._current_rep_start_frame = self.current_frame
            self._current_rep_start_ms = self.current_timestamp_ms
        elif is_match:
            self.feedback = f"{self.expected_exercise.replace('_', ' ').title()} — {ml_confidence:.0%}"
        elif ml_confidence > 0.6:
            self.feedback = f"⚠️ Detected {ml_label} ({ml_confidence:.0%})"
            self._current_rep_start_frame = self.current_frame
            self._current_rep_start_ms = self.current_timestamp_ms
        else:
            self.feedback = "Analyzing..."

        self._prev_confidence = ml_confidence if is_match else 0.0

        return {
            "counter": self.counter,
            "feedback": self.feedback,
            **ml_info,
        }

    def get_ml_prediction(self, landmarks):
        return self.predict_exercise(landmarks)
