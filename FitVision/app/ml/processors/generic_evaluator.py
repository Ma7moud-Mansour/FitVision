"""
Generic ML-only Evaluator for exercises without heuristic logic.
Used for pull_up, situp, jumping_jack — relies purely on ML classification.
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

    def evaluate(self, landmarks, w, h, engine):
        ml_label, ml_confidence = self.predict_exercise(landmarks)
        ml_info = {"ml_label": ml_label, "ml_confidence": round(ml_confidence, 3)}

        # Simple state-machine rep counting based on ML confidence oscillation
        # When confidence for the expected exercise drops and rises, it signals a rep cycle
        is_match = (ml_label == self.expected_exercise and ml_confidence > 0.5)

        if is_match and self._prev_confidence < 0.4:
            self.stage = "active"
            self.counter += 1
            self.feedback = f"✅ Rep #{self.counter} ({ml_confidence:.0%})"
        elif is_match:
            self.feedback = f"{self.expected_exercise.replace('_', ' ').title()} — {ml_confidence:.0%}"
        elif ml_confidence > 0.6:
            self.feedback = f"⚠️ Detected {ml_label} ({ml_confidence:.0%})"
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
