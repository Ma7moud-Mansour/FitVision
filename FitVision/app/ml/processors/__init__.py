from .squat_evaluator import SquatEvaluator
from .pushup_evaluator import PushupEvaluator

_REGISTRY = {
    "squat": SquatEvaluator,
    "pushup": PushupEvaluator,
}

def get_evaluator(exercise_type: str):
    """Factory function — returns a fresh evaluator instance for the given exercise type."""
    cls = _REGISTRY.get(exercise_type.lower())
    return cls() if cls else None
