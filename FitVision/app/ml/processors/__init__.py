"""
Exercise Evaluator Factory
──────────────────────────
Registers ML-enhanced ("smart") evaluators as default, with automatic
fallback to heuristic-only evaluators if the ML model is not available.
"""
from .squat_evaluator import SquatEvaluator
from .pushup_evaluator import PushupEvaluator
from .smart_squat_evaluator import SmartSquatEvaluator
from .smart_pushup_evaluator import SmartPushupEvaluator
from .generic_evaluator import GenericMLEvaluator

# ML-enhanced evaluators (preferred)
_SMART_REGISTRY = {
    "squat": SmartSquatEvaluator,
    "pushup": SmartPushupEvaluator,
    # Generic ML evaluators for exercises without dedicated heuristic logic
    "pull_up": lambda: GenericMLEvaluator("pull_up"),
    "situp": lambda: GenericMLEvaluator("situp"),
    "jumping_jack": lambda: GenericMLEvaluator("jumping_jack"),
}

# Pure heuristic evaluators (fallback — only squat & pushup have heuristic logic)
_HEURISTIC_REGISTRY = {
    "squat": SquatEvaluator,
    "pushup": PushupEvaluator,
}


def get_evaluator(exercise_type: str, use_ml: bool = True):
    """
    Factory function — returns a fresh evaluator instance.

    Args:
        exercise_type: "squat", "pushup", "pull_up", "situp", "jumping_jack"
        use_ml: If True, returns ML-enhanced evaluator. If False, pure heuristic.

    Returns:
        An evaluator instance or None if exercise type is unsupported.
    """
    key = exercise_type.lower()

    if use_ml:
        factory = _SMART_REGISTRY.get(key)
        if factory:
            try:
                # Smart registry entries can be classes or lambdas
                return factory() if callable(factory) else factory
            except Exception:
                pass

    cls = _HEURISTIC_REGISTRY.get(key)
    return cls() if cls else None
