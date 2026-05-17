"""
BaseEvaluator — Abstract base class for all exercise evaluators.
─────────────────────────────────────────────────────────────────
Provides shared state management, helper methods for form scoring,
partial rep detection, and percentage-based wrong exercise detection.
"""
from abc import ABC, abstractmethod


# ─── Biomechanics Constants (reusable across evaluators) ───
SQUAT_IDEAL_ROM = 80.0      # Standing ~170° to bottom ~90° = ~80° ROM
PUSHUP_IDEAL_ROM = 100.0    # Extended ~170° to bottom ~70° = ~100° ROM


class ExerciseEvaluator(ABC):
    def __init__(self):
        self.counter = 0        # عداد العدات
        self.stage = None       # الحالة (نازل ولا طالع - Down/Up)
        self.feedback = ""      # رسالة لليوزر
        self.rep_data = []
        self.wrong_exercise_detected = False

        # ── Angle tracking per rep ──
        self._current_rep_min_angle = 180.0
        self._current_rep_max_angle = 0.0

        # ── Frame/timing tracking ──
        self.current_frame = 0
        self.current_timestamp_ms = 0
        self._current_rep_start_frame = 0
        self._current_rep_start_ms = 0
        self._start_frame_locked = False    # Prevents jitter overwrite

        # ── Descent tracking (for partial rep detection) ──
        self._descent_started = False
        self._descent_min_angle = 180.0

        # ── Wrong exercise detection (percentage-based) ──
        self._wrong_exercise_frame_count = 0
        self._total_analyzed_frames = 0
        self._WRONG_EXERCISE_RATIO_THRESHOLD = 0.30  # Flag if >30% frames disagree

    @abstractmethod
    def evaluate(self, results, w, h, engine):
        """اللوجيك الخاص بكل تمرينة هيتكتب هنا"""
        pass

    def get_ml_prediction(self, landmarks):
        """
        Override point for ML-enabled evaluators.
        Returns (label, confidence) or None if ML is not available.
        """
        return None

    # ─── Shared Helper Methods ───

    def _compute_form_score(self, actual_rom: float, ideal_rom: float,
                            penalty: float = 0.0) -> float:
        """
        ROM-deviation based form scoring.
        Returns a float 0.0 – 1.0.

        Args:
            actual_rom:  The range of motion achieved in this rep.
            ideal_rom:   The biomechanically ideal ROM for this exercise.
            penalty:     Additional penalty (0.0–0.5) for detected errors
                         like back rounding or knee valgus.
        """
        if ideal_rom <= 0:
            return 0.0
        rom_ratio = actual_rom / ideal_rom
        base_score = min(1.0, rom_ratio)   # Cap at 1.0 (exceeding ideal is fine)
        penalized = base_score * (1.0 - min(penalty, 0.5))
        return float(round(max(0.0, min(1.0, penalized)), 2))

    def _build_rep_entry(self, is_valid: bool, feedback: str,
                         form_score: float) -> dict:
        """
        Build a standardized rep_data dict with proper Python type casting.
        Prevents NumPy float64/int64 from reaching SQLAlchemy.
        """
        duration_ms = (self.current_timestamp_ms - self._current_rep_start_ms
                       if self._current_rep_start_ms else None)
        start_frame = (self._current_rep_start_frame
                       if self._current_rep_start_frame else None)

        return {
            "rep_number": int(self.counter),
            "min_angle": float(round(self._current_rep_min_angle, 1)),
            "max_angle": float(round(self._current_rep_max_angle, 1)),
            "rom": float(round(
                self._current_rep_max_angle - self._current_rep_min_angle, 1)),
            "is_valid": bool(is_valid),
            "feedback": str(feedback),
            "form_score": float(round(form_score, 2)),
            "start_frame": int(start_frame) if start_frame is not None else None,
            "end_frame": int(self.current_frame),
            "duration_ms": int(duration_ms) if duration_ms is not None else None,
        }

    def _reset_rep_tracking(self):
        """Reset per-rep angle trackers after a rep is recorded."""
        self._current_rep_min_angle = 180.0
        self._current_rep_max_angle = 0.0
        self._descent_started = False
        self._descent_min_angle = 180.0
        self._start_frame_locked = False

    def _lock_start_frame(self):
        """Lock start_frame on the FIRST transition to 'up', preventing jitter overwrites."""
        if not self._start_frame_locked:
            self._current_rep_start_frame = self.current_frame
            self._current_rep_start_ms = self.current_timestamp_ms
            self._start_frame_locked = True

    def _update_wrong_exercise_tracking(self, ml_label: str,
                                        ml_confidence: float,
                                        expected_label: str,
                                        confidence_threshold: float = 0.6):
        """
        Percentage-based wrong exercise detection.
        Only flags the session if >30% of analyzed frames disagree.
        """
        self._total_analyzed_frames += 1
        if ml_confidence >= confidence_threshold and ml_label != expected_label:
            self._wrong_exercise_frame_count += 1

        if self._total_analyzed_frames > 0:
            ratio = self._wrong_exercise_frame_count / self._total_analyzed_frames
            self.wrong_exercise_detected = ratio > self._WRONG_EXERCISE_RATIO_THRESHOLD

    @property
    def wrong_exercise_ratio(self) -> float:
        """Percentage of frames where ML disagreed with expected exercise."""
        if self._total_analyzed_frames == 0:
            return 0.0
        return self._wrong_exercise_frame_count / self._total_analyzed_frames