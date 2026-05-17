"""
SmartSquatEvaluator — Hybrid ML + Heuristic Squat Analysis
──────────────────────────────────────────────────────────
Uses ML model to classify exercise type and confirm it's a squat,
then applies heuristic angle-based logic for rep counting and form feedback.
Falls back to pure heuristics if ML confidence is below threshold.

3-Tier Grading System:
    🟢 Full Rep:    knee angle < 100°  →  counted, is_valid=True
    🟡 Partial Rep: knee angle 100–130° →  counted, is_valid=False, coaching feedback
    🔴 Failed Rep:  knee angle > 130°   →  NOT counted, recorded with error
"""
import numpy as np
from .base_evaluator import ExerciseEvaluator, SQUAT_IDEAL_ROM
from .ml_evaluator import MLEvaluatorMixin

# ─── Thresholds ───
ML_CONFIDENCE_THRESHOLD = 0.6

# Biomechanics thresholds (knee angle at bottom of squat)
FULL_REP_ANGLE = 100       # Below this = full depth (relaxed from 90° for real-world tolerance)
PARTIAL_REP_ANGLE = 130    # Between FULL and PARTIAL = partial rep
STANDING_ANGLE = 140       # Above this = standing position


class SmartSquatEvaluator(MLEvaluatorMixin, ExerciseEvaluator):
    """
    ML-Enhanced Squat Evaluator with 3-tier rep grading.

    Pipeline per frame:
        1. ML model classifies exercise type + confidence
        2. Heuristic angle logic runs for rep counting (always)
        3. Rep quality is graded: Full / Partial / Failed
        4. Wrong exercise flagged only if >30% frames disagree
    """

    def evaluate(self, results, w, h, engine):
        landmarks = results.pose_landmarks.landmark
        world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else landmarks
        
        # ─── ML Prediction (exercise classification) ───
        ml_label, ml_confidence = self.predict_exercise(world_landmarks, 1, 1)
        ml_info = {"ml_label": ml_label, "ml_confidence": round(ml_confidence, 3)}

        # ─── Wrong exercise tracking (percentage-based) ───
        self._update_wrong_exercise_tracking(ml_label, ml_confidence, "squat")

        # ─── Heuristic: Angle-based rep counting ───
        hip = [landmarks[engine.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[engine.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[engine.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[engine.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[engine.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[engine.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

        angle = engine.calculate_angle(hip, knee, ankle)
        self._current_rep_min_angle = min(self._current_rep_min_angle, angle)
        self._current_rep_max_angle = max(self._current_rep_max_angle, angle)

        # ─── Descent detection ───
        if angle < STANDING_ANGLE and self.stage == "up":
            self._descent_started = True
            self._descent_min_angle = min(self._descent_min_angle, angle)
        elif self._descent_started:
            self._descent_min_angle = min(self._descent_min_angle, angle)

        # ─── State Machine: Rep Counting with 3-Tier Grading ───
        if angle > STANDING_ANGLE:
            if self.stage == "down" or (self._descent_started and self.stage == "up"):
                # User returned to standing — evaluate the rep attempt
                min_angle_reached = self._current_rep_min_angle
                actual_rom = self._current_rep_max_angle - min_angle_reached

                if min_angle_reached < FULL_REP_ANGLE:
                    # ────── 🟢 FULL REP ──────
                    self.counter += 1
                    form_score = self._compute_form_score(actual_rom, SQUAT_IDEAL_ROM)
                    depth_feedback = f"Great depth at {int(min_angle_reached)}°! Full ROM."
                    self.feedback = f"✅ Perfect Rep #{self.counter} — {depth_feedback}"
                    entry = self._build_rep_entry(
                        is_valid=True, feedback=depth_feedback, form_score=form_score
                    )
                    self.rep_data.append(entry)

                elif min_angle_reached < PARTIAL_REP_ANGLE:
                    # ────── 🟡 PARTIAL REP ──────
                    self.counter += 1
                    form_score = self._compute_form_score(actual_rom, SQUAT_IDEAL_ROM)
                    depth_feedback = (
                        f"Shallow Depth: {int(min_angle_reached)}° — "
                        f"aim for below {FULL_REP_ANGLE}°"
                    )
                    self.feedback = f"⚠️ Partial Rep #{self.counter} — {depth_feedback}"
                    entry = self._build_rep_entry(
                        is_valid=False, feedback=depth_feedback, form_score=form_score
                    )
                    self.rep_data.append(entry)

                else:
                    # ────── 🔴 FAILED ATTEMPT ──────
                    if self._descent_started and min_angle_reached < STANDING_ANGLE:
                        # They tried but barely moved — record but don't count
                        form_score = self._compute_form_score(actual_rom, SQUAT_IDEAL_ROM)
                        depth_feedback = (
                            f"Incomplete rep — only reached {int(min_angle_reached)}°. "
                            f"Need below {PARTIAL_REP_ANGLE}° to count."
                        )
                        self.feedback = f"🔴 {depth_feedback}"
                        # Don't increment counter, but record for coaching
                        temp_counter = self.counter
                        self.counter += 1  # Temporarily for entry
                        entry = self._build_rep_entry(
                            is_valid=False, feedback=depth_feedback, form_score=form_score
                        )
                        self.counter = temp_counter  # Restore — failed reps don't count
                        entry["rep_number"] = self.counter + 1  # Label as "next attempt"
                        self.rep_data.append(entry)

                # Reset tracking for next rep
                self._reset_rep_tracking()

            self.stage = "up"
            # Lock start frame on FIRST transition to up (prevents jitter)
            self._lock_start_frame()

        if angle < FULL_REP_ANGLE and self.stage == "up":
            self.stage = "down"

        # ─── Hybrid Feedback (ML-enhanced messaging) ───
        if ml_confidence >= ML_CONFIDENCE_THRESHOLD:
            if ml_label == "squat":
                if angle < FULL_REP_ANGLE and self.stage == "down":
                    if not self.feedback.startswith("✅"):
                        self.feedback = f"✅ ML-Confirmed Squat ({ml_confidence:.0%}) — Go deeper!"
                elif angle > STANDING_ANGLE and self.stage == "up":
                    if not self.feedback.startswith(("✅", "⚠️", "🔴")):
                        self.feedback = f"Squat detected ({ml_confidence:.0%})"
            else:
                if not self.feedback.startswith(("✅", "⚠️", "🔴")):
                    self.feedback = f"⚠️ Expected squat, detected {ml_label} ({ml_confidence:.0%})"
        else:
            if not self.feedback:
                if angle > FULL_REP_ANGLE and angle < STANDING_ANGLE:
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
