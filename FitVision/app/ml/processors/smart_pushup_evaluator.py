"""
SmartPushupEvaluator — Hybrid ML + Heuristic Pushup Analysis
─────────────────────────────────────────────────────────────
3-Tier Grading System:
    🟢 Full Rep:    elbow angle < 90°  →  counted, is_valid=True
    🟡 Partial Rep: elbow angle 90–110° →  counted, is_valid=False, coaching feedback
    🔴 Failed Rep:  elbow angle > 110°  →  NOT counted, recorded with error

Also checks back alignment (hip-shoulder Y distance) as a penalty factor.
"""
import numpy as np
from .base_evaluator import ExerciseEvaluator, PUSHUP_IDEAL_ROM
from .ml_evaluator import MLEvaluatorMixin

# ─── Thresholds ───
ML_CONFIDENCE_THRESHOLD = 0.6

# Biomechanics thresholds (elbow angle at bottom of pushup)
FULL_REP_ANGLE = 90        # Below this = full depth
PARTIAL_REP_ANGLE = 110    # Between FULL and PARTIAL = partial rep
STANDING_ANGLE = 140       # Above this = arms extended (top position)

# Alignment
MAX_ALIGNMENT_ERROR = 0.15  # Normalized hip-shoulder Y distance threshold


class SmartPushupEvaluator(MLEvaluatorMixin, ExerciseEvaluator):
    """
    ML-Enhanced Pushup Evaluator with 3-tier rep grading.
    ML classifies exercise type; heuristics handle rep counting,
    alignment check, and form grading.
    """

    def evaluate(self, results, w, h, engine):
        landmarks = results.pose_landmarks.landmark
        world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else landmarks
        
        # ─── ML Prediction ───
        ml_label, ml_confidence = self.predict_exercise(world_landmarks, 1, 1)
        ml_info = {"ml_label": ml_label, "ml_confidence": round(ml_confidence, 3)}

        # ─── Wrong exercise tracking (percentage-based) ───
        self._update_wrong_exercise_tracking(ml_label, ml_confidence, "push_up")

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
        form_ok = alignment_error <= MAX_ALIGNMENT_ERROR
        alignment_penalty = 0.3 if not form_ok else 0.0

        self._current_rep_min_angle = min(self._current_rep_min_angle, angle)
        self._current_rep_max_angle = max(self._current_rep_max_angle, angle)

        # ─── Descent detection ───
        if angle < STANDING_ANGLE and self.stage == "up":
            self._descent_started = True
            self._descent_min_angle = min(self._descent_min_angle, angle)
        elif self._descent_started:
            self._descent_min_angle = min(self._descent_min_angle, angle)

        # ─── State Machine: Rep Counting with 3-Tier Grading ───
        if angle < FULL_REP_ANGLE:
            self.stage = "down"

        if angle > STANDING_ANGLE and (self.stage == "down" or self._descent_started):
            # User returned to top — evaluate the rep attempt
            min_angle_reached = self._current_rep_min_angle
            actual_rom = self._current_rep_max_angle - min_angle_reached

            alignment_msg = " | Back sagging — keep core tight!" if not form_ok else ""

            if min_angle_reached < FULL_REP_ANGLE:
                # ────── 🟢 FULL REP ──────
                self.counter += 1
                form_score = self._compute_form_score(
                    actual_rom, PUSHUP_IDEAL_ROM, penalty=alignment_penalty
                )
                depth_feedback = f"Good depth at {int(min_angle_reached)}°"
                form_msg = "Good Form" if form_ok else "Keep your back straight!"
                full_feedback = f"{depth_feedback} — {form_msg}"
                self.feedback = f"✅ Rep #{self.counter} — {full_feedback}"
                entry = self._build_rep_entry(
                    is_valid=True, feedback=full_feedback, form_score=form_score
                )
                self.rep_data.append(entry)

            elif min_angle_reached < PARTIAL_REP_ANGLE:
                # ────── 🟡 PARTIAL REP ──────
                self.counter += 1
                form_score = self._compute_form_score(
                    actual_rom, PUSHUP_IDEAL_ROM, penalty=alignment_penalty
                )
                depth_feedback = (
                    f"Shallow push-up: {int(min_angle_reached)}° — "
                    f"aim below {FULL_REP_ANGLE}°{alignment_msg}"
                )
                self.feedback = f"⚠️ Partial Rep #{self.counter} — {depth_feedback}"
                entry = self._build_rep_entry(
                    is_valid=False, feedback=depth_feedback, form_score=form_score
                )
                self.rep_data.append(entry)

            else:
                # ────── 🔴 FAILED ATTEMPT ──────
                if self._descent_started and min_angle_reached < STANDING_ANGLE:
                    form_score = self._compute_form_score(
                        actual_rom, PUSHUP_IDEAL_ROM, penalty=alignment_penalty
                    )
                    depth_feedback = (
                        f"Incomplete push-up — only reached {int(min_angle_reached)}°. "
                        f"Need below {PARTIAL_REP_ANGLE}° to count.{alignment_msg}"
                    )
                    self.feedback = f"🔴 {depth_feedback}"
                    temp_counter = self.counter
                    self.counter += 1
                    entry = self._build_rep_entry(
                        is_valid=False, feedback=depth_feedback, form_score=form_score
                    )
                    self.counter = temp_counter
                    entry["rep_number"] = self.counter + 1
                    self.rep_data.append(entry)

            # Reset tracking for next rep
            self._reset_rep_tracking()
            self.stage = "up"
            self._lock_start_frame()

        elif angle > STANDING_ANGLE:
            self.stage = "up"
            self._lock_start_frame()

        # ─── Hybrid Feedback (ML-enhanced messaging) ───
        if ml_confidence >= ML_CONFIDENCE_THRESHOLD:
            if ml_label == "push_up":
                if not self.feedback.startswith(("✅", "⚠️", "🔴")):
                    self.feedback = f"Pushup detected ({ml_confidence:.0%})"
            else:
                if not self.feedback.startswith(("✅", "⚠️", "🔴")):
                    self.feedback = f"⚠️ Expected pushup, detected {ml_label} ({ml_confidence:.0%})"
        else:
            if not self.feedback:
                self.feedback = "Good Form" if form_ok else "Keep your back straight!"

        return {
            "angle": angle,
            "counter": self.counter,
            "feedback": self.feedback,
            **ml_info,
        }

    def get_ml_prediction(self, landmarks):
        return self.predict_exercise(landmarks)
