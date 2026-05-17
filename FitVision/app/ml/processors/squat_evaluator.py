import numpy as np
from .base_evaluator import ExerciseEvaluator

class SquatEvaluator(ExerciseEvaluator):
    def evaluate(self, results, w, h, engine):
        landmarks = results.pose_landmarks.landmark
        
        # 1. تحديد النقط (Hip, Knee, Ankle)
        hip = [landmarks[engine.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[engine.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[engine.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[engine.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[engine.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[engine.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

        # 2. حساب الزاوية
        angle = engine.calculate_angle(hip, knee, ankle)

        # 3. الـ Counter Logic والـ Feedback
        # الـ Thresholds دي بنجيبها من مراجع الـ Biomechanics
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.feedback = f"Perfect! Rep: {self.counter}"
        elif angle > 90 and self.stage == "up":
            self.feedback = "Go lower for better ROM!"

        return {"angle": angle, "counter": self.counter, "feedback": self.feedback}