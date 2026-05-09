from .base_evaluator import ExerciseEvaluator


class PushupEvaluator(ExerciseEvaluator):
    def evaluate(self, landmarks, w, h, engine):
        # مفاصل الذراع (Shoulder, Elbow, Wrist)
        shoulder = [landmarks[engine.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[engine.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        elbow = [landmarks[engine.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                 landmarks[engine.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        wrist = [landmarks[engine.mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                 landmarks[engine.mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

        angle = engine.calculate_angle(shoulder, elbow, wrist)

        # لوجيك الـ Form: لازم الظهر والوسط يكونوا على خط واحد تقريباً
        hip_y = landmarks[engine.mp_pose.PoseLandmark.LEFT_HIP.value].y
        shoulder_y = landmarks[engine.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        
        # لو الوسط نازل أوي أو طالع أوي (Wrong Form)
        alignment_error = abs(hip_y - shoulder_y)
        
        if angle < 70:
            self.stage = "down"
        if angle > 160 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            
        form_msg = "Keep your back straight!" if alignment_error > 0.15 else "Good Form"
        
        return {"angle": angle, "counter": self.counter, "feedback": form_msg}