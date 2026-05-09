import cv2
import mediapipe as mp
import numpy as np

class PoseEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, # False عشان إحنا شغالين على فيديو (Tracking)
            model_complexity=1,      # 0 سريع، 1 متوازن، 2 دقيق جداً بس تقيل
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_landmarks(self, frame):
        # MediaPipe بيشتغل بـ RGB و OpenCV بـ BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results  # Return full results (video_processor checks .pose_landmarks)

    @staticmethod
    def calculate_angle(a, b, c):
        # a, b, c are [x, y] coordinates
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle