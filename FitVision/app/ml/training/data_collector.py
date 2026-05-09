import cv2
import mediapipe as mp
import csv
import os

# إعدادات MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def collect_data(video_path, label, csv_file):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # حساب الـ skip interval عشان نوصل لـ 10 فريم في الثانية
    # لو الفيديو 30 فريم، هناخد فريم كل 3 فريمات
    skip_interval = max(1, int(fps / 10))
    
    frame_count = 0
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # الـ Frame Skipping
            if frame_count % skip_interval == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                
                if results.pose_landmarks:
                    # تحويل الـ landmarks لـ List واحدة مفرودة (Flattened)
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                    
                    # إضافة الـ label في أول الصف
                    writer.writerow([label] + landmarks)
            
            frame_count += 1
            
    cap.release()
    print(f"✅ Finished processing: {video_path}")

# مثال للاستخدام:
# collect_data('raw_data/squat_correct.mp4', 1, 'training_data.csv')