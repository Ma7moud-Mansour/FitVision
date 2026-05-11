import cv2
import os
import logging
from app.ml.processors.pose_engine import PoseEngine
from app.ml.processors import get_evaluator
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VideoProcessor:
    def __init__(self):
        self.engine = PoseEngine()
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

    def process_workout(self, video_path: str, exercise_type: str) -> dict | None:
        """
        Main processing pipeline.
        Returns a dict with processing results or None on failure.
        """
        # 1. اختيار الـ Evaluator المناسب بناءً على نوع التمرين
        evaluator = get_evaluator(exercise_type)
        if not evaluator:
            logger.error(f"Exercise '{exercise_type}' not supported")
            return None

        cap = cv2.VideoCapture(video_path)
        out = None

        try:
            if not cap.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return None

            # 2. جلب بيانات الفيديو الأصلية
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # تنفيذ الـ FPS Skipping
            target_fps = settings.TARGET_FPS
            skip_interval = max(1, int(original_fps / target_fps))

            # 3. تجهيز ملف الـ Output
            output_filename = f"processed_{os.path.basename(video_path)}"
            output_path = os.path.join(settings.PROCESSED_DIR, output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

            frame_count = 0
            processed_count = 0
            last_analysis = {}

            logger.info(
                f"Processing started: {video_path} | "
                f"FPS: {original_fps} → {target_fps} | "
                f"Skip: every {skip_interval} frames | "
                f"Total frames: {total_frames}"
            )

            while cap.isOpened():
                # الـ Optimization: لو الفريم مش هيتعالج، بس نعمل grab() بدون decode
                if frame_count % skip_interval != 0:
                    cap.grab()  # ~4x faster — advances without decoding
                    frame_count += 1
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                # أ. استخراج الـ Landmarks
                results = self.engine.extract_landmarks(frame)

                if results and results.pose_landmarks:
                    # ب. رسم الـ Skeleton الأصلي
                    self.engine.mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, self.engine.mp_pose.POSE_CONNECTIONS,
                        self.engine.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        self.engine.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                    # ج. تنفيذ لوجيك التمرين (Squat, Pushup, etc.)
                    last_analysis = evaluator.evaluate(
                        results.pose_landmarks.landmark,
                        width,
                        height,
                        self.engine
                    )

                    # د. عرض النتائج على الشاشة (UI Overlay)
                    self._draw_ui(frame, last_analysis, exercise_type)

                # حفظ الفريم المعالج
                out.write(frame)
                processed_count += 1
                frame_count += 1

            logger.info(
                f"Processing complete: {output_path} | "
                f"Frames processed: {processed_count}/{total_frames}"
            )

            return {
                "output_path": output_path,
                "total_reps": last_analysis.get("counter", 0),
                "frames_processed": processed_count,
                "duration_seconds": round(processed_count / target_fps, 2),
            }

        except Exception as e:
            logger.exception(f"Error processing video: {e}")
            return None
        finally:
            cap.release()
            if out:
                out.release()

    def _draw_ui(self, frame, analysis, exercise_type):
        """دالة مساعدة لرسم الـ Feedback والـ Counter على الفيديو"""
        # رسم مستطيل خلفية للبيانات (عشان تبان لو الخلفية فاتحة)
        cv2.rectangle(frame, (0, 0), (350, 180), (245, 117, 16), -1)

        # 1. عرض نوع التمرين
        cv2.putText(frame, f"EXE: {exercise_type.upper()}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. عرض الـ Rep Counter
        cv2.putText(frame, f"REPS: {analysis.get('counter', 0)}",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

        # 3. عرض الـ Feedback (Stage or Error)
        status_color = (0, 255, 0) if "Perfect" in analysis.get('feedback', '') or "\u2705" in analysis.get('feedback', '') else (0, 255, 255)
        cv2.putText(frame, f"{analysis.get('feedback', '')}",
                    (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)

        # 4. عرض الزاوية الحالية عند المفصل
        if 'angle' in analysis:
            cv2.putText(frame, f"Angle: {int(analysis['angle'])}", (15, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. ML Model Prediction (if available)
        if 'ml_label' in analysis:
            conf = analysis.get('ml_confidence', 0)
            ml_color = (0, 255, 0) if conf > 0.7 else (0, 200, 255) if conf > 0.5 else (0, 100, 255)
            cv2.putText(frame, f"ML: {analysis['ml_label']} ({conf:.0%})", (15, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ml_color, 1, cv2.LINE_AA)