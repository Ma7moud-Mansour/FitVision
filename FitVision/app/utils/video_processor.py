"""
VideoProcessor — Core video processing pipeline.
──────────────────────────────────────────────────
Handles frame skipping, pose extraction, evaluator execution,
overlay drawing, and result aggregation.
"""
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
                # FIX E1: الـ Optimization مع فحص نتيجة grab() لمنع الـ Infinite Loop
                if frame_count % skip_interval != 0:
                    if not cap.grab():  # Break if grab() fails (corrupted/EOF)
                        break
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
                    # FIX E3: Use processed_count for accurate timestamp calculation
                    evaluator.current_frame = processed_count
                    evaluator.current_timestamp_ms = int((processed_count / target_fps) * 1000)
                    last_analysis = evaluator.evaluate(
                        results,
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

            # ─── Aggregate Results ───
            rep_data = evaluator.rep_data
            avg_score = None
            if rep_data:
                valid_scores = [r["form_score"] for r in rep_data
                                if r["form_score"] is not None]
                avg_score = float(sum(valid_scores) / len(valid_scores)) if valid_scores else None

            # ─── Generate Data-Driven AI Insights ───
            ai_insights = self._generate_ai_insights(
                evaluator, rep_data, avg_score, exercise_type
            )

            wrong_exe = evaluator.wrong_exercise_detected

            return {
                "output_path": output_path,
                "total_reps": last_analysis.get("counter", 0),
                "frames_processed": processed_count,
                "duration_seconds": round(processed_count / target_fps, 2),
                "rep_data": rep_data,
                "avg_form_score": avg_score,
                "wrong_exercise_detected": wrong_exe,
                "ai_insights": ai_insights
            }

        except Exception as e:
            logger.exception(f"Error processing video: {e}")
            return None
        finally:
            cap.release()
            if out:
                out.release()
            # FIX E2: Release PoseEngine resources to prevent memory leak
            try:
                self.engine.pose.close()
            except Exception:
                pass

    def _generate_ai_insights(self, evaluator, rep_data: list,
                              avg_score: float | None,
                              exercise_type: str) -> str:
        """Generate specific, data-driven coaching insights based on rep analysis."""
        if evaluator.wrong_exercise_detected:
            ratio_pct = int(evaluator.wrong_exercise_ratio * 100)
            return (
                f"⚠️ Our AI detected you may be performing a different exercise "
                f"({ratio_pct}% of frames disagreed). Please ensure you selected "
                f"the correct exercise type."
            )

        if not rep_data:
            return "No reps were detected. Make sure the full body is visible in the frame."

        # Count valid vs invalid reps
        valid_count = sum(1 for r in rep_data if r["is_valid"])
        invalid_count = len(rep_data) - valid_count
        total = len(rep_data)

        insights = []

        # Overall assessment
        if avg_score is not None:
            if avg_score >= 0.85:
                insights.append(f"🏆 Excellent form! Average score: {int(avg_score * 100)}%.")
            elif avg_score >= 0.65:
                insights.append(f"👍 Good effort! Average score: {int(avg_score * 100)}%. Room for improvement.")
            else:
                insights.append(f"💪 Keep practicing! Average score: {int(avg_score * 100)}%. Focus on depth and control.")

        # Valid/Invalid breakdown
        if invalid_count > 0:
            insights.append(
                f"📊 {valid_count}/{total} reps had full range of motion. "
                f"{invalid_count} rep(s) were partial or incomplete."
            )

        # Identify the weakest rep
        scored_reps = [r for r in rep_data if r["form_score"] is not None]
        if scored_reps:
            weakest = min(scored_reps, key=lambda r: r["form_score"])
            if weakest["form_score"] < 0.6:
                insights.append(
                    f"📉 Weakest rep: #{weakest['rep_number']} "
                    f"(Score: {int(weakest['form_score'] * 100)}%). "
                    f"{weakest.get('feedback', '')}"
                )

        # Trend analysis (are they getting tired?)
        if len(scored_reps) >= 3:
            first_half = scored_reps[:len(scored_reps) // 2]
            second_half = scored_reps[len(scored_reps) // 2:]
            first_avg = sum(r["form_score"] for r in first_half) / len(first_half)
            second_avg = sum(r["form_score"] for r in second_half) / len(second_half)
            if second_avg < first_avg - 0.15:
                insights.append(
                    "📉 Your form declined in the second half. "
                    "Consider reducing reps or taking longer rest periods."
                )
            elif second_avg > first_avg + 0.1:
                insights.append("📈 Your form improved as you warmed up. Great consistency!")

        return " ".join(insights) if insights else "Great form! Keep it up."

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