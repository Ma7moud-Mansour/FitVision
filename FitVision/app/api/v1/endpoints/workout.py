import logging
import shutil
import os
import uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends
from sqlalchemy.orm import Session
from app.utils.video_processor import VideoProcessor
from app.core.config import get_settings
from app.core.enums import ExerciseType
from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models import User
from app.crud.crud import (
    create_workout_session,
    update_workout_session_results,
    mark_workout_failed,
    get_workout_session,
    get_user_workouts,
)
from app.schemas.schemas import WorkoutUploadResponse, WorkoutSessionResponse, WorkoutDetailResponse

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# Create upload dir
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


def _process_in_background(session_id, video_path: str, exercise_type: str):
    """
    Background task — processes the video and updates the DB with results.
    Uses its own DB session since background tasks run outside the request lifecycle.
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        processor = VideoProcessor()
        result = processor.process_workout(video_path, exercise_type)

        if result:
            update_workout_session_results(
                db=db,
                session_id=session_id,
                processed_video_path=result["output_path"],
                total_reps=result["total_reps"],
                duration_seconds=result["duration_seconds"],
            )
            logger.info(f"Session {session_id} completed — {result['total_reps']} reps")
        else:
            mark_workout_failed(db, session_id)
            logger.error(f"Session {session_id} failed during processing")
    except Exception as e:
        mark_workout_failed(db, session_id)
        logger.exception(f"Background task error for session {session_id}: {e}")
    finally:
        db.close()


@router.post("/upload-video", response_model=WorkoutUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    exercise_type: ExerciseType,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a workout video for processing. Requires authentication."""
    # 1. التأكد من نوع الملف (Security Check)
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # 2. Check file size (read first, then validate)
    contents = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
        )

    # 3. توليد اسم فريد للملف
    file_extension = file.filename.split(".")[-1] if file.filename else "mp4"
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

    # 4. حفظ الملف على السيرفر
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    # 5. Create DB session record
    workout_session = create_workout_session(
        db=db,
        user_id=current_user.id,
        exercise_type=exercise_type.value,
        original_video_path=file_path,
    )

    # 6. تشغيل المعالجة في الخلفية (Background Task)
    background_tasks.add_task(
        _process_in_background,
        workout_session.id,
        file_path,
        exercise_type.value,
    )

    return WorkoutUploadResponse(
        message="Video uploaded successfully. Processing started.",
        session_id=workout_session.id,
        filename=unique_filename,
        exercise=exercise_type.value,
    )


@router.get("/session/{session_id}/status", response_model=WorkoutSessionResponse)
def get_session_status(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Check the processing status of a workout session."""
    session = get_workout_session(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/my-workouts", response_model=list[WorkoutSessionResponse])
def list_my_workouts(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all workout sessions for the authenticated user."""
    return get_user_workouts(db, current_user.id, skip=skip, limit=limit)