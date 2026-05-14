from datetime import datetime
from uuid import UUID
from sqlalchemy.orm import Session
from app.db.models import User, WorkoutSession, RepData
from app.core.security import hash_password


# ──────────────── User CRUD ────────────────

def create_user(db: Session, email: str, username: str, password: str, full_name: str = None) -> User:
    user = User(
        email=email,
        username=username,
        hashed_password=hash_password(password),
        full_name=full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: UUID) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


# ──────────────── Workout Session CRUD ────────────────

def create_workout_session(
    db: Session, user_id: UUID, exercise_type: str, original_video_path: str
) -> WorkoutSession:
    session = WorkoutSession(
        user_id=user_id,
        exercise_type=exercise_type,
        original_video_path=original_video_path,
        status="processing",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def update_workout_session_results(
    db: Session,
    session_id: UUID,
    processed_video_path: str,
    total_reps: int,
    duration_seconds: float,
    avg_form_score: float = None,
    ai_insights: str = None,
    wrong_exercise_detected: bool = False,
    rep_data: list = None,
    status: str = "completed",
) -> WorkoutSession | None:
    session = db.query(WorkoutSession).filter(WorkoutSession.id == session_id).first()
    if not session:
        return None
    session.processed_video_path = processed_video_path
    session.total_reps = total_reps
    session.duration_seconds = duration_seconds
    session.avg_form_score = avg_form_score
    session.ai_insights = ai_insights
    session.wrong_exercise_detected = wrong_exercise_detected
    session.status = status
    session.completed_at = datetime.utcnow()

    if rep_data:
        for r in rep_data:
            db.add(RepData(
                session_id=session_id,
                rep_number=r["rep_number"],
                min_angle=r.get("min_angle"),
                max_angle=r.get("max_angle"),
                rom=r.get("rom"),
                form_score=r.get("form_score"),
                feedback=r.get("feedback"),
                is_valid=r.get("is_valid", True),
                start_frame=r.get("start_frame"),
                end_frame=r.get("end_frame"),
                duration_ms=r.get("duration_ms")
            ))
    db.commit()
    db.refresh(session)
    return session


def mark_workout_failed(db: Session, session_id: UUID) -> None:
    session = db.query(WorkoutSession).filter(WorkoutSession.id == session_id).first()
    if session:
        session.status = "failed"
        session.completed_at = datetime.utcnow()
        db.commit()


def get_workout_session(db: Session, session_id: UUID) -> WorkoutSession | None:
    return db.query(WorkoutSession).filter(WorkoutSession.id == session_id).first()


def get_user_workouts(db: Session, user_id: UUID, skip: int = 0, limit: int = 20):
    return (
        db.query(WorkoutSession)
        .filter(WorkoutSession.user_id == user_id)
        .order_by(WorkoutSession.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
