import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(150), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sessions = relationship("WorkoutSession", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username}>"


class WorkoutSession(Base):
    __tablename__ = "workout_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    exercise_type = Column(String(50), nullable=False)
    status = Column(String(20), default="processing")  # processing | completed | failed

    # Video paths
    original_video_path = Column(String(500), nullable=False)
    processed_video_path = Column(String(500), nullable=True)

    # Aggregate metrics (populated after processing)
    total_reps = Column(Integer, default=0)
    avg_form_score = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    frames_processed = Column(Integer, nullable=True)

    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="sessions")
    reps = relationship("RepData", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<WorkoutSession {self.exercise_type} — {self.status}>"


class RepData(Base):
    __tablename__ = "rep_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("workout_sessions.id", ondelete="CASCADE"), nullable=False)
    rep_number = Column(Integer, nullable=False)

    # Biomechanics data
    min_angle = Column(Float, nullable=True)
    max_angle = Column(Float, nullable=True)
    rom = Column(Float, nullable=True)  # Range of Motion

    # Timing
    start_frame = Column(Integer, nullable=True)
    end_frame = Column(Integer, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Quality assessment
    form_score = Column(Float, nullable=True)
    feedback = Column(Text, nullable=True)
    is_valid = Column(Boolean, default=True)

    # Raw landmark snapshot (for future ML retraining)
    landmark_snapshot = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("WorkoutSession", back_populates="reps")

    def __repr__(self):
        return f"<RepData #{self.rep_number} — score: {self.form_score}>"
