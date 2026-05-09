from pydantic import BaseModel, EmailStr, Field
from uuid import UUID
from datetime import datetime
from typing import Optional
from app.core.enums import ExerciseType


# ──────────────── Auth Schemas ────────────────

class UserCreate(BaseModel):
    email: str
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: UUID
    email: str
    username: str
    full_name: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ──────────────── Workout Schemas ────────────────

class WorkoutSessionResponse(BaseModel):
    id: UUID
    exercise_type: str
    status: str
    original_video_path: str
    processed_video_path: Optional[str]
    total_reps: int
    avg_form_score: Optional[float]
    duration_seconds: Optional[float]
    started_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class WorkoutUploadResponse(BaseModel):
    message: str
    session_id: UUID
    filename: str
    exercise: str


class RepDataResponse(BaseModel):
    rep_number: int
    min_angle: Optional[float]
    max_angle: Optional[float]
    rom: Optional[float]
    form_score: Optional[float]
    feedback: Optional[str]
    is_valid: bool

    model_config = {"from_attributes": True}


class WorkoutDetailResponse(WorkoutSessionResponse):
    reps: list[RepDataResponse] = []
