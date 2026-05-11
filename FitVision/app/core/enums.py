from enum import Enum


class ExerciseType(str, Enum):
    SQUAT = "squat"
    PUSHUP = "pushup"
    PULL_UP = "pull_up"
    SITUP = "situp"
    JUMPING_JACK = "jumping_jack"
