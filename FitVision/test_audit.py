"""Quick test to verify the audit refactor logic."""
from app.ml.processors.base_evaluator import ExerciseEvaluator, SQUAT_IDEAL_ROM

# Create a concrete test evaluator
class TestEval(ExerciseEvaluator):
    def evaluate(self, landmarks, w, h, engine):
        pass

e = TestEval()

# === ROM Scoring Tests ===
print("=== ROM Scoring Tests ===")
s1 = e._compute_form_score(80, SQUAT_IDEAL_ROM)
print(f"  Perfect ROM (80/80): {s1}")
assert s1 == 1.0, f"Expected 1.0, got {s1}"

s2 = e._compute_form_score(50, SQUAT_IDEAL_ROM)
print(f"  Partial ROM (50/80): {s2}")
assert 0.6 <= s2 <= 0.65, f"Expected ~0.63, got {s2}"

s3 = e._compute_form_score(20, SQUAT_IDEAL_ROM)
print(f"  Poor ROM (20/80): {s3}")
assert s3 == 0.25, f"Expected 0.25, got {s3}"

s4 = e._compute_form_score(80, SQUAT_IDEAL_ROM, penalty=0.3)
print(f"  Good ROM + penalty (80/80, p=0.3): {s4}")
assert s4 == 0.7, f"Expected 0.7, got {s4}"

# === Rep Entry Builder Tests ===
print("\n=== Rep Entry Builder ===")
e.counter = 3
e._current_rep_min_angle = 85.3
e._current_rep_max_angle = 168.7
e.current_frame = 100
e.current_timestamp_ms = 5000
e._current_rep_start_frame = 50
e._current_rep_start_ms = 2500

entry = e._build_rep_entry(is_valid=True, feedback="Test", form_score=0.95)
print(f"  Entry: {entry}")
assert isinstance(entry["min_angle"], float)
assert isinstance(entry["rep_number"], int)
assert isinstance(entry["is_valid"], bool)
assert entry["duration_ms"] == 2500
assert entry["rom"] == round(168.7 - 85.3, 1)
print("  Types OK: all native Python types [PASS]")

# === Wrong Exercise Tracking ===
print("\n=== Wrong Exercise Tracking ===")
e2 = TestEval()
e2._update_wrong_exercise_tracking("squat", 0.9, "squat")
print(f"  1 matching frame: detected={e2.wrong_exercise_detected}")
assert e2.wrong_exercise_detected == False

for _ in range(3):
    e2._update_wrong_exercise_tracking("pushup", 0.8, "squat")
print(f"  After 3 wrong + 1 right (75% wrong): detected={e2.wrong_exercise_detected}")
assert e2.wrong_exercise_detected == True  # 3/4 = 75% > 30%

for _ in range(10):
    e2._update_wrong_exercise_tracking("squat", 0.9, "squat")
ratio = e2.wrong_exercise_ratio
print(f"  After 10 more right (ratio={ratio:.2f}): detected={e2.wrong_exercise_detected}")
assert e2.wrong_exercise_detected == False  # 3/14 = 21% < 30%

# === Start Frame Lock ===
print("\n=== Start Frame Lock ===")
e3 = TestEval()
e3.current_frame = 10
e3.current_timestamp_ms = 1000
e3._lock_start_frame()
assert e3._current_rep_start_frame == 10

e3.current_frame = 15  # Jitter -- should NOT overwrite
e3.current_timestamp_ms = 1500
e3._lock_start_frame()
assert e3._current_rep_start_frame == 10  # Still locked at 10
print(f"  Locked at frame 10, jitter at 15 blocked [PASS]")

e3._reset_rep_tracking()
e3.current_frame = 20
e3.current_timestamp_ms = 2000
e3._lock_start_frame()
assert e3._current_rep_start_frame == 20  # Reset unlocked it
print(f"  After reset, new lock at frame 20 [PASS]")

print("\n==============================")
print("  ALL TESTS PASSED")
print("==============================")
