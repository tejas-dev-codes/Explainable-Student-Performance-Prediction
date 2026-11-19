"""
Synthetic academic dataset generator for XAI Student Performance Dashboard V2.
Generated records are simulated and must never be treated as real student data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SUBJECTS = [
    "Mathematics", "Portuguese", "Physics", "Chemistry",
    "Computer Science", "English", "Biology", "Electronics"
]

def generate_synthetic_data(n_students: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible synthetic student profiles and one row per student-subject."""
    if n_students < 1:
        raise ValueError("n_students must be at least 1")
    rng = np.random.default_rng(seed)
    ids = [f"SYN-{10000+i}" for i in range(n_students)]
    age = rng.integers(16, 23, n_students)
    study_hours = np.clip(rng.gamma(shape=2.2, scale=3.2, size=n_students), 0, 35).round(1)
    attendance = np.clip(rng.normal(83, 12, n_students), 35, 100).round(1)
    assignment_completion = np.clip(rng.beta(5, 2, n_students)*100, 10, 100).round(1)
    participation = np.clip(rng.normal(6.5, 2, n_students), 0, 10).round(1)
    sleep = np.clip(rng.normal(7.1, 1.1, n_students), 3.5, 11).round(1)
    previous_average = np.clip(rng.normal(65, 16, n_students), 0, 100).round(1)
    internet_access = rng.choice(["Yes", "No"], n_students, p=[0.88, 0.12])
    family_education = rng.choice(["Primary", "Secondary", "Undergraduate", "Postgraduate"], n_students,
                                 p=[0.14, 0.42, 0.34, 0.10])
    extracurricular = rng.choice(["Low", "Medium", "High"], n_students, p=[0.3, 0.5, 0.2])
    risk_noise = rng.normal(0, 9, n_students)

    subject_difficulty = {
        "Mathematics": -4, "Portuguese": 2, "Physics": -6, "Chemistry": -3,
        "Computer Science": 5, "English": 4, "Biology": 1, "Electronics": -5
    }
    rows = []
    for i in range(n_students):
        # Latent academic tendency shared across subjects, plus per-subject randomness.
        profile_effect = (
            0.28*(previous_average[i]-60)
            + 0.75*(study_hours[i]-6)
            + 0.22*(attendance[i]-80)
            + 0.18*(assignment_completion[i]-75)
            + 1.1*(participation[i]-6)
            + 1.4*(sleep[i]-7)
            + (2.0 if internet_access[i] == "Yes" else -3.0)
            + risk_noise[i]
        )
        for subject in SUBJECTS:
            marks = np.clip(60 + profile_effect + subject_difficulty[subject] + rng.normal(0, 12), 0, 100)
            # Exam and coursework components are noisy but sum consistently to final marks.
            coursework = np.clip(marks + rng.normal(0, 8), 0, 100)
            exam = np.clip(0.65*marks + 0.35*coursework + rng.normal(0, 7), 0, 100)
            final_mark = np.clip(0.45*coursework + 0.55*exam, 0, 100).round(1)
            passed = bool(final_mark >= 50)
            if final_mark >= 85: grade = "A"
            elif final_mark >= 70: grade = "B"
            elif final_mark >= 60: grade = "C"
            elif final_mark >= 50: grade = "D"
            else: grade = "F"
            risk = "High" if final_mark < 45 else ("Medium" if final_mark < 60 else "Low")
            rows.append({
                "student_id": ids[i], "synthetic": True, "age": int(age[i]),
                "study_hours_week": float(study_hours[i]), "attendance_pct": float(attendance[i]),
                "assignment_completion_pct": float(assignment_completion[i]),
                "participation_score_10": float(participation[i]), "sleep_hours": float(sleep[i]),
                "previous_average_pct": float(previous_average[i]), "internet_access": internet_access[i],
                "family_education": family_education[i], "extracurricular_level": extracurricular[i],
                "subject": subject, "coursework_mark": float(coursework.round(1)),
                "exam_mark": float(exam.round(1)), "final_mark": float(final_mark),
                "pass": passed, "grade_band": grade, "risk_category": risk
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_student_subjects.csv", index=False)
    print(f"Created {len(df):,} synthetic student-subject records.")
