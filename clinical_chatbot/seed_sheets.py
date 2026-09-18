"""
One-time (or re-run-anytime) seeding script: creates the Sheet tabs if
missing, then pushes rows from local CSVs into them. Safe to re-run — it
clears and rewrites each tab's data rows, leaving headers intact.

Expected CSVs in this folder (same shape as the old MongoDB version, plus
two new doctor-only columns):
  cdss_patients.csv        patient_id,name,age,gender,role,allergies,current_medications,medical_history
      -- role is patient/doctor/staff/clinician, same as before.
      -- for role=doctor rows, ALSO include: specialty,working_hours_start,working_hours_end
         (e.g. specialty=Cardiology, working_hours_start=09:00, working_hours_end=17:00)
      -- visit_reason/visit_category/needs_prescription_renewal/needs_medical_note
         are populated later by the in-app intake questionnaire, not by this
         CSV — leave them blank here.
  cdss_otc_medications.csv  medication_name,targets,interacts_with,contraindications,usage_note
      -- 'targets' replaces the old single-symptom-target column: a
         comma-separated list of symptom targets this medication covers.

Run: python seed_sheets.py
"""

import pandas as pd
import database as db


def _rows_from_csv(path: str) -> list[dict]:
    df = pd.read_csv(path)
    return df.where(pd.notnull(df), "").to_dict(orient="records")


def _clear_data_rows(tab: str):
    ws = db._ws(tab)
    row_count = len(ws.get_all_values())
    if row_count > 1:
        ws.batch_clear([f"A2:Z{row_count}"])
    db._invalidate(tab)


def seed_people(path: str = "cdss_patients.csv"):
    people = _rows_from_csv(path)
    patients, doctors, staff = [], [], []
    for p in people:
        role = str(p.get("role", "patient")).strip().lower()
        row = {
            "patient_id": p.get("patient_id"), "name": p.get("name", ""),
            "age": p.get("age", ""), "gender": p.get("gender", ""),
            "allergies": p.get("allergies", ""), "current_medications": p.get("current_medications", ""),
            "medical_history": p.get("medical_history", ""), "intake_completed": "true",
        }
        if role in ("doctor", "clinician"):
            doctors.append({
                "patient_id": p.get("patient_id"), "name": p.get("name", ""),
                "specialty": p.get("specialty", ""),
                "working_hours_start": p.get("working_hours_start", "09:00"),
                "working_hours_end": p.get("working_hours_end", "17:00"),
                "slot_minutes": p.get("slot_minutes", 30),
            })
        elif role == "staff":
            staff.append({"patient_id": p.get("patient_id"), "name": p.get("name", ""), "department": p.get("department", "")})
        else:
            patients.append(row)

    # "patients" is deliberately NOT seeded/cleared here — it's SMART CDSS's
    # own shared "Patients" sheet, containing real patient records. Wiping
    # its data rows on every seed run would destroy that data. Real patient
    # rows are created organically by the app (create_patient()) the first
    # time each SMART CDSS patient opens the chatbot.
    if patients:
        print(
            f"Note: {len(patients)} row(s) in {path} have role=patient (or blank) "
            "and were skipped — patients now come from SMART CDSS, not this CSV."
        )
    for tab, rows in (("doctors", doctors), ("staff", staff)):
        _clear_data_rows(tab)
        for r in rows:
            db._append_row(tab, r)
    print(f"Seeded {len(doctors)} doctors, {len(staff)} staff.")


def seed_otc(path: str = "cdss_otc_medications.csv"):
    rows = _rows_from_csv(path)
    _clear_data_rows("otc_medications")
    for r in rows:
        db._append_row("otc_medications", {
            "medication_name": r.get("medication_name", ""), "targets": r.get("targets", ""),
            "interacts_with": r.get("interacts_with", ""), "contraindications": r.get("contraindications", ""),
            "usage_note": r.get("usage_note", ""),
        })
    print(f"Seeded {len(rows)} OTC medication entries.")


if __name__ == "__main__":
    if not db.is_available():
        raise SystemExit(f"Google Sheets unavailable: {db.DB_ERROR}")
    print(db.ensure_schema())
    seed_people()
    seed_otc()
    print("Done.")
