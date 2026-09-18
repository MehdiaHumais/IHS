DOCTOR SCHEDULE V7 FIX

This build fixes the remaining empty-doctor-schedule problem by making schedule
identity resolution tolerant of both current and older appointment rows.

New bookings:
- Resolve selected doctor to the canonical registration user_id before writing.
- Persist both doctor_id and doctor_name.
- Perform a read-after-write check. The patient does NOT get a successful
  booking result unless the same appointment can be read back in that doctor's
  schedule.

Doctor schedule:
- Fresh Google Sheets read every time.
- Matches canonical doctor user_id.
- Also recognizes legacy rows where older builds stored the doctor display name
  in doctor_id, doctor_name, or specialist.
- Normalizes whitespace/capitalization.
- Returns a safe appointment_count for diagnostics.

Run RUN_APP_DOCTOR_SCHEDULE_V7.bat from a newly extracted folder.
