DOCTOR SCHEDULE NOTIFICATION FIX (v6)

- The selected doctor's exact SMART CDSS user_id is saved to appointments.doctor_id.
- doctor_name is also stored for audit/display.
- Doctor schedule matching now ignores accidental whitespace/capitalization differences.
- /api/doctor-schedule already forces a fresh Google Sheets read, so newly booked appointments appear on login/Refresh.
- New appointments are marked NEW the first time that doctor views the schedule in that browser.
- Existing appointment rows remain compatible; doctor_name is appended non-destructively as a new header.

Run RUN_APP_DOCTOR_SCHEDULE_FIX.bat from a newly extracted folder.
