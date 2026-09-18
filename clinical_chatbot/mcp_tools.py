from datetime import datetime, timedelta
from typing import Optional

from fastmcp import FastMCP
import database as db

mcp = FastMCP("CDSS Tools")

# ---------------------------------------------------------------------------
# Deterministic safety data — kept in code, not in a prompt, on purpose.
# ---------------------------------------------------------------------------
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "severe bleeding", "unconscious", "unresponsive",
    "suicide", "kill myself", "self harm", "self-harm", "overdose",
    "stroke", "face drooping", "slurred speech", "sudden numbness",
    "anaphylaxis", "seizure", "not breathing", "blue lips",
    "worst headache of my life",
]

VALID_APPOINTMENT_ACTIONS = {"create", "lookup", "update", "cancel"}
VALID_STATUSES = {"scheduled", "cancelled", "completed", "waitlisted"}
VALID_CONSULTATION_MODES = {"video", "phone", "in-clinic"}
DEFAULT_SLOT_MINUTES = 30


def _db_unavailable_response() -> Optional[dict]:
    if not db.is_available():
        return {"error": f"Database unavailable: {db.DB_ERROR}. Check .env / sheet sharing."}
    return None


def check_emergency(message: str) -> dict:
    """
    Deterministic keyword scan for medical-emergency language in raw patient
    text. MUST be called first, before any other reasoning about the message.
    """
    normalized = " ".join(message.lower().split())
    for phrase in EMERGENCY_KEYWORDS:
        if phrase in normalized:
            return {
                "emergency": True,
                "matched_phrase": phrase,
                "message": (
                    "This may be a medical emergency. Advise the patient to call "
                    "emergency services or go to the nearest ER immediately. "
                    "Do not proceed with symptom assessment."
                ),
            }
    return {"emergency": False}


@mcp.tool()
def get_patient_record(patient_id: str, requester_patient_id: str, fields: Optional[list[str]] = None) -> dict:
    """
    Retrieves patient allergies, active medications, history, and today's
    visit context (visit_reason, visit_category, needs_prescription_renewal,
    needs_medical_note, regular_patient).

    requester_patient_id identifies who is asking (never trusted as a claimed
    role — looked up from the sheets):
      - the patient themselves -> full access to their own record
      - doctor/clinician        -> full access to any patient's record
      - staff                  -> no access; staff is scheduling-only and
                                   must never see clinical data
    """
    if not patient_id or not patient_id.strip():
        return {"error": "patient_id is required"}
    if not requester_patient_id or not requester_patient_id.strip():
        return {"error": "requester_patient_id is required to verify authorization."}

    db_error = _db_unavailable_response()
    if db_error:
        return db_error

    requester_role = db.resolve_role(requester_patient_id)
    if requester_role is None:
        return {"error": "Requester patient_id not found; cannot verify authorization."}

    is_self = requester_patient_id == patient_id
    if not is_self and not _caller_is_clinical(requester_role):
        return {"error": "Not authorized to view this patient's clinical record."}

    patient = db.get_patient(patient_id)
    if not patient:
        return {"error": "Patient not found"}

    if fields:
        allowed = {
            "allergies", "current_medications", "medical_history", "age", "gender", "patient_id", "name",
            "visit_reason", "visit_category", "needs_prescription_renewal", "needs_medical_note", "regular_patient",
        }
        requested = set(fields) & allowed
        if not requested:
            return {"error": f"None of the requested fields are valid. Allowed: {sorted(allowed)}"}
        return {k: patient.get(k) for k in requested}

    return patient


def get_patient_list() -> list[dict]:
    """Returns a patient list summary for UI selection."""
    if _db_unavailable_response():
        return []
    return db.get_patient_list()


# Register as an MCP tool without replacing the callable Python function.
mcp.tool()(get_patient_list)


def get_doctor_list() -> list[dict]:
    """Returns a doctor list summary (id, name, specialty) for UI selection."""
    if _db_unavailable_response():
        return []
    return db.get_doctor_list()


# Register as an MCP tool without replacing the callable Python function.
mcp.tool()(get_doctor_list)


def get_staff_list() -> list[dict]:
    """Returns a staff list summary for UI selection."""
    if _db_unavailable_response():
        return []
    return db.get_staff_list()


# Register as an MCP tool without replacing the callable Python function.
mcp.tool()(get_staff_list)


def check_medication_conflicts(patient_id: str, medication_names: list[str]) -> dict:
    """
    Checks candidate OTC medication names (freely reasoned by the agent, not
    looked up from a fixed database) against a patient's recorded allergies
    and current medications. This is a safety net on the PATIENT's own
    record, not a content knowledge base — it works from whatever names the
    agent proposes, using simple string matching against the patient's
    allergy/medication list.
    """
    db_error = _db_unavailable_response()
    if db_error:
        return {"conflicts": {}, "safe_to_suggest": False, "error": db_error["error"]}

    patient = db.get_patient(patient_id)
    if not patient:
        return {"conflicts": {}, "safe_to_suggest": False, "error": "Patient not found"}

    allergies = [a.lower() for a in patient.get("allergies", [])]
    current_meds = [m.lower() for m in patient.get("current_medications", [])]

    conflicts = {}
    for med_name in medication_names:
        name_lower = med_name.strip().lower()
        med_conflicts = []

        for allergy in allergies:
            if allergy in name_lower or name_lower in allergy:
                med_conflicts.append(f"Patient allergy on record: {allergy}")

        for current in current_meds:
            if current and (current in name_lower or name_lower in current):
                med_conflicts.append(f"Patient is already taking a related medication: {current}")

        conflicts[med_name] = med_conflicts

    has_any_conflict = any(v for v in conflicts.values())
    return {"conflicts": conflicts, "safe_to_suggest": not has_any_conflict}


# Register as an MCP tool without replacing the callable Python function.
mcp.tool()(check_medication_conflicts)


@mcp.tool()
def update_patient_profile(patient_id: str, requester_patient_id: str, field: str, add_values: list[str]) -> dict:
    """
    Adds new value(s) to a patient's own allergies, current_medications, or
    medical_history — WITHOUT erasing what's already on record. Use this any
    time during a conversation the patient mentions something new (e.g. "I
    just found out I'm allergic to penicillin", "I started taking metformin
    last month"), not only during the intake questionnaire — allergies and
    medications discovered mid-treatment are exactly the case this exists
    for.

    This only ever appends; it never removes or overwrites an existing
    entry. Duplicates (case-insensitive) are silently skipped rather than
    added twice.

    requester_patient_id identifies who is asking (looked up from the
    sheets, never trusted as a claimed identity) — same authorization rule
    as get_patient_record: the patient themselves, or a doctor, may update a
    record; general staff (scheduling-only) may not.
    """
    allowed_fields = {"allergies", "current_medications", "medical_history"}
    if field not in allowed_fields:
        return {"error": f"field must be one of {sorted(allowed_fields)}"}
    if not add_values:
        return {"error": "add_values cannot be empty"}
    if not patient_id or not patient_id.strip():
        return {"error": "patient_id is required"}
    if not requester_patient_id or not requester_patient_id.strip():
        return {"error": "requester_patient_id is required to verify authorization."}

    db_error = _db_unavailable_response()
    if db_error:
        return db_error

    requester_role = db.resolve_role(requester_patient_id)
    if requester_role is None:
        return {"error": "Requester patient_id not found; cannot verify authorization."}

    is_self = requester_patient_id == patient_id
    if not is_self and not _caller_is_clinical(requester_role):
        return {"error": "Not authorized to update this patient's record."}

    patient = db.get_patient(patient_id)
    if not patient:
        return {"error": "Patient not found"}

    existing: list[str] = list(patient.get(field, []))
    existing_lower = {v.strip().lower() for v in existing}
    added = []
    for value in add_values:
        cleaned = str(value).strip()
        if cleaned and cleaned.lower() not in existing_lower:
            existing.append(cleaned)
            existing_lower.add(cleaned.lower())
            added.append(cleaned)

    if not added:
        return {"ok": True, "added": [], "message": "Already on record — nothing new to add.", field: existing}

    db.update_patient(patient_id, {field: existing})
    return {"ok": True, "added": added, field: existing}


def _caller_is_staff(requester_role: str) -> bool:
    return requester_role in ("staff", "doctor")


def _caller_is_clinical(requester_role: str) -> bool:
    return requester_role == "doctor"


def _validate_date(date: Optional[str]) -> bool:
    if not date:
        return False
    try:
        datetime.fromisoformat(date)
        return True
    except ValueError:
        return False


def _overlaps(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    return start_a < end_b and start_b < end_a


def _parse_hhmm(value, default: str = "09:00"):
    """Tolerant HH:MM parser. Google Sheets can hand back working-hours cells
    in more than one shape depending on how they were typed/formatted
    (plain '09:00', 12-hour '9:00 AM', with seconds, etc) — silently
    defaulting to 9-5 on any parse failure previously meant a doctor whose
    real hours differ from the default could be mis-evaluated with no
    visible sign anything was wrong."""
    text = str(value).strip() if value not in (None, "") else default
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p", "%I:%M:%S %p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return datetime.strptime(default, "%H:%M").time()


def _doctor_working_window(doc: dict) -> tuple:
    """Return configured hours, or (None, None) when none were configured.

    Registration alone should make a doctor bookable. We only enforce a
    working-hours window when the clinic actually supplied both endpoints.
    """
    raw_start = str(doc.get("working_hours_start") or "").strip()
    raw_end = str(doc.get("working_hours_end") or "").strip()
    wh_start = _parse_hhmm(raw_start, "09:00") if raw_start else None
    wh_end = _parse_hhmm(raw_end, "17:00") if raw_end else None
    try:
        slot = int(doc.get("slot_minutes") or DEFAULT_SLOT_MINUTES)
    except (TypeError, ValueError):
        slot = DEFAULT_SLOT_MINUTES
    return wh_start, wh_end, slot


def _doctor_conflicts(doctor_id: str, start: datetime, end: datetime) -> bool:
    for appt in db.get_appointments(doctor_id=doctor_id, active_only=True):
        try:
            a_start = datetime.fromisoformat(str(appt["date"]))
        except (KeyError, ValueError, TypeError):
            continue
        a_end = a_start + timedelta(minutes=int(appt.get("duration_minutes") or DEFAULT_SLOT_MINUTES))
        if _overlaps(start, end, a_start, a_end):
            return True
    return False


MAX_SUGGESTION_DAYS_AHEAD = 7
MAX_SUGGESTIONS = 5


def _scan_next_slots(candidates: list, from_dt: datetime, duration_minutes: int,
                      limit: int = MAX_SUGGESTIONS) -> list[dict]:
    """Scans forward across real working hours and real existing bookings to
    find concrete next-available slots — so a patient whose requested time
    doesn't work gets actual alternative times to pick from, not just a
    list of the same (unavailable) doctors' names."""
    suggestions = []
    for day_offset in range(MAX_SUGGESTION_DAYS_AHEAD + 1):
        day = (from_dt + timedelta(days=day_offset)).date()
        for doc in candidates:
            wh_start, wh_end, slot = _doctor_working_window(doc)
            if wh_start is None or wh_end is None:
                # No clinic hours configured: preserve the patient's requested
                # clock time and look for the same time on successive days.
                cursor = datetime.combine(day, from_dt.time().replace(second=0, microsecond=0))
                day_end = cursor + timedelta(minutes=duration_minutes)
            else:
                cursor = datetime.combine(day, wh_start)
                day_end = datetime.combine(day, wh_end)
            while cursor + timedelta(minutes=duration_minutes) <= day_end:
                if cursor >= from_dt and not _doctor_conflicts(doc["patient_id"], cursor, cursor + timedelta(minutes=duration_minutes)):
                    suggestions.append({
                        "doctor_id": doc["patient_id"],
                        "doctor_name": doc.get("name"),
                        "date": cursor.isoformat(),
                    })
                    if len(suggestions) >= limit:
                        return suggestions
                cursor += timedelta(minutes=slot)
    return suggestions


def find_available_doctor(specialty: str, date: str, duration_minutes: int = DEFAULT_SLOT_MINUTES) -> dict:
    """
    Finds a doctor matching `specialty` who is free at `date` (ISO8601) for
    `duration_minutes`. Checks each candidate doctor's own working hours and
    existing appointments — never suggests a doctor of the wrong specialty,
    and never double-books one who already has something at that time.

    Returns {"available": True, "doctor": {...}} or, if nobody matches or
    everyone's booked, {"available": False, "message": "...", "reason": "...",
    "suggested_slots": [{"doctor_id", "doctor_name", "date"}, ...]}.
    `reason` is one of "no_doctors", "outside_working_hours", "fully_booked",
    or "mixed" — use it (and the real suggested_slots) rather than assuming
    every non-match means "busy".
    """
    db_error = _db_unavailable_response()
    if db_error:
        return {"available": False, "message": db_error["error"], "reason": "error", "suggested_slots": []}
    if not _validate_date(date):
        return {"available": False, "message": "date must be a valid ISO8601 string.", "reason": "error", "suggested_slots": []}

    candidates = db.find_doctors_by_specialty(specialty)
    if not candidates:
        all_specialties = sorted({d.get("specialty", "") for d in db.get_doctor_list() if d.get("specialty")})
        return {
            "available": False,
            "message": f"No doctor with specialty '{specialty}' on file.",
            "reason": "no_doctors",
            "known_specialties": all_specialties,
            "suggested_slots": [],
        }

    requested_start = datetime.fromisoformat(date)
    requested_end = requested_start + timedelta(minutes=duration_minutes)

    outside_hours_count = 0
    booked_count = 0
    for doc in candidates:
        wh_start, wh_end, _slot = _doctor_working_window(doc)
        if wh_start is not None and wh_end is not None:
            if not (wh_start <= requested_start.time() and requested_end.time() <= wh_end):
                outside_hours_count += 1
                continue
        if _doctor_conflicts(doc["patient_id"], requested_start, requested_end):
            booked_count += 1
            continue
        return {"available": True, "doctor": doc}

    suggested_slots = _scan_next_slots(candidates, requested_start, duration_minutes)
    if booked_count == 0 and outside_hours_count > 0:
        reason = "outside_working_hours"
        message = f"No {specialty} doctor is open at {date} — that time falls outside their working hours."
    elif outside_hours_count == 0 and booked_count > 0:
        reason = "fully_booked"
        message = f"All {specialty} doctors already have another appointment at {date}."
    else:
        reason = "mixed"
        message = f"No {specialty} doctor is both open and unbooked at {date}."

    if suggested_slots:
        message += " Here are the next real openings."
    else:
        message += " No upcoming openings were found in the next week — try a different specialty or contact the clinic directly."

    return {
        "available": False,
        "message": message,
        "reason": reason,
        "suggested_slots": suggested_slots,
    }


# Register as an MCP tool without replacing the callable Python function.
mcp.tool()(find_available_doctor)


def get_doctor_schedule(doctor_id: str, requester_patient_id: str) -> dict:
    """
    Returns the logged-in doctor's schedule from a fresh Google Sheets read.
    Matching is canonical-user-id based with compatibility for appointments
    created by older builds that stored the doctor's name instead.
    """
    db_error = _db_unavailable_response()
    if db_error:
        return {"authorized": False, "message": db_error["error"]}

    requested = str(requester_patient_id or "").strip()
    requested_doctor = db.get_doctor(requested)
    target_doctor = db.get_doctor(doctor_id)

    requester_role = db.resolve_role(requested)
    if requester_role is None:
        return {"authorized": False, "message": "Requester not found; cannot verify authorization."}

    requested_canonical = str((requested_doctor or {}).get("patient_id") or requested).strip().casefold()
    target_canonical = str((target_doctor or {}).get("patient_id") or doctor_id).strip().casefold()
    is_self = requested_canonical == target_canonical

    if not is_self and requester_role != "staff":
        return {"authorized": False, "message": "Only staff, or the doctor themself, may view this schedule."}

    appointments = db.get_doctor_schedule_appointments(
        target_canonical or doctor_id, active_only=True
    )
    appointments.sort(key=lambda a: a.get("date", ""))
    return {
        "authorized": True,
        "doctor_id": (target_doctor or {}).get("patient_id", doctor_id),
        "doctor_name": (target_doctor or {}).get("name", ""),
        "appointments": appointments,
    }


# Register as an MCP tool without replacing the callable Python function.
mcp.tool()(get_doctor_schedule)


def manage_appointment(
    action: str,
    patient_id: str,
    requester_patient_id: str,
    date: Optional[str] = None,
    specialist: Optional[str] = None,
    doctor_id: Optional[str] = None,
    appointment_id: Optional[str] = None,
    updates: Optional[dict] = None,
    reason: Optional[str] = None,
    consultation_mode: Optional[str] = None,
    contact_phone: Optional[str] = None,
) -> dict:
    """
    Handles appointment create/lookup/update/cancel with role validation and
    specialty + schedule-conflict matching on create. consultation_mode
    (video/phone/in-clinic) is optional on create and defaults to in-clinic.

    requester_patient_id's role is derived server-side (which sheet their ID
    is found in) — never accepted as a caller-supplied claim.
    """
    db_error = _db_unavailable_response()
    if db_error:
        return {"authorized": False, "message": db_error["error"]}

    if action not in VALID_APPOINTMENT_ACTIONS:
        return {"authorized": False, "message": f"Unknown action '{action}'. Valid: {sorted(VALID_APPOINTMENT_ACTIONS)}"}
    if not requester_patient_id or not requester_patient_id.strip():
        return {"authorized": False, "message": "requester_patient_id is required to verify authorization."}

    requester_role = db.resolve_role(requester_patient_id)
    if requester_role is None:
        return {"authorized": False, "message": "Requester patient_id not found; cannot verify authorization."}

    is_staff = _caller_is_staff(requester_role)
    if not is_staff and requester_patient_id != patient_id:
        return {"authorized": False, "message": "You may only manage your own appointments."}

    if action == "create":
        if not date or (not doctor_id and not specialist):
            return {"authorized": False, "message": "date and either doctor_id or specialist are required to create an appointment."}
        if not _validate_date(date):
            return {"authorized": False, "message": "date must be a valid ISO8601 string."}
        mode = (consultation_mode or "in-clinic").strip().lower()
        if mode not in VALID_CONSULTATION_MODES:
            return {"authorized": False, "message": f"consultation_mode must be one of {sorted(VALID_CONSULTATION_MODES)}"}

        # When the UI supplies doctor_id, book that exact registered doctor.
        # This avoids ambiguous/duplicate names and does not require a doctor
        # to have a specialty configured in the optional doctors worksheet.
        selected_doctor = None
        if doctor_id:
            selected_doctor = db.get_doctor(doctor_id)
            if not selected_doctor:
                return {
                    "authorized": True, "status": "no_availability",
                    "message": "That doctor is no longer present in the registered doctor accounts. Please refresh the doctor list.",
                    "reason": "doctor_not_found", "doctor_id": doctor_id, "suggested_slots": [],
                }
            duplicate = next(
                (a for a in db.get_appointments(patient_id=patient_id)
                 if a.get("date") == date and str(a.get("doctor_id", "")).casefold() == str(doctor_id).casefold()),
                None,
            )
            if duplicate:
                duplicate = dict(duplicate)
                duplicate["assigned_doctor"] = selected_doctor.get("name")
                return {
                    "authorized": True, "status": "already_exists",
                    "appointment_id": duplicate["appointment_id"], "details": duplicate,
                    "message": "An appointment with the same doctor, date, and patient already exists.",
                }
            match = find_available_doctor(doctor_id, date)
        else:
            duplicate = next(
                (a for a in db.get_appointments(patient_id=patient_id)
                 if a.get("date") == date and str(a.get("specialist", "")).casefold() == str(specialist or "").casefold()),
                None,
            )
            if duplicate:
                return {
                    "authorized": True, "status": "already_exists",
                    "appointment_id": duplicate["appointment_id"], "details": duplicate,
                    "message": "An appointment with the same date, specialist, and patient already exists.",
                }
            match = find_available_doctor(specialist, date)

        if not match.get("available"):
            return {
                "authorized": True, "status": "no_availability",
                "message": match.get("message"),
                "reason": match.get("reason"),
                "specialist": specialist or (selected_doctor or {}).get("specialty", ""),
                "doctor_id": doctor_id,
                "suggested_slots": match.get("suggested_slots", []),
            }

        doctor = match["doctor"]
        persisted_specialist = (
            str(doctor.get("specialty") or "").strip()
            or str(specialist or "").strip()
            or str(doctor.get("name") or doctor.get("patient_id") or "").strip()
        )

        appt = db.create_appointment(
            patient_id=patient_id,
            doctor_id=doctor["patient_id"],
            doctor_name=doctor.get("name", ""),
            specialist=persisted_specialist,
            date=date,
            reason=reason or "",
            consultation_mode=mode,
            contact_phone=contact_phone or "",
        )
        appt["assigned_doctor"] = doctor.get("name")
        return {"authorized": True, "status": "created", "appointment_id": appt["appointment_id"], "details": appt}

    if action == "lookup":
        # Patients only ever see their own; staff/doctor may look up by
        # specialist/date filters across everyone (doctor's own-schedule
        # restriction for browsing OTHER doctors is enforced in
        # get_doctor_schedule, not here — this path is patient-appointment
        # lookup, which staff/doctor are allowed to filter broadly).
        appts = db.get_appointments(patient_id=None if is_staff else patient_id)
        if specialist:
            appts = [a for a in appts if a.get("specialist", "").lower() == specialist.lower()]
        if date:
            appts = [a for a in appts if a.get("date") == date]
        appts.sort(key=lambda a: a.get("date", ""))
        return {"authorized": True, "status": "lookup_completed", "appointments": appts}

    if action == "update":
        if not appointment_id or not updates:
            return {"authorized": False, "message": "appointment_id and updates are required."}
        # Patients may reschedule/change mode on their own appointments.
        # Status changes and other admin-level changes remain staff-only.
        patient_allowed_fields = {"date", "consultation_mode", "contact_phone", "reason"}
        update_keys = set(updates.keys())
        requires_staff = bool(update_keys - patient_allowed_fields)
        if requires_staff and not is_staff:
            return {"authorized": False, "message": "Only staff/clinician may update appointment status or other admin fields."}
        if not is_staff:
            # Extra safety: confirm the appointment belongs to this patient.
            target_check = next((a for a in db.get_appointments(patient_id=patient_id, active_only=False)
                                  if a.get("appointment_id") == appointment_id), None)
            if not target_check:
                return {"authorized": False, "message": "Appointment not found or does not belong to you."}
        if "status" in updates and updates["status"] not in VALID_STATUSES:
            return {"authorized": False, "message": f"status must be one of {sorted(VALID_STATUSES)}"}
        # Validate new date if provided.
        if "date" in updates:
            if not _validate_date(updates["date"]):
                return {"authorized": False, "message": "date must be a valid ISO8601 string."}
            new_start = datetime.fromisoformat(updates["date"])
            if new_start < datetime.now():
                return {"authorized": False, "message": "Cannot reschedule to a date/time in the past."}
        if "consultation_mode" in updates:
            mode = updates["consultation_mode"].strip().lower()
            if mode not in VALID_CONSULTATION_MODES:
                return {"authorized": False, "message": f"consultation_mode must be one of {sorted(VALID_CONSULTATION_MODES)}"}
            updates["consultation_mode"] = mode
        ok = db.update_appointment(appointment_id, updates)
        return {"authorized": True, "status": "updated" if ok else "not_found"}

    if action == "cancel":
        if not appointment_id:
            return {"authorized": False, "message": "appointment_id is required to cancel."}
        target = next((a for a in db.get_appointments(patient_id=patient_id, active_only=False)
                        if a.get("appointment_id") == appointment_id), None)
        if not target:
            return {"authorized": True, "status": "not_found"}
        ok = db.update_appointment(appointment_id, {"status": "cancelled"})
        return {"authorized": True, "status": "cancelled" if ok else "not_found"}

    return {"authorized": False, "message": "Unhandled appointment action."}


# Register as an MCP tool without replacing the callable Python function.
mcp.tool()(manage_appointment)


def clear_appointments(patient_id: str) -> dict:
    """Cancels all active appointments for the given patient_id."""
    if not patient_id or not patient_id.strip():
        return {"error": "patient_id is required"}
    if db.get_patient(patient_id) is None:
        return {"error": "Patient not found"}
    appts = db.get_appointments(patient_id=patient_id, active_only=True)
    for a in appts:
        db.update_appointment(a["appointment_id"], {"status": "cancelled"})
    return {"message": "All active appointments for the patient have been cancelled.", "cancelled_count": len(appts)}


# Register as an MCP tool without replacing the callable Python function.
mcp.tool()(clear_appointments)


if __name__ == "__main__":
    mcp.run(show_banner=False)