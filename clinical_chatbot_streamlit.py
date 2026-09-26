import asyncio
import importlib.util
import os
import sys
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

_BRIDGE_DIR = Path(__file__).resolve().parent
_CHATBOT_DIR = _BRIDGE_DIR / "clinical_chatbot"

_LLM_ENV_KEYS = [
    "GROQ_API_KEY", "GROQ_MODEL",
    "CEREBRAS_API_KEY", "CEREBRAS_MODEL",
    "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_VISION_MODEL",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "OLLAMA_BASE_URL", "OLLAMA_MODEL",
]

_ASYNC_LOOP = None
_cached = None


def _ensure_cloud_env() -> None:
    try:
        cfg = st.secrets.get("google_config", {})
    except Exception:
        cfg = {}
    gs = cfg.get("google_sheet", {}) if isinstance(cfg, dict) else {}
    if isinstance(gs, dict) and gs.get("spreadsheet_id"):
        os.environ.setdefault("GOOGLE_SHEET_ID", str(gs["spreadsheet_id"]))
    for key in _LLM_ENV_KEYS:
        if os.environ.get(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)


def _resolve_service_account_file() -> str:
    existing = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "")
    if existing and os.path.isfile(existing):
        return existing
    candidates = [
        _BRIDGE_DIR / "service_account.json",
        _CHATBOT_DIR / "service_account.json",
    ]
    for path in candidates:
        if path.is_file():
            os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"] = str(path)
            return str(path)
    try:
        sa = st.secrets.get("service_account")
    except Exception:
        sa = None
    if sa:
        payload = dict(sa)
        fd, path = tempfile.mkstemp(suffix="-service-account.json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"] = path
        return path
    return ""


def _load_chatbot_agents():
    """Load clinical_chatbot/agents.py under a namespaced module name.

    The chatbot file and the general_diagnostics agents package both want the
    bare name ``agents``; registering ours as ``clinical_chatbot.agents`` keeps
    ``sys.modules["agents"]`` free so the Full Diagnostics page imports the
    correct package no matter which page ran first in the session."""
    spec = importlib.util.spec_from_file_location(
        "clinical_chatbot.agents", str(_CHATBOT_DIR / "agents.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_chatbot_ready() -> dict:
    global _cached
    if _cached is not None:
        return _cached
    os.environ["CDSS_DIRECT_TOOLS"] = "1"
    _ensure_cloud_env()
    _resolve_service_account_file()
    chatbot_dir = str(_CHATBOT_DIR)
    if chatbot_dir not in sys.path:
        sys.path.insert(0, chatbot_dir)
    import database as cb_database
    import mcp_tools as cb_tools
    cb_agents = _load_chatbot_agents()
    _cached = {
        "database": cb_database,
        "mcp_tools": cb_tools,
        "agents": cb_agents,
    }
    return _cached


def _run_async(coro):
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
        _ASYNC_LOOP = asyncio.new_event_loop()
    return _ASYNC_LOOP.run_until_complete(coro)


def _init_state() -> None:
    st.session_state.setdefault("cb_messages", [])
    st.session_state.setdefault("cb_pending", None)
    st.session_state.setdefault("cb_original", "")
    st.session_state.setdefault("cb_clarify_round", 0)
    st.session_state.setdefault("cb_visit_done", False)
    st.session_state.setdefault("cb_patient_id", None)
    st.session_state.setdefault("cb_patient_name", "")
    st.session_state.setdefault("cb_appt_form", False)


def _is_doctor() -> bool:
    return str(
        st.session_state.get("current_user", {}).get("user_type", "")
    ).strip().casefold() == "doctor"


def _patient_display_name(pid: str) -> str:
    return st.session_state.get("cb_patient_name") or pid


def _resolve_patient(cb: dict) -> tuple[str, str]:
    pid = st.session_state.cb_patient_id
    current_user = st.session_state.get("current_user", {})
    fallback_name = str(current_user.get("full_name") or current_user.get("user_id") or "")
    if not pid:
        pid = str(current_user.get("user_id") or "")
        if not pid:
            pid = "patient-self"
        st.session_state.cb_patient_id = pid
        st.session_state.cb_patient_name = fallback_name or _patient_display_name(pid)

    selected_pid = None
    if _is_doctor():
        try:
            patients = cb["mcp_tools"].get_patient_list()
        except Exception:
            patients = []
        if patients:
            chooser = {}
            for row in patients:
                row_id = str(row.get("patient_id") or "")
                row_name = str(row.get("name") or row_id)
                if row_id:
                    chooser[f"{row_name} ({row_id})"] = row_id
            if chooser:
                current_key = f"{_patient_display_name(pid)} ({pid})"
                if current_key not in chooser:
                    current_key = list(chooser)[0]
                picked = st.selectbox("Patient", list(chooser), index=list(chooser).index(current_key), key="cb_patient_picker")
                selected_pid = chooser[picked]
                if selected_pid != pid:
                    st.session_state.cb_patient_id = selected_pid
                    st.session_state.cb_patient_name = picked.rsplit(" (", 1)[0]
                    st.session_state.cb_messages = []
                    st.session_state.cb_pending = None
                    st.session_state.cb_original = ""
                    st.session_state.cb_clarify_round = 0
                    st.session_state.cb_visit_done = False
                    pid = selected_pid

    patient = cb["database"].get_patient(pid)
    if not patient:
        cb["database"].create_patient(pid, _patient_display_name(pid), age="", gender="")
        patient = cb["database"].get_patient(pid)
    return pid, (patient or {})


def _render_intakes(cb: dict, pid: str, patient: dict) -> None:
    agents = cb["agents"]
    profile_needed = str(patient.get("intake_completed", "")).strip().lower() != "true"

    if profile_needed:
        with st.expander("📋 Your health profile", expanded=True):
            st.caption("These are saved once so we don't ask every time.")
            profile_answers = _build_question_form(agents.PROFILE_QUESTIONS, "cb_profile")
            if st.button("Save profile", key="cb_profile_save", use_container_width=True):
                if profile_answers:
                    agents.submit_intake_answers(pid, profile_answers, "profile")
                    st.success("Profile saved.")
                    st.rerun()

    if not st.session_state.cb_visit_done:
        with st.expander("🩺 About today's visit", expanded=True):
            st.caption("Fills in your visit context so the chat can tailor its assessment.")
            visit_answers = _build_question_form(agents.VISIT_QUESTIONS, "cb_visit")
            if st.button("Save visit details", key="cb_visit_save", use_container_width=True):
                if visit_answers:
                    agents.submit_intake_answers(pid, visit_answers, "visit")
                    st.session_state.cb_visit_done = True
                    st.success("Visit details saved.")
                    st.rerun()
            else:
                st.info("You can skip this and describe your visit directly in the chat box.")


def _build_question_form(questions: list, prefix: str) -> dict:
    answers = {}
    for question in questions:
        question_id = question["id"]
        kind = question.get("type", "text")
        label = question.get("question", question_id)
        key = f"{prefix}_{question_id}"
        if kind == "number":
            value = st.number_input(label, min_value=0, max_value=130, step=1, key=key)
            answers[question_id] = str(int(value))
        elif kind == "select":
            options = question.get("options") or []
            value = st.selectbox(label, options, key=key)
            answers[question_id] = value
        elif kind == "yesno":
            value = st.radio(label, ["Yes", "No"], horizontal=True, key=key)
            answers[question_id] = value
        elif kind == "list":
            value = st.text_area(label, placeholder="One per line (or comma-separated)", key=key)
            cleaned = [item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()]
            answers[question_id] = cleaned
        else:
            value = st.text_input(label, key=key)
            answers[question_id] = value
    if not any(str(v).strip() or (isinstance(v, list) and v) for v in answers.values()):
        return {}
    return answers


def _render_chat_history() -> None:
    for message in st.session_state.get("cb_messages", []):
        role = message.get("role", "assistant")
        content = str(message.get("content", ""))
        source = message.get("source")
        with st.chat_message(role):
            if source == "lab_report":
                st.markdown(f"**📄 Lab report review**\n\n{content}")
            elif role == "assistant":
                st.markdown(content)
            else:
                st.markdown(content)


def _append_message(role: str, content: str, source: str = "chat") -> None:
    st.session_state.cb_messages.append(
        {"role": role, "content": str(content), "source": source}
    )


def _history() -> list:
    return list(st.session_state.get("cb_messages", []))


def _handle_flow_result(cb: dict, pid: str, result: dict) -> None:
    route = result.get("route") or "meta"
    text = str(result.get("display_text") or "").strip()
    if not text:
        text = "Thank you. Is there anything else I can help with?"
    if route == "clarify":
        st.session_state.cb_pending = {"route": "clarify"}
        _append_message("assistant", text)
    elif route == "ask_medication":
        st.session_state.cb_pending = {
            "route": "ask_medication",
            "specialist": result.get("offer_appointment_specialist") or result.get("recommended_specialist") or "General Physician",
        }
        _append_message("assistant", text)
    elif route == "self_care":
        st.session_state.cb_pending = {"route": "appointment_offer"}
        _append_message("assistant", text)
    elif route == "appointment_management":
        appointments = result.get("appointments") or []
        st.session_state.cb_pending = {"route": "appointment_management", "appointments": appointments}
        _append_message("assistant", text)
    elif route == "intent_clarify":
        st.session_state.cb_pending = {"route": "intent_clarify"}
        _append_message("assistant", text)
    elif route == "appointment_booked":
        st.session_state.cb_pending = None
        _append_message("assistant", text)
    elif route == "emergency":
        st.session_state.cb_pending = {"route": "emergency"}
        _append_message("assistant", f"**⚠️ {text}**")
    elif route == "error":
        st.session_state.cb_pending = None
        _append_message("assistant", text)
    else:
        st.session_state.cb_pending = None
        _append_message("assistant", text)


def _process_message(cb: dict, pid: str, text: str) -> None:
    text = text.strip()
    if not text:
        return
    _append_message("user", text)
    agents = cb["agents"]
    pending = st.session_state.cb_pending
    try:
        with st.spinner("Analysing your symptoms..."):
            if pending and pending.get("route") == "clarify":
                result = _run_async(
                    agents.run_patient_symptom_flow(
                        pid, st.session_state.cb_original,
                        clarifying_answer=text,
                        clarify_round=int(st.session_state.cb_clarify_round),
                        session_history=_history(),
                    )
                )
                st.session_state.cb_clarify_round = int(st.session_state.cb_clarify_round) + 1
            else:
                st.session_state.cb_original = text
                st.session_state.cb_clarify_round = 0
                st.session_state.cb_appt_form = False
                result = _run_async(
                    agents.run_patient_symptom_flow(pid, text, None, 0, session_history=_history())
                )
    except Exception as exc:
        result = {
            "route": "error",
            "display_text": "Sorry, the chat service could not be reached right now. Please try again in a moment.",
        }
        st.session_state._cb_last_error = str(exc)
    st.session_state.cb_pending = None
    _handle_flow_result(cb, pid, result)


def _iterable_label(option):
    if isinstance(option, dict):
        return option.get("label") or option.get("id") or str(option)
    return str(option)


def _render_pending_ui(cb: dict, pid: str) -> None:
    agents = cb["agents"]
    pending = st.session_state.cb_pending
    if not pending:
        return
    route = pending.get("route")

    if route == "clarify":
        with st.container():
            answer = st.text_area("Your answer", key="cb_clarify_input", placeholder="Type your answer here")
            if st.button("Send answer", key="cb_clarify_send", use_container_width=True) and answer.strip():
                _process_message(cb, pid, answer)
                st.rerun()

    elif route == "ask_medication":
        st.markdown("**Would you like medication and self-care recommendations as well?**")
        yes_col, no_col = st.columns(2)
        with yes_col:
            if st.button("Yes, please", key="cb_med_yes", use_container_width=True):
                complaint = st.session_state.cb_original
                specialist = pending.get("specialist") or "General Physician"
                with st.spinner("Checking OTC and self-care options..."):
                    result = _run_async(agents.run_patient_medication_then_appointment(pid, complaint, specialist))
                _handle_flow_result(cb, pid, result)
                st.rerun()
        with no_col:
            if st.button("No thanks", key="cb_med_no", use_container_width=True):
                _append_message("assistant", "Understood — I'll skip medication suggestions. You can ask anytime.")
                st.session_state.cb_pending = {"route": "appointment_offer"}
                st.rerun()

    elif route == "appointment_offer":
        st.markdown("**Would you like to book an appointment?**")
        yes_col, no_col = st.columns(2)
        with yes_col:
            if st.button("Book appointment", key="cb_appt_yes", use_container_width=True):
                st.session_state.cb_appt_form = True
                st.rerun()
        with no_col:
            if st.button("Not right now", key="cb_appt_no", use_container_width=True):
                _append_message("assistant", "No problem. Let me know if you need anything else.")
                st.session_state.cb_pending = None
                st.rerun()

    elif route == "appointment_management":
        appointments = pending.get("appointments") or []
        if appointments:
            for appt in appointments:
                appt_id = str(appt.get("appointment_id") or "")
                date_val = str(appt.get("date") or "")
                specialist_val = str(appt.get("specialist") or "")
                doctor_val = str(appt.get("doctor_name") or "")
                status_val = str(appt.get("status") or "")
                line = f"{date_val} — {specialist_val or doctor_val} ({status_val}) [ID: {appt_id}]"
                st.markdown(f"- {line}")
        if st.button("Book a new appointment", key="cb_appt_manage_new", use_container_width=True):
            st.session_state.cb_appt_form = True
            st.rerun()
        if appointments:
            for appt in appointments:
                appt_id = str(appt.get("appointment_id") or "")
                date_val = str(appt.get("date") or "")
                if st.button(f"Cancel: {date_val}", key=f"cb_cancel_{appt_id}", use_container_width=True):
                    with st.spinner("Cancelling appointment..."):
                        result = _run_async(
                            agents.run_appointment_direct(
                                patient_id=pid, action="cancel", appointment_id=appt_id
                            )
                        )
                    _handle_flow_result(cb, pid, result)
                    st.rerun()

    elif route == "intent_clarify":
        st.markdown("**Just to make sure I help with the right thing:**")
        options = [
            ("symptom", "A symptom or health concern"),
            ("appointment_management", "An existing appointment"),
            ("other", "Something else"),
        ]
        for value, label in options:
            if st.button(label, key=f"cb_intent_{value}", use_container_width=True):
                result = _run_async(
                    agents.run_patient_symptom_flow(
                        pid, st.session_state.cb_original, forced_intent=value, session_history=_history()
                    )
                )
                _handle_flow_result(cb, pid, result)
                st.rerun()

    elif route == "emergency":
        st.error("If this is an emergency, please call emergency services immediately.")


def _render_appointment_form(cb: dict, pid: str) -> None:
    if not st.session_state.get("cb_appt_form"):
        return
    agents = cb["agents"]
    with st.expander("📅 Book appointment", expanded=True):
        today = datetime.now()
        date_picked = st.date_input("Date", value=today, min_value=today, key="cb_appt_date")
        slot_options = []
        current = datetime.combine(date_picked, datetime.strptime("09:00", "%H:%M").time())
        end = datetime.combine(date_picked, datetime.strptime("17:00", "%H:%M").time())
        while current < end:
            slot_options.append(current.strftime("%H:%M"))
            current += timedelta(minutes=30)
        time_picked = st.selectbox("Time", slot_options, key="cb_appt_time")
        iso_date = f"{date_picked.isoformat()}T{time_picked}:00"

        doctors = []
        try:
            doctors = cb["mcp_tools"].get_doctor_list()
        except Exception:
            doctors = []
        doctor_labels = {"Auto (match any available doctor)": ""}
        for doctor in doctors:
            doc_id = str(doctor.get("patient_id") or doctor.get("doctor_id") or "")
            doc_name = str(doctor.get("name") or doc_id)
            doc_spec = str(doctor.get("specialty") or "")
            label = f"{doc_name} ({doc_spec})" if doc_spec else doc_name
            doctor_labels[label] = doc_id
        doctor_choice = st.selectbox("Doctor", list(doctor_labels), key="cb_appt_doctor")
        doctor_id = doctor_labels[doctor_choice]

        specialist_default = "General Physician"
        specialist = st.text_input("Specialist", value=specialist_default, key="cb_appt_specialist")

        mode = st.selectbox("Consultation mode", ["in-clinic", "video", "phone"], key="cb_appt_mode")
        phone = ""
        if mode in ("video", "phone"):
            phone = st.text_input("Contact phone (required for video/phone)", key="cb_appt_phone")
            specialist_ok = True
        else:
            specialist_ok = True

        reason = st.text_area("Reason (optional)", key="cb_appt_reason")

        if st.button("Confirm booking", key="cb_appt_confirm", type="primary", use_container_width=True):
            if mode in ("video", "phone") and not phone.strip():
                st.error("Please provide a contact phone for video/phone consultation.")
                return
            with st.spinner("Booking the appointment..."):
                result = _run_async(
                    agents.run_appointment_direct(
                        patient_id=pid,
                        action="create",
                        date=iso_date,
                        specialist=specialist,
                        doctor_id=doctor_id or None,
                        reason=reason,
                        consultation_mode=mode,
                        contact_phone=phone if mode in ("video", "phone") else None,
                    )
                )
            st.session_state.cb_appt_form = False
            _handle_flow_result(cb, pid, result)
            if result.get("route") != "appointment_booked":
                st.session_state.cb_pending = {"route": "appointment_offer"}
            st.rerun()


def _render_lab_upload(cb: dict, pid: str) -> None:
    agents = cb["agents"]
    with st.expander("📎 Upload a lab report for explanation", expanded=False):
        uploaded = st.file_uploader(
            "Choose a PDF or image",
            type=["pdf", "jpg", "jpeg", "png", "webp"],
            key="cb_lab_upload",
        )
        if uploaded is not None:
            if st.button("Explain this lab report", key="cb_lab_explain", use_container_width=True):
                data = uploaded.getvalue()
                name = uploaded.name
                mime = uploaded.type or "application/octet-stream"
                with st.spinner("Reading the lab report..."):
                    result = _run_async(agents.run_lab_report_agent(pid, data, name, mime))
                detail = result.get("display_text") or result.get("error") or "Could not process the report."
                _append_message("assistant", detail, source="lab_report")
                summary = None
                try:
                    summary = agents._lab_report_history_summary(result)
                except Exception:
                    summary = None
                if summary:
                    _append_message("assistant", summary, source="lab_report")
                st.session_state.cb_pending = None
                st.rerun()


def _configured_provider_names() -> list:
    mapping = {
        "GROQ_API_KEY": "Groq",
        "CEREBRAS_API_KEY": "Cerebras",
        "GEMINI_API_KEY": "Gemini",
        "OPENROUTER_API_KEY": "OpenRouter",
    }
    names = []
    for key, name in mapping.items():
        value = os.environ.get(key) or os.environ.get(key.replace("_API_KEY", "_KEY")) or ""
        if value:
            names.append(name)
    if os.environ.get("OLLAMA_BASE_URL"):
        names.append("Ollama")
    return names


def render_clinical_chatbot_page() -> None:
    _init_state()

    st.markdown("### 💬 Clinical Chatbot")
    display_name = str(st.session_state.get("current_user", {}).get("full_name", "User"))
    st.caption(f"Signed in as {display_name}")

    if st.button("← Back to dashboard", key="cb_back_dashboard"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    try:
        cb = ensure_chatbot_ready()
    except Exception as exc:
        st.error("The chatbot database layer could not be loaded.")
        st.code(str(exc))
        return

    if not cb["database"].is_available():
        st.error(
            "Google Sheets is unreachable for the chatbot. "
            + str(cb["database"].DB_ERROR or "")
        )
        return

    configured = _configured_provider_names()
    if not configured:
        st.warning(
            "No AI provider is configured yet. Add GROQ_API_KEY, CEREBRAS_API_KEY, "
            "GEMINI_API_KEY or OPENROUTER_API_KEY to your secrets (or the clinical_chatbot/.env "
            "file locally). Basic responses and safety checks still work, but full AI answers "
            "need one of these keys."
        )

    pid, patient = _resolve_patient(cb)

    tab_label = f"Chatting as **{_patient_display_name(pid) or pid}**"
    st.caption(tab_label)

    _render_intakes(cb, pid, patient)
    st.divider()

    prompt = st.chat_input("Describe your symptoms or ask a health question…")
    if prompt:
        _process_message(cb, pid, prompt)
        st.rerun()

    _render_chat_history()
    _render_pending_ui(cb, pid)
    _render_appointment_form(cb, pid)
    _render_lab_upload(cb, pid)

    if getattr(st.session_state, "_cb_last_error", None):
        with st.expander("Technical error"):
            st.code(str(st.session_state._cb_last_error))