"""
Per-session state, the way ChatGPT/Claude keep a "conversation" separate
from your permanent account data.

Deliberately in-memory (a plain dict guarded by a lock), NOT written to the
Google Sheet:
  - Session data (rolling chat history, this-session's cached answers) is
    high-frequency and short-lived — writing it to Sheets on every turn
    would add real latency (gspread round-trips + the retry/backoff logic
    in database.py) for something that doesn't need to survive a server
    restart.
  - Everything that DOES need to persist (intake answers once actually
    submitted, appointments, medical history) already goes through
    database.py / agents.submit_intake_answers exactly as before. This
    module only holds the *scratch space* on the way there.

If you later need sessions to survive a server restart/redeploy (e.g. once
this is deployed somewhere that restarts often), the natural upgrade is a
small SQLite file or a Redis instance — the interface below (create/get/
touch/append_turn/cache_answer) would not need to change at the call site,
only this file's internals.
"""

import threading
import time
import uuid
from typing import Optional

# How long a session can sit idle before it's treated as gone. 4 hours
# comfortably covers "closed the tab and came back after lunch" without
# keeping abandoned sessions around forever.
SESSION_TTL_SECONDS = 4 * 60 * 60

# Cap on how many chat turns we keep per session. This is a memory/prompt-
# size guard, not a product decision — old turns just age out; nothing is
# lost from the patient's permanent record, since that's in the Sheet.
MAX_HISTORY_TURNS = 30

_lock = threading.Lock()
_sessions: dict[str, dict] = {}


def create_session(patient_id: str) -> str:
    """Starts a new session for this patient and returns its session_id."""
    session_id = uuid.uuid4().hex
    with _lock:
        _sessions[session_id] = {
            "patient_id": patient_id,
            "created_at": time.time(),
            "last_active": time.time(),
            "history": [],           # [{"role": "user"|"assistant", "content": str}, ...]
            "profile_answers": {},   # {field_id: answer} — this-session intake cache, see cache_answer()
        }
    return session_id


def _expire_if_stale(session_id: str) -> Optional[dict]:
    """Caller must hold _lock. Returns the session dict, or None (and evicts
    it) if it doesn't exist or has expired."""
    session = _sessions.get(session_id)
    if session is None:
        return None
    if time.time() - session["last_active"] > SESSION_TTL_SECONDS:
        _sessions.pop(session_id, None)
        return None
    return session


def get_session(session_id: str) -> Optional[dict]:
    """Returns a shallow copy of the session, or None if it doesn't exist /
    has expired. Callers should treat this as read-only — use the mutator
    functions below (append_turn, cache_answer, touch) to actually change
    session state, not by mutating what this returns."""
    if not session_id:
        return None
    with _lock:
        session = _expire_if_stale(session_id)
        if session is None:
            return None
        return {
            "patient_id": session["patient_id"],
            "created_at": session["created_at"],
            "last_active": session["last_active"],
            "history": list(session["history"]),
            "profile_answers": dict(session["profile_answers"]),
        }


def touch(session_id: str) -> bool:
    """Marks the session as recently active (resets its TTL clock) without
    changing anything else. Returns False if the session doesn't exist."""
    with _lock:
        session = _expire_if_stale(session_id)
        if session is None:
            return False
        session["last_active"] = time.time()
        return True


def append_turn(session_id: str, role: str, content: str) -> bool:
    """Appends one chat turn to the session's rolling history. role should
    be 'user' or 'assistant'. Returns False if the session doesn't exist."""
    with _lock:
        session = _expire_if_stale(session_id)
        if session is None:
            return False
        session["history"].append({"role": role, "content": content})
        if len(session["history"]) > MAX_HISTORY_TURNS:
            session["history"] = session["history"][-MAX_HISTORY_TURNS:]
        session["last_active"] = time.time()
        return True


def cache_answer(session_id: str, field_id: str, value) -> bool:
    """Caches one intake/profile answer for the rest of this session, so it
    is never asked twice in the same conversation even before it's been
    written to the patient's permanent record. Returns False if the
    session doesn't exist."""
    with _lock:
        session = _expire_if_stale(session_id)
        if session is None:
            return False
        session["profile_answers"][field_id] = value
        session["last_active"] = time.time()
        return True


def get_cached_answers(session_id: str) -> dict:
    session = get_session(session_id)
    return session["profile_answers"] if session else {}


def get_history(session_id: str) -> list[dict]:
    session = get_session(session_id)
    return session["history"] if session else []


def cleanup_expired() -> int:
    """Sweeps out expired sessions. Not required for correctness (every
    read already self-expires via _expire_if_stale), but calling this
    periodically (e.g. from a background task) keeps the dict from
    growing unbounded with abandoned sessions between real reads."""
    now = time.time()
    with _lock:
        expired = [sid for sid, s in _sessions.items() if now - s["last_active"] > SESSION_TTL_SECONDS]
        for sid in expired:
            _sessions.pop(sid, None)
        return len(expired)