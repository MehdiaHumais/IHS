# IHS · Smart CDSS

AI-powered clinical decision-support platform: a multi-module diagnostic suite
plus a clinical chatbot, delivered as one Streamlit experience.

## Modules

| Module | Entry point | What it does |
|---|---|---|
| Smart Clinic (main shell) | `app_diagnostics.py` | Login, dashboard, routing for every module, patient search |
| Clinical Chatbot | `clinical_chatbot/` (FastAPI + `index.html`, embedded via iframe) | Symptom triage, appointments, lab-report explanation |
| General / Agentic Diagnostics | `general_diagnostics/app.py` (embedded via iframe) | Document & image analysis, diagnostic reports |
| Brain Stroke | `stroke_agent/` | Stroke risk prediction |
| Heart Disease | `heart_disease/` | Heart-disease prediction |
| Breast Cancer | `breast_cancer/` | Breast-cancer prediction |

`app_patient_diagnostics_fixed.py` is a patient-facing variant of the main shell.

## Quick start (local)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .streamlit\secrets.example.toml .streamlit\secrets.toml
# edit .streamlit\secrets.toml and add at least one AI provider key

streamlit run app_diagnostics.py
```

The app opens at http://localhost:8501.

## Configuration & secrets

All credentials live outside the repo:

- `.streamlit/secrets.toml` — Streamlit Cloud / local secrets (**git-ignored**).
  Start from the committed template `.streamlit/secrets.example.toml`.
- `clinical_chatbot/.env` — local chatbot provider keys (**git-ignored**).
- `service_account.json` — Google service-account key (**git-ignored**).
- `google_config.json` — Google Sheet routing (tracked; contains no secret).

At least one of `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `GEMINI_API_KEY`,
`OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) is required for AI answers.

> Never commit `.streamlit/secrets.toml`, `service_account.json` or any `.env`.

## Deployment (Streamlit Community Cloud)

1. Push to the IHS GitHub repository (see below).
2. Create the app pointing at `app_diagnostics.py`.
3. Paste the contents of `secrets.toml` into **Settings → Secrets**.
4. Deploy. Use a separate app for the sandbox environment.

## GitHub organization handoff

The repo currently lives on a personal account. To move it under the IHS
organization and open it to the team:

```powershell
# 1. Transfer ownership (do this from the GitHub web UI: Settings → Transfer)
#    Target: the IHS organization. Requires admin rights on both sides.

# 2. Once transferred, set the new remote locally:
git remote set-url origin https://github.com/<IHS-ORG>/IHS.git
```

Recommended org settings:

- **Teams & access** — create a `ihs-core` team with Write access; grant
  contractors access only to the sandbox repository/branch.
- **Branch protection** on `main` — require PR review, disallow direct pushes.
- **Sandbox** — either a second repository (`IHS-sandbox`) or a long-lived
  `sandbox` branch. Pilot fixes land on `main`; the sandbox is synced in
  parallel (merge `main` → `sandbox`).
- **Secrets** — store deploy keys/API keys in GitHub **Environment** secrets,
  never in the repo.

## Repository hygiene

- Large binaries (`*.pkl`, `*.joblib`, images, archives) are tracked with
  **Git LFS** via `.gitattributes`. Run `git lfs install` once per machine.
- `stroke_agent/stroke_model.pkl` (~45 MB) and the diagnostic models are
  required for deployment.

Optional one-time cleanup (rewrites history — coordinate with the team first):

```powershell
# Move already-committed large binaries into LFS
git lfs migrate import --include="*.pkl,*.joblib,*.png,*.webp,*.db" --everything

# Stop tracking generated artifacts that are now git-ignored
git rm -r --cached logs general_diagnostics/agentic_diagnostic_server.log
git rm --cached **/medical_data.db **/stroke_history.db
```

## Disclaimer

Research prototype. All output must be reviewed by a qualified clinician before
any clinical decision.
