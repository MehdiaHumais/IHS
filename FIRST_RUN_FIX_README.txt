SMART CLINIC - FIRST RUN LAUNCHER FIX

What was fixed:
- Removed the packaged .venv from the distributable. A virtual environment is machine-specific and must not be copied between PCs.
- Removed stale install-marker behavior that could falsely report packages as already installed.
- RUN_SMART_CDSS.bat now validates the local virtual environment before trusting it.
- If an old/broken .venv exists, the launcher deletes and recreates it automatically.
- Required imports are checked every launch; missing packages trigger an automatic repair install.
- RUN_APP.bat and the other wrapper launchers now start the real application file through RUN_SMART_CDSS.bat.
- The launcher keeps the console open and prints useful errors if installation or Streamlit startup fails.
- The main browser interface opens only after Streamlit reports healthy.

How to run:
1. Extract the ZIP completely to a normal folder.
2. Double-click RUN_SMART_CDSS.bat (recommended), or RUN_APP.bat.
3. On the first run, keep internet access enabled while packages install.
4. The browser should open automatically at http://127.0.0.1:8503.

Python:
- Python 3.10+ is required.
- If Python is missing, install it and enable "Add python.exe to PATH".

Important:
- Do not copy a .venv folder from another computer into this project.
