SMART CLINIC – FULL DIAGNOSTICS INTEGRATION

Correct layout:
SMART-CDSS/
  app.py                         <- updated Smart Clinic interface
  RUN_SMART_CDSS.bat             <- updated launcher
  requirements.txt
  general_diagnostics/
    app.py                       <- your supplied Full Diagnostics app
    requirements.txt
    agents/
    utils/
    config/
    RAG/
    model/
    heart.py
    breast_cancer.py
    ...all the other existing diagnostics files...

IMPORTANT:
Do NOT replace the whole general_diagnostics folder with only app.py.
Your supplied diagnostics app imports agents, utils, config, RAG, heart,
breast_cancer and other project resources. Keep those existing files.

How it works:
1. RUN_SMART_CDSS.bat starts only the Smart Clinic interface on port 8503.
2. It tells Smart Clinic the exact general_diagnostics folder and preferred port 8600.
3. When Full Diagnostics Scan is clicked, Smart Clinic launches exactly
   general_diagnostics\app.py with that folder as its working directory.
4. The running diagnostics UI is embedded inside the Smart Clinic page.
5. If 8600 is occupied, Smart Clinic automatically selects the next free port.
