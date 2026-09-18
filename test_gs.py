import json
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SERVICE_ACCOUNT_FILE = BASE_DIR / "service_account.json"
GOOGLE_CONFIG_FILE = BASE_DIR / "google_config.json"

print('Base dir:', BASE_DIR)

try:
    print('\nChecking files...')
    print('service_account.json exists:', SERVICE_ACCOUNT_FILE.exists())
    print('google_config.json exists:', GOOGLE_CONFIG_FILE.exists())

    sa = json.loads(SERVICE_ACCOUNT_FILE.read_text(encoding='utf-8'))
    print('\nservice_account.json client_email:', sa.get('client_email'))

    cfg = json.loads(GOOGLE_CONFIG_FILE.read_text(encoding='utf-8-sig'))
    sheet_cfg = cfg.get('google_sheet', cfg)
    spreadsheet_id = sheet_cfg.get('spreadsheet_id')
    print('Configured spreadsheet_id:', spreadsheet_id)

    from google.oauth2.service_account import Credentials
    import gspread

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
    ]

    creds = Credentials.from_service_account_file(str(SERVICE_ACCOUNT_FILE), scopes=scopes)
    print('\nCredentials created. Service account email (from creds):', getattr(creds, 'service_account_email', None))

    client = gspread.authorize(creds)
    print('gspread authorized, trying to open spreadsheet...')
    ss = client.open_by_key(spreadsheet_id)
    print('Spreadsheet title:', ss.title)
    print('Worksheets:', [ws.title for ws in ss.worksheets()])

    print('\nSUCCESS: Connected to Google Sheets.')
except Exception as e:
    print('\nERROR: An exception occurred while testing Google Sheets connection:')
    traceback.print_exc()
    raise
