# File: C:\Users\konos\DocLayout-YOLO\bookscanner\detect_number\run_sync.ps1
Set-Location "C:\Users\konos\DocLayout-YOLO"
.\venv\Scripts\Activate.ps1
Set-Location "C:\Users\konos\DocLayout-YOLO\bookscanner\detect_number"
python .\sync_dirs.py
pause  # keeps window open to see output