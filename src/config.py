 # src/config.py
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"
