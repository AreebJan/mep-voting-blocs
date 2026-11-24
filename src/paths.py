from pathlib import Path

#Creates a path that is easy to use for cleaning the data
ROOT = Path(__file__).resolve().parent.parent # repo root
RAW  = ROOT / "data" / "raw" 
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)




