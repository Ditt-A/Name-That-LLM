"""Run project utilities from a checkout before installing the editable package."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ntl.cli import main

if __name__=='__main__':raise SystemExit(main())
