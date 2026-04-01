import sys
from pathlib import Path

# Add the repo root to sys.path so absolute imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
