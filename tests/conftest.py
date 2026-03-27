import sys
from pathlib import Path

# Add the repo's parent to sys.path so "from AI_QuantumMechanic.tools ..."
# resolves when pytest runs from inside AI_QuantumMechanic/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
