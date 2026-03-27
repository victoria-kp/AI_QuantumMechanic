"""Configuration: model settings and physical constants."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-5-20250929"

# --- Agent ---
MAX_RETRIES = 2        # checker retry cap (prevent infinite loops)
MAX_TOOL_CALLS = 200   # safety limit per problem

# --- Physical constants (SI) ---
HBAR = 1.0545718e-34   # reduced Planck constant (J*s)
M_E = 9.10938e-31      # electron mass (kg)
E_CHARGE = 1.602e-19   # elementary charge (C)
A0 = 5.29177e-11       # Bohr radius (m)
EV_TO_J = 1.602e-19    # eV to Joules conversion