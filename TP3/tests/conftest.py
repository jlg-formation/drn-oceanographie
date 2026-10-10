"""Rend les modules de TP3/ importables depuis les tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
