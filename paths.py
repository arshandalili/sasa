import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
ACTIVATIONS = Path(os.environ.get("SASA_ACTIVATIONS", REPO / "activations"))
CHECKPOINTS = Path(os.environ.get("SASA_CHECKPOINTS", REPO / "checkpoints"))
RESULTS = Path(os.environ.get("SASA_RESULTS", REPO / "results"))
DATA = Path(os.environ.get("SASA_DATA", REPO / "data"))
