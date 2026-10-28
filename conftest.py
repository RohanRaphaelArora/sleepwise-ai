"""Make `import sleepwise` work when pytest is run from the project root."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
