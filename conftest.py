"""Root-level conftest.py for pytest.

Ensures the project root is on sys.path so that `core` and `experiments`
packages are importable from any test directory.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
