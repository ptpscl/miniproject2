"""Pytest configuration.

Placing this file at the repository root ensures the root directory is
on sys.path during test collection, so `from src...` imports resolve
both locally and on a clean CI runner.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)