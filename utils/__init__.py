"""Root package info."""

import os
import sys

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

SAT_ROOT = f"{_PACKAGE_ROOT}/.."
sys.path.insert(0, SAT_ROOT)
