"""Pytest configuration: adds the project root to sys.path so tests import birdie directly."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
