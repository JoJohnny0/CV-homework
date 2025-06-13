"""
Module to define global values used in the whole project.

Useful constants:
- MVTECH_DIR: Path to the MVTech dataset directory.
"""

from pathlib import Path


DATASET_DIR: Path = Path('datasets')
MVTECH_DIR: Path = DATASET_DIR / 'mvtech'
