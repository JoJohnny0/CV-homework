"""
Module to define paths used in the whole project.

Useful constants:
- MVTECH_DIR: Path to the MVTech dataset directory.
"""

from pathlib import Path


DATA_DIR: Path = Path('data')
DATASET_DIR: Path = DATA_DIR / 'datasets'
MVTECH_DIR: Path = DATASET_DIR / 'mvtech'
