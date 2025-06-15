"""
Module to define global values used in the whole project.

Useful constants:
- DATASET_DIR: Path to the datasets directory.
- CHECKPOINT_DIR: Path to the checkpoints directory.
- LOG_DIR: Path to the logs directory.
"""

from pathlib import Path


DATASET_DIR: Path = Path('datasets')
OUTPUT_DIR: Path = Path('output')
CHECKPOINT_DIR: Path = OUTPUT_DIR / 'checkpoints'
LOG_DIR: Path = OUTPUT_DIR / 'logs'
