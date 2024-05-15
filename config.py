import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = ROOT_DIR / Path(r"Data/")
DATA_RAW_DIR = ROOT_DIR / Path(r"Data/raw/")
DATA_PROCESSED_DIR = ROOT_DIR / Path(r"Data/preprocessed/")
DATA_FEATURE_ENGINEERED_DIR = ROOT_DIR / Path(r"Data/feature_engineered/")
DATA_TRIAL_DIR = ROOT_DIR / Path(r"Data/trial/")
