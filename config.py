# config.py - updated

import os

BASE_DIR = r"D:\Hybrid NIDS Project"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")   # <-- ADD THIS LINE

# Create folders if missing
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)   # <-- ADD THIS TOO

RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_FRAC = 0.3

LABEL_COL_CANDIDATES = ['Label', ' Label', 'label', ' Attack']

print("Config loaded - paths ready")