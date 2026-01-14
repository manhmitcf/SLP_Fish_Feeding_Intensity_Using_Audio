import os

"""
Configuration file for Audio Preprocessing.
Simplifies path management by reading directly from Environment Variables.
"""


# 1. Raw Data Path: Must be set via Environment Variable 'RAW_DATA_PATH'
# Example: export RAW_DATA_PATH="/path/to/your/dataset"
# Defaults to './dataset' if not set.
RAW_DATA_ROOT = os.environ.get("RAW_DATA_PATH", "./dataset")

# 2. Processed Data Path: Where to save .pt files
# Can be set via 'PROCESSED_DATA_PATH', otherwise defaults to './processed_data'
PROCESSED_DATA_ROOT = os.environ.get("PROCESSED_DATA_PATH", "./processed_data")

# --- AUDIO CONFIGURATION ---
# Target sample rate for resampling
SAMPLE_RATE = 64000

# Mel Spectrogram parameters
NUM_MEL_BINS = 128
N_FFT = 2048
HOP_LENGTH = 512  # Frame shift
WIN_LENGTH = 1024

# --- DATA SPLIT CONFIGURATION ---
SEED = 42
TEST_SAMPLES_PER_CLASS = 75

# --- LABEL MAPPING ---
# Map folder names to integer labels
CLASS_MAPPING = {
    'none': 0,
    'strong': 1,
    'middle': 2,
    'weak': 3
}
