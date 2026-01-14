import os

# --- PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 1. Priority: Get from Environment Variable (set by user on Kaggle/Terminal)
ENV_RAW_PATH = os.environ.get("RAW_DATA_PATH")

# 2. Fallback: Default Kaggle path (if user forgets to set env)
DEFAULT_KAGGLE_PATH = "/kaggle/input/fish-feeding-intensity"

if ENV_RAW_PATH:
    print(f"Config: Using RAW_DATA_PATH from Environment: {ENV_RAW_PATH}")
    RAW_DATA_ROOT = ENV_RAW_PATH
    
    # Automatically determine where to save processed data
    if os.path.exists("/kaggle/working"):
        PROCESSED_DATA_ROOT = "/kaggle/working/processed_data"
    else:
        PROCESSED_DATA_ROOT = os.path.join(PROJECT_ROOT, "processed_data")

elif os.path.exists(DEFAULT_KAGGLE_PATH):
    print(f"Config: Detected default Kaggle path: {DEFAULT_KAGGLE_PATH}")
    RAW_DATA_ROOT = DEFAULT_KAGGLE_PATH
    PROCESSED_DATA_ROOT = "/kaggle/working/processed_data"

else:
    print("Config: Using Local Project path.")
    RAW_DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
    PROCESSED_DATA_ROOT = os.path.join(PROJECT_ROOT, "processed_data")

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
