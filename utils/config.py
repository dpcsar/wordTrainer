"""
Configuration settings for the keyword detection model.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
KEYWORDS_DIR = os.path.join(DATA_DIR, 'keywords')
BACKGROUNDS_DIR = os.path.join(DATA_DIR, 'backgrounds')
MIXED_DIR = os.path.join(DATA_DIR, 'mixed')

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Recordings directory
RECORDINGS_DIR = os.path.join(BASE_DIR, 'recordings')

# Audio settings
SAMPLE_RATE = 16000
FEATURE_PARAMS = {
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'window_length_ms': 30,
    'window_step_ms': 10,
}

# Training settings
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# Detection settings
DEFAULT_DETECTION_THRESHOLD = 0.5

# Available accents in gTTS
ACCENTS = {
    'us': 'American',
    'uk': 'British',
    'ca': 'Canadian',
    'au': 'Australian',
    'in': 'Indian',
    'ie': 'Irish',
    'za': 'South African'
}

# Age groups for simulation
AGE_GROUPS = ['child', 'young_adult', 'adult', 'senior']

# Genders for simulation
GENDERS = ['male', 'female']

# Background noise types
NOISE_TYPES = ['propeller', 'jet', 'cockpit']

# Default SNR range for mixing
DEFAULT_SNR_RANGE = (-5, 20)

# Create directories if they don't exist
for directory in [KEYWORDS_DIR, BACKGROUNDS_DIR, MIXED_DIR, MODELS_DIR, RECORDINGS_DIR]:
    os.makedirs(directory, exist_ok=True)
