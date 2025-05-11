"""
Configuration settings for the keyword detection model.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
KEYWORDS_DIR = os.path.join(DATA_DIR, 'keywords')
BACKGROUNDS_DIR = os.path.join(DATA_DIR, 'backgrounds')
MIXED_DIR = os.path.join(DATA_DIR, 'mixed')

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

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
DEFAULT_NEGATIVE_SAMPLES_RATIO = 3.0

# Detection settings
DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_KEYWORD = "activate"  # Default keyword for model training and testing

# Audio generation settings
DEFAULT_KEYWORD_SAMPLES = 50
DEFAULT_NON_KEYWORD_SAMPLES = 50  # Number of non-keyword samples to generate
DEFAULT_SILENCE_MS = 500
DEFAULT_BACKGROUND_SAMPLES = 200  # Originally 20, increased to match workflow script
DEFAULT_MIN_DURATION = 3.0
DEFAULT_MAX_DURATION = 10.0
DEFAULT_NUM_MIXES = 100

# Testing settings
DEFAULT_TEST_SAMPLES = 10
DEFAULT_AUDIO_DURATION = 2  # seconds
DEFAULT_BUFFER_DURATION = 5  # seconds

# Available accents in gTTS with detailed information
ACCENTS = [
    {"lang": "en", "tld": "com", "name": "US English", "gender": "male"},
    {"lang": "en", "tld": "com", "name": "US English", "gender": "female"},
    {"lang": "en", "tld": "co.uk", "name": "British English", "gender": "male"},
    {"lang": "en", "tld": "co.uk", "name": "British English", "gender": "female"},
    {"lang": "en", "tld": "com.au", "name": "Australian English", "gender": "male"},
    {"lang": "en", "tld": "com.au", "name": "Australian English", "gender": "female"},
    {"lang": "en", "tld": "co.in", "name": "Indian English", "gender": "male"},
    {"lang": "en", "tld": "co.in", "name": "Indian English", "gender": "female"},
    {"lang": "en", "tld": "ie", "name": "Irish English", "gender": "male"},
    {"lang": "en", "tld": "ie", "name": "Irish English", "gender": "female"},
    {"lang": "en", "tld": "ca", "name": "Canadian English", "gender": "male"},
    {"lang": "en", "tld": "ca", "name": "Canadian English", "gender": "female"},
]

# Age groups for simulation
AGE_GROUPS = ['child', 'young_adult', 'adult', 'senior']

# Background noise types
NOISE_TYPES = ['propeller', 'jet', 'cockpit']

# Default SNR range for mixing
DEFAULT_SNR_RANGE = (-5, 20)

# List of common words to use as non-keywords
# These are selected to be distinct from typical wake words 
# but represent a variety of speech patterns
NON_KEYWORDS = [
    "hello", "thanks", "sorry", "please", "coffee", 
    "water", "today", "weather", "music", "play",
    "stop", "pause", "continue", "next", "previous",
    "volume", "morning", "evening", "dinner", "lunch",
    "breakfast", "meeting", "schedule", "reminder", "alarm",
    "computer", "system", "network", "download", "upload",
    "message", "email", "phone", "call", "text",
    "picture", "photo", "camera", "video", "record",
    "time", "date", "month", "year", "hour",
    "minute", "second", "tomorrow", "yesterday", "weekend"
]
