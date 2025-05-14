#!/usr/bin/env python3
"""
Utility functions for handling file paths in the project.
"""

import os
import glob

from config import MODELS_DIR, DEFAULT_KEYWORD

def normalize_model_path(model_path, script_dir=None):
    """
    Normalize model path to absolute path.
    
    Args:
        model_path: Relative or absolute path to model
        script_dir: Directory of the calling script (optional)
        
    Returns:
        Absolute path to model
    """
    if os.path.isabs(model_path):
        return model_path
    
    # If script_dir is provided, use it as reference
    if script_dir:
        return os.path.abspath(os.path.join(script_dir, '..', model_path))
    
    # Otherwise use current working directory
    return os.path.abspath(model_path)

def find_latest_model_by_keyword(keyword=DEFAULT_KEYWORD, models_dir=MODELS_DIR):
    """
    Find the latest model for a given keyword.
    
    Args:
        keyword: Keyword to find the latest model for
        models_dir: Directory containing trained models
        
    Returns:
        Path to the latest model, or None if no model found
    """
    # Look for .tflite files first
    tflite_files = glob.glob(os.path.join(models_dir, f"*{keyword}*.tflite"))
    tflite_files = [f for f in tflite_files if not f.endswith("_quantized.tflite")]
    
    if tflite_files:
        # Return the most recent one based on filename timestamp
        return max(tflite_files, key=os.path.getmtime)
    
    # If no .tflite files found, look for .h5 files
    h5_files = glob.glob(os.path.join(models_dir, f"*{keyword}*.h5"))
    
    if h5_files:
        return max(h5_files, key=os.path.getmtime)
    
    return None
