#!/usr/bin/env python3
"""
Main entry point for the keyword detection model trainer.
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration settings
from config import (
    DEFAULT_DETECTION_THRESHOLD, DEFAULT_KEYWORD_SAMPLES, DEFAULT_NON_KEYWORD_SAMPLES,
    DEFAULT_SILENCE_MS, DEFAULT_BACKGROUND_SAMPLES, DEFAULT_MIN_DURATION, DEFAULT_MAX_DURATION,
    DEFAULT_NUM_MIXES, DEFAULT_SNR_RANGE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_TEST_SAMPLES, DEFAULT_KEYWORD
)

def load_module(module_path):
    """
    Dynamically load a Python module from a file path.
    
    Args:
        module_path: Path to Python module
        
    Returns:
        Loaded module
    """
    module_name = os.path.basename(module_path).split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_module(module_path, args=None):
    """
    Run a Python module with arguments.
    
    Args:
        module_path: Path to Python module
        args: Arguments to pass to module
    """
    if args is None:
        args = []
    
    # Set argv to module path + args
    sys.argv = [module_path] + args
    
    # Load module
    module = load_module(module_path)
    
    # Always execute the main function
    # This ensures consistent behavior for all modules including mix_audio_samples.py
    module.main()

def main():
    # Base parser
    parser = argparse.ArgumentParser(
        description='Keyword Detection Model Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate keyword samples:
    python main.py generate-keywords --keyword "activate" --samples 50

  Generate background noise:
    python main.py generate-noise --type all --samples 20

  Mix audio samples:
    python main.py mix-audio --keyword "activate" --num-mixes 100 --noise-types jet propeller

  Train model:
    python main.py train --keywords "activate" "shutdown" --epochs 50

  Test model with gTTS samples:
    python main.py test-gtts --model "models/keyword_detection_activate.tflite" --dir "data/keywords/activate"

  Test model with microphone:
    python main.py test-mic --model "models/keyword_detection_activate.tflite" --threshold 0.7
"""
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Define simple subparsers for each command without duplicating all arguments
    subparsers.add_parser('generate-keywords', help='Generate keyword samples using gTTS')
    subparsers.add_parser('generate-noise', help='Generate background noise samples')
    subparsers.add_parser('generate-non-keywords', help='Generate non-keyword samples for training')
    subparsers.add_parser('mix-audio', help='Mix keyword and non-keyword samples with background noise')
    subparsers.add_parser('train', help='Train keyword detection model')
    subparsers.add_parser('test-gtts', help='Test model using gTTS samples')
    subparsers.add_parser('test-non-keywords', help='Test model using non-keyword samples')
    subparsers.add_parser('test-mic', help='Test model using microphone input')
    
    # Parse only the first argument (the command)
    # We use parse_known_args instead of parse_args to get the command without validating other args
    args, remaining_args = parser.parse_known_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')
    tests_dir = os.path.join(script_dir, 'tests')
    
    # Process commands and pass remaining arguments directly to the modules
    if args.command == 'generate-keywords':
        run_module(os.path.join(src_dir, 'generate_keywords.py'), remaining_args)
    
    elif args.command == 'generate-noise':
        run_module(os.path.join(src_dir, 'generate_background_noise.py'), remaining_args)
    
    elif args.command == 'generate-non-keywords':
        run_module(os.path.join(src_dir, 'generate_non_keywords.py'), remaining_args)
    
    elif args.command == 'mix-audio':
        run_module(os.path.join(src_dir, 'mix_audio_samples.py'), remaining_args)
    
    elif args.command == 'train':
        run_module(os.path.join(src_dir, 'train_model.py'), remaining_args)
    
    elif args.command == 'test-gtts':
        run_module(os.path.join(tests_dir, 'test_model_gtts.py'), remaining_args)
    
    elif args.command == 'test-mic':
        run_module(os.path.join(tests_dir, 'test_model_mic.py'), remaining_args)
    
    elif args.command == 'test-non-keywords':
        run_module(os.path.join(tests_dir, 'test_model_non_keywords.py'), remaining_args)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
