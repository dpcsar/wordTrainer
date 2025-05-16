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
    DEFAULT_SNR_RANGE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, 
    DEFAULT_TEST_SAMPLES, DEFAULT_KEYWORD, KEYWORDS_DIR
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
    
    # Save original argv
    original_argv = sys.argv.copy()
    
    # Set argv to module path + args
    sys.argv = [module_path] + args
    
    # Load module
    module = load_module(module_path)
    
    try:
        # Always execute the main function
        # This ensures consistent behavior for all modules including mix_audio_samples.py
        module.main()
    finally:
        # Restore original argv
        sys.argv = original_argv


def main():
    # Base parser
    parser = argparse.ArgumentParser(
        description='Keyword Detection Model Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate keyword samples:
    python main.py generate-keywords --keyword "activate" --samples 50 --silence 200

  Generate non-keyword samples (negative examples):
    python main.py generate-non-keywords --samples 200 --avoid-keyword "activate" --silence 200

  Generate background noise:
    python main.py generate-noise --type all --samples 20

  Mix audio samples:
    python main.py mix-audio --keyword "activate" --num-mixes 100 

  Train model:
    python main.py train --keywords "activate"

  Test model with TTS samples:
    python main.py test-tts --keyword "activate" --samples 20

  Test model with non-keyword samples:
    python main.py test-non-keywords --keyword "active"

  Test model with microphone:
    python main.py test-mic --keyword "activate" --list-models
    
  Prepare model for Android:
    python main.py prepare-android --keyword "activate" --output-dir ./android_output
"""
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Add full argument information for each subcommand
    # 1. Generate keywords subcommand
    generate_keywords = subparsers.add_parser('generate-keywords', 
                                            help='Generate keyword samples using Google Cloud TTS',
                                            description='Generate keyword speech samples using Google Cloud Text-to-Speech API')
    generate_keywords.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD,
                       help=f'Keyword phrase to generate TTS samples for (default: "{DEFAULT_KEYWORD}")')
    generate_keywords.add_argument('--samples', type=int, default=DEFAULT_KEYWORD_SAMPLES,
                       help=f'Total number of samples to generate (default: {DEFAULT_KEYWORD_SAMPLES})')
    generate_keywords.add_argument('--output-dir', type=str, default=KEYWORDS_DIR,
                       help=f'Output directory for generated samples (default: {KEYWORDS_DIR})')
    generate_keywords.add_argument('--silence', type=int, default=DEFAULT_SILENCE_MS,
                       help=f'Silence padding to add at beginning and end in milliseconds (default: {DEFAULT_SILENCE_MS}ms)')
    
    # 2. Generate noise subcommand
    generate_noise = subparsers.add_parser('generate-noise', 
                                         help='Generate background noise samples',
                                         description='Generate background noise samples for training')
    generate_noise.add_argument('--type', type=str, default='all',
                   help='Type of noise to generate: "cockpit", "jet", "propeller", or "all" (default: all)')
    generate_noise.add_argument('--samples', type=int, default=DEFAULT_BACKGROUND_SAMPLES,
                   help=f'Number of samples to generate for each type (default: {DEFAULT_BACKGROUND_SAMPLES})')
    generate_noise.add_argument('--min-duration', type=float, default=DEFAULT_MIN_DURATION,
                   help=f'Minimum duration of samples in seconds (default: {DEFAULT_MIN_DURATION}s)')
    generate_noise.add_argument('--max-duration', type=float, default=DEFAULT_MAX_DURATION,
                   help=f'Maximum duration of samples in seconds (default: {DEFAULT_MAX_DURATION}s)')
    
    # 3. Generate non-keywords subcommand
    generate_non_keywords = subparsers.add_parser('generate-non-keywords', 
                                               help='Generate non-keyword samples using Google Cloud TTS',
                                               description='Generate non-keyword samples for negative training examples')
    generate_non_keywords.add_argument('--samples', type=int, default=DEFAULT_NON_KEYWORD_SAMPLES,
                         help=f'Number of non-keyword samples to generate (default: {DEFAULT_NON_KEYWORD_SAMPLES})')
    generate_non_keywords.add_argument('--avoid-keyword', type=str, default=DEFAULT_KEYWORD,
                         help=f'Avoid generating samples similar to this keyword (default: "{DEFAULT_KEYWORD}")')
    generate_non_keywords.add_argument('--silence', type=int, default=DEFAULT_SILENCE_MS,
                         help=f'Silence padding to add at beginning and end in milliseconds (default: {DEFAULT_SILENCE_MS}ms)')
    
    # 4. Mix audio subcommand
    mix_audio = subparsers.add_parser('mix-audio', 
                                    help='Mix keyword and non-keyword samples with background noise',
                                    description='Mix keyword and non-keyword audio with background noise at different SNR levels')
    mix_audio.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD,
                help=f'Keyword to use for mixing (default: "{DEFAULT_KEYWORD}")')
    mix_audio.add_argument('--num-mixes', type=int, default=100,
                help='Number of mixed samples to generate (default: 100)')
    mix_audio.add_argument('--snr-range', type=str, default=DEFAULT_SNR_RANGE,
                help=f'Range of Signal-to-Noise Ratio in dB as "min,max" (default: "{DEFAULT_SNR_RANGE}")')
    
    # 5. Train model subcommand
    train = subparsers.add_parser('train', 
                               help='Train keyword detection model',
                               description='Train a TensorFlow Lite model for keyword detection')
    train.add_argument('--keywords', type=str, default=DEFAULT_KEYWORD,
           help=f'Comma-separated list of keywords to train for (default: "{DEFAULT_KEYWORD}")')
    train.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
           help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    train.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
           help=f'Batch size for training (default: {DEFAULT_BATCH_SIZE})')
    train.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE,
           help=f'Learning rate for training (default: {DEFAULT_LEARNING_RATE})')
    
    # 6. Test with TTS subcommand
    test_tts = subparsers.add_parser('test-tts', 
                                  help='Test model using TTS samples',
                                  description='Test the trained model using newly generated TTS samples')
    test_tts.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD,
             help=f'Keyword to test (default: "{DEFAULT_KEYWORD}")')
    test_tts.add_argument('--samples', type=int, default=20,
             help='Number of test samples to generate (default: 20)')
    test_tts.add_argument('--threshold', type=float, default=DEFAULT_DETECTION_THRESHOLD,
             help=f'Detection threshold (default: {DEFAULT_DETECTION_THRESHOLD})')
    
    # 7. Test with non-keywords subcommand
    test_non_keywords = subparsers.add_parser('test-non-keywords', 
                                           help='Test model using non-keyword samples',
                                           description='Test the trained model against non-keyword samples for false positive analysis')
    test_non_keywords.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD,
                     help=f'Keyword model to test against (default: "{DEFAULT_KEYWORD}")')
    test_non_keywords.add_argument('--threshold', type=float, default=DEFAULT_DETECTION_THRESHOLD,
                     help=f'Detection threshold (default: {DEFAULT_DETECTION_THRESHOLD})')
    
    # 8. Test with microphone subcommand
    test_mic = subparsers.add_parser('test-mic', 
                                  help='Test model using microphone input',
                                  description='Test the trained model in real-time using microphone input')
    test_mic.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD,
              help=f'Keyword to detect (default: "{DEFAULT_KEYWORD}")')
    test_mic.add_argument('--threshold', type=float, default=DEFAULT_DETECTION_THRESHOLD,
              help=f'Detection threshold (default: {DEFAULT_DETECTION_THRESHOLD})')
    test_mic.add_argument('--list-models', action='store_true',
              help='List available trained models')
              
    # 9. Prepare for Android subcommand
    prepare_android = subparsers.add_parser('prepare-android',
                                         help='Prepare trained model for Android integration',
                                         description='Prepare and export trained model for Android integration and deployment')
    prepare_android.add_argument('--model', type=str, required=False,
                    help='Path to trained model file (.tflite format). If not specified, the latest model will be used.')
    prepare_android.add_argument('--output-dir', type=str,
                    help='Output directory for Android assets and generated files')
    prepare_android.add_argument('--package-name', type=str, default="com.example.keyworddetection",
                    help='Android package name for template generation (default: com.example.keyworddetection)')
    prepare_android.add_argument('--no-template', action='store_true',
                    help='Skip creation of Android integration template project')
    prepare_android.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD,
                    help=f'Keyword to find the latest model for (default: "{DEFAULT_KEYWORD}")')
    
    # Parse only the first argument (the command) and remaining args
    args, remaining_args = parser.parse_known_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')
    tests_dir = os.path.join(script_dir, 'tests')
    
    # Map commands to module paths
    module_paths = {
        'generate-keywords': os.path.join(src_dir, 'generate_keywords.py'),
        'generate-noise': os.path.join(src_dir, 'generate_background_noise.py'),
        'generate-non-keywords': os.path.join(src_dir, 'generate_non_keywords.py'),
        'mix-audio': os.path.join(src_dir, 'mix_audio_samples.py'),
        'train': os.path.join(src_dir, 'train_model.py'),
        'test-tts': os.path.join(tests_dir, 'test_model_tts.py'),
        'test-mic': os.path.join(tests_dir, 'test_model_mic.py'),
        'test-non-keywords': os.path.join(tests_dir, 'test_model_non_keywords.py'),
        'prepare-android': os.path.join(src_dir, 'prepare_for_android.py'),
    }
    
    # Process commands and pass remaining arguments directly to the modules
    if args.command and args.command in module_paths:
        run_module(module_paths[args.command], remaining_args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
