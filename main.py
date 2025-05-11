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
    DEFAULT_DETECTION_THRESHOLD, DEFAULT_KEYWORD_SAMPLES, DEFAULT_SILENCE_MS,
    DEFAULT_BACKGROUND_SAMPLES, DEFAULT_MIN_DURATION, DEFAULT_MAX_DURATION,
    DEFAULT_NUM_MIXES, DEFAULT_SNR_RANGE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_TEST_SAMPLES
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
    
    # Load and run module
    module = load_module(module_path)
    
    # Check if the module has a main function and call it
    if hasattr(module, 'main') and callable(module.main):
        module.main()
    
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
    python main.py generate-keywords --keyword "activate" --samples 50

  Generate background noise:
    python main.py generate-noise --type all --samples 20

  Mix audio samples:
    python main.py mix-audio --keyword "activate" --num-mixes 100

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
    
    # Generate keywords command
    parser_gk = subparsers.add_parser('generate-keywords', help='Generate keyword samples using gTTS')
    parser_gk.add_argument('--keyword', type=str, required=True, help='Keyword to generate samples for')
    parser_gk.add_argument('--samples', type=int, default=DEFAULT_KEYWORD_SAMPLES, help=f'Number of samples to generate (default: {DEFAULT_KEYWORD_SAMPLES})')
    parser_gk.add_argument('--output-dir', type=str, help='Output directory')
    parser_gk.add_argument('--silence', type=int, default=DEFAULT_SILENCE_MS, help=f'Silence to add at beginning and end (milliseconds) (default: {DEFAULT_SILENCE_MS})')
    
    # Generate background noise command
    parser_gn = subparsers.add_parser('generate-noise', help='Generate background noise samples')
    parser_gn.add_argument('--type', type=str, choices=['propeller', 'jet', 'cockpit', 'all'], 
                        default='all', help='Type of noise to generate')
    parser_gn.add_argument('--samples', type=int, default=DEFAULT_BACKGROUND_SAMPLES, help=f'Number of samples to generate (default: {DEFAULT_BACKGROUND_SAMPLES})')
    parser_gn.add_argument('--output-dir', type=str, help='Output directory')
    parser_gn.add_argument('--min-duration', type=float, default=DEFAULT_MIN_DURATION, help=f'Minimum duration in seconds (default: {DEFAULT_MIN_DURATION})')
    parser_gn.add_argument('--max-duration', type=float, default=DEFAULT_MAX_DURATION, help=f'Maximum duration in seconds (default: {DEFAULT_MAX_DURATION})')
    
    # Generate non-keywords command
    parser_nk = subparsers.add_parser('generate-non-keywords', help='Generate non-keyword samples for training')
    parser_nk.add_argument('--samples', type=int, default=DEFAULT_KEYWORD_SAMPLES, help=f'Number of samples to generate (default: {DEFAULT_KEYWORD_SAMPLES})')
    parser_nk.add_argument('--output-dir', type=str, help='Output directory')
    parser_nk.add_argument('--silence', type=int, default=DEFAULT_SILENCE_MS, help=f'Silence to add at beginning and end (milliseconds) (default: {DEFAULT_SILENCE_MS})')
    parser_nk.add_argument('--avoid-keyword', type=str, help='Keyword to avoid using as non-keyword')
    
    # Mix audio samples command
    parser_ma = subparsers.add_parser('mix-audio', help='Mix keyword samples with background noise')
    parser_ma.add_argument('--keyword', type=str, required=True, help='Keyword to mix')
    parser_ma.add_argument('--noise-types', type=str, nargs='+', choices=['propeller', 'jet', 'cockpit'], 
                        help='Types of background noise to mix with')
    parser_ma.add_argument('--num-mixes', type=int, default=DEFAULT_NUM_MIXES, help=f'Number of mixed samples to generate (default: {DEFAULT_NUM_MIXES})')
    parser_ma.add_argument('--min-snr', type=float, default=DEFAULT_SNR_RANGE[0], help=f'Minimum SNR in dB (default: {DEFAULT_SNR_RANGE[0]})')
    parser_ma.add_argument('--max-snr', type=float, default=DEFAULT_SNR_RANGE[1], help=f'Maximum SNR in dB (default: {DEFAULT_SNR_RANGE[1]})')
    parser_ma.add_argument('--keywords-dir', type=str, help='Keywords directory')
    parser_ma.add_argument('--backgrounds-dir', type=str, help='Backgrounds directory')
    parser_ma.add_argument('--output-dir', type=str, help='Output directory')
    
    # Train model command
    parser_tm = subparsers.add_parser('train', help='Train keyword detection model')
    parser_tm.add_argument('--keywords', type=str, nargs='+', required=True, help='Keywords to detect')
    parser_tm.add_argument('--data-dir', type=str, help='Directory containing audio data')
    parser_tm.add_argument('--model-dir', type=str, help='Directory to save trained models')
    parser_tm.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser_tm.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help=f'Training batch size (default: {DEFAULT_BATCH_SIZE})')
    parser_tm.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, help=f'Initial learning rate (default: {DEFAULT_LEARNING_RATE})')
    
    # Test model with gTTS samples command
    parser_tg = subparsers.add_parser('test-gtts', help='Test model using gTTS samples')
    parser_tg.add_argument('--model', type=str, help='Path to trained model (.h5 or .tflite) otherwise it will use the latest model')
    parser_tg.add_argument('--keyword', type=str, help='Keyword to find the latest model for or to test with')
    parser_tg.add_argument('--file', type=str, help='Path to a single audio file to test')
    parser_tg.add_argument('--dir', type=str, help='Directory containing audio files to test')
    parser_tg.add_argument('--samples', type=int, default=DEFAULT_TEST_SAMPLES, help=f'Maximum number of samples to test in batch mode (default: {DEFAULT_TEST_SAMPLES})')
    parser_tg.add_argument('--keywords-dir', type=str, help='Directory containing keyword samples')
    
    # Test model with non-keywords command
    parser_tnk = subparsers.add_parser('test-non-keywords', help='Test model using non-keyword samples')
    parser_tnk.add_argument('--model', type=str, help='Path to trained model (.h5 or .tflite)')
    parser_tnk.add_argument('--keyword', type=str, help='Keyword to find the latest model for')
    parser_tnk.add_argument('--file', type=str, help='Path to a single audio file to test')
    parser_tnk.add_argument('--dir', type=str, help='Directory containing audio files to test')
    parser_tnk.add_argument('--samples', type=int, default=DEFAULT_TEST_SAMPLES, help=f'Maximum number of samples to test in batch mode (default: {DEFAULT_TEST_SAMPLES})')
    parser_tnk.add_argument('--keywords-dir', type=str, help='Directory containing keyword samples')
    parser_tnk.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser_tnk.add_argument('--check-exist', action='store_true', help='Only check if non-keywords exist and exit')

    # Test model with microphone command
    parser_tm = subparsers.add_parser('test-mic', help='Test model using microphone input')
    parser_tm.add_argument('--model', type=str, help='Path to trained model (.h5 or .tflite)')
    parser_tm.add_argument('--keyword', type=str, help='Keyword to find the latest model for or to test with')
    parser_tm.add_argument('--threshold', type=float, default=DEFAULT_DETECTION_THRESHOLD, 
                        help=f'Detection threshold (default: {DEFAULT_DETECTION_THRESHOLD} from config)')
    parser_tm.add_argument('--device', type=int, help='Audio device index')
    parser_tm.add_argument('--no-viz', action='store_true', help='Disable visualization')
   
    # Parse arguments
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')
    tests_dir = os.path.join(script_dir, 'tests')
    
    # Process commands
    if args.command == 'generate-keywords':
        cmd_args = []
        if args.keyword:
            cmd_args.extend(['--keyword', args.keyword])
        if args.samples:
            cmd_args.extend(['--samples', str(args.samples)])
        if args.output_dir:
            cmd_args.extend(['--output-dir', args.output_dir])
        if args.silence:
            cmd_args.extend(['--silence', str(args.silence)])
        
        run_module(os.path.join(src_dir, 'generate_keywords.py'), cmd_args)
    
    elif args.command == 'generate-noise':
        cmd_args = []
        if args.type:
            cmd_args.extend(['--type', args.type])
        if args.samples:
            cmd_args.extend(['--samples', str(args.samples)])
        if args.output_dir:
            cmd_args.extend(['--output-dir', args.output_dir])
        if args.min_duration:
            cmd_args.extend(['--min-duration', str(args.min_duration)])
        if args.max_duration:
            cmd_args.extend(['--max-duration', str(args.max_duration)])
        
        run_module(os.path.join(src_dir, 'generate_background_noise.py'), cmd_args)
    
    elif args.command == 'generate-non-keywords':
        cmd_args = []
        if args.samples:
            cmd_args.extend(['--samples', str(args.samples)])
        if args.output_dir:
            cmd_args.extend(['--output-dir', args.output_dir])
        if args.silence:
            cmd_args.extend(['--silence', str(args.silence)])
        if args.avoid_keyword:
            cmd_args.extend(['--avoid-keyword', args.avoid_keyword])
        
        run_module(os.path.join(src_dir, 'generate_non_keywords.py'), cmd_args)
    
    elif args.command == 'mix-audio':
        cmd_args = []
        if args.keyword:
            cmd_args.extend(['--keyword', args.keyword])
        if args.noise_types:
            cmd_args.extend(['--noise-types'] + args.noise_types)
        if args.num_mixes:
            cmd_args.extend(['--num-mixes', str(args.num_mixes)])
        if args.min_snr:
            cmd_args.extend(['--min-snr', str(args.min_snr)])
        if args.max_snr:
            cmd_args.extend(['--max-snr', str(args.max_snr)])
        if args.keywords_dir:
            cmd_args.extend(['--keywords-dir', args.keywords_dir])
        if args.backgrounds_dir:
            cmd_args.extend(['--backgrounds-dir', args.backgrounds_dir])
        if args.output_dir:
            cmd_args.extend(['--output-dir', args.output_dir])
        
        run_module(os.path.join(src_dir, 'mix_audio_samples.py'), cmd_args)
    
    elif args.command == 'train':
        cmd_args = []
        if args.keywords:
            cmd_args.extend(['--keywords'] + args.keywords)
        if args.data_dir:
            cmd_args.extend(['--data-dir', args.data_dir])
        if args.model_dir:
            cmd_args.extend(['--model-dir', args.model_dir])
        if args.epochs:
            cmd_args.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            cmd_args.extend(['--batch-size', str(args.batch_size)])
        if args.learning_rate:
            cmd_args.extend(['--learning-rate', str(args.learning_rate)])
        
        run_module(os.path.join(src_dir, 'train_model.py'), cmd_args)
    
    elif args.command == 'test-gtts':
        cmd_args = []
        if args.model:
            cmd_args.extend(['--model', args.model])
        if args.keyword:
            cmd_args.extend(['--keyword', args.keyword])
        if args.file:
            cmd_args.extend(['--file', args.file])
        if args.dir:
            cmd_args.extend(['--dir', args.dir])
        if args.samples:
            cmd_args.extend(['--samples', str(args.samples)])
        if args.keywords_dir:
            cmd_args.extend(['--keywords-dir', args.keywords_dir])
        
        run_module(os.path.join(tests_dir, 'test_model_gtts.py'), cmd_args)
    
    elif args.command == 'test-mic':
        cmd_args = []
        if args.model:
            cmd_args.extend(['--model', args.model])
        if args.threshold:
            cmd_args.extend(['--threshold', str(args.threshold)])
        if args.device is not None:
            cmd_args.extend(['--device', str(args.device)])
        if args.no_viz:
            cmd_args.append('--no-viz')
        
        run_module(os.path.join(tests_dir, 'test_model_mic.py'), cmd_args)
    
    elif args.command == 'test-non-keywords':
        cmd_args = []
        if args.model:
            cmd_args.extend(['--model', args.model])
        if args.keyword:
            cmd_args.extend(['--keyword', args.keyword])
        if args.file:
            cmd_args.extend(['--file', args.file])
        if args.dir:
            cmd_args.extend(['--dir', args.dir])
        if args.samples:
            cmd_args.extend(['--samples', str(args.samples)])
        if args.keywords_dir:
            cmd_args.extend(['--keywords-dir', args.keywords_dir])
        if args.list_models:
            cmd_args.append('--list-models')
        if args.check_exist:
            cmd_args.append('--check-exist')
        
        run_module(os.path.join(tests_dir, 'test_model_non_keywords.py'), cmd_args)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
