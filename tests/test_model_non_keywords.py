#!/usr/bin/env python3
"""
Test keyword detection model using non-keywords samples.
"""

import os
import argparse
import numpy as np
import json
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import time
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODELS_DIR, KEYWORDS_DIR, SAMPLE_RATE, DEFAULT_DETECTION_THRESHOLD,
                    DEFAULT_KEYWORD, DEFAULT_TEST_SAMPLES, FEATURE_PARAMS, DEFAULT_SHOW_PLOTS)
from src.audio_utils import load_audio, extract_features, plot_waveform
from src.path_utils import normalize_model_path, find_latest_model_by_keyword

class NonKeywordTester:
    def __init__(self, model_path, keywords_dir, sample_rate=SAMPLE_RATE, threshold=None):
        """
        Initialize NonKeywordTester.
        
        Args:
            model_path: Path to trained model (.h5 or .tflite)
            keywords_dir: Directory containing keyword samples
            sample_rate: Audio sample rate (default from config.SAMPLE_RATE)
            threshold: Detection threshold (None to use default from config)
        """
        self.threshold = threshold if threshold is not None else DEFAULT_DETECTION_THRESHOLD
        self.model_path = model_path
        self.keywords_dir = keywords_dir
        self.sample_rate = sample_rate
        
        # Load model metadata
        model_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            # Find this specific model in the metadata
            model_name = os.path.basename(model_path).split('.')[0]
            # Handle optimized models by removing the _optimized suffix
            base_model_name = model_name.replace('_optimized', '')
            model_info = None
            
            for model in self.model_metadata['models']:
                if model['name'] == model_name or model['name'] == base_model_name:
                    model_info = model
                    break
            
            if model_info:
                self.keywords = model_info['keywords']
                self.feature_params = model_info.get('feature_params', {})
                self.class_names = ['negative'] + self.keywords
                print(f"Model keywords: {self.keywords}")
            else:
                # More descriptive error message
                base_name = model_name.replace('_optimized', '')
                raise ValueError(f"Model {model_name} not found in metadata. Available models: {[model['name'] for model in self.model_metadata['models']]}")
        else:
            print(f"Warning: Model metadata not found at {metadata_path}")
            self.keywords = []
            self.feature_params = {}
        
        # Load model
        if model_path.endswith('.tflite'):
            self.load_tflite_model(model_path)
        else:
            self.load_keras_model(model_path)
    
    def load_keras_model(self, model_path):
        """
        Load TensorFlow Keras model.
        
        Args:
            model_path: Path to .keras model file
        """
        self.model_type = 'keras'
        self.model = tf.keras.models.load_model(model_path)
        print(f"Loaded Keras model from {model_path}")
    
    def load_tflite_model(self, model_path):
        """
        Load TFLite model.
        
        Args:
            model_path: Path to .tflite model file
        """
        self.model_type = 'tflite'
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Loaded TFLite model from {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input type: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        
        # Check if the model is quantized
        if self.input_details[0]['dtype'] == np.int8:
            print("This is a quantized INT8 model")
            if 'quantization_parameters' in self.input_details[0]:
                scale = self.input_details[0]['quantization_parameters']['scales'][0]
                zero_point = self.input_details[0]['quantization_parameters']['zero_points'][0]
                print(f"Input quantization scale: {scale}, zero point: {zero_point}")
        
        # Number of classes is the last dimension of output shape
        num_classes = self.output_details[0]['shape'][-1]
        if not self.keywords:
            # Infer number of keywords from output shape
            self.keywords = [f"keyword_{i}" for i in range(1, num_classes)]
            self.class_names = ['negative'] + self.keywords
            print(f"Inferred keywords: {self.keywords}")
    
    def extract_features(self, audio_data):
        """
        Extract features from audio data.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            features: Extracted features
        """
        # Extract MFCCs using defaults from config if not specified in model's feature_params
        n_mfcc = self.feature_params.get('n_mfcc', FEATURE_PARAMS['n_mfcc'])
        n_fft = self.feature_params.get('n_fft', FEATURE_PARAMS['n_fft'])
        hop_length = self.feature_params.get('hop_length', FEATURE_PARAMS['hop_length'])
        
        mfccs = extract_features(
            audio_data, 
            sr=self.sample_rate, 
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Transpose to get time on the first axis
        mfccs = mfccs.T
        
        # Handle dimension matching for model input
        if hasattr(self, 'input_details') and self.input_details:
            expected_time_dim = self.input_details[0]['shape'][1]
            if mfccs.shape[0] != expected_time_dim:
                if mfccs.shape[0] > expected_time_dim:
                    # Truncate longer sequences
                    mfccs = mfccs[:expected_time_dim, :]
                else:
                    # Pad shorter sequences with zeros
                    padding = np.zeros((expected_time_dim - mfccs.shape[0], mfccs.shape[1]))
                    mfccs = np.vstack((mfccs, padding))
        
        return mfccs
    
    def predict(self, features):
        """
        Run inference on input features.
        
        Args:
            features: Input features
            
        Returns:
            predictions: Model predictions
        """
        if self.model_type == 'keras':
            # Add batch dimension if not present
            if len(features.shape) == 2:
                features = np.expand_dims(features, axis=0)
            
            # Run inference
            predictions = self.model.predict(features)
            
            return predictions[0]
        
        else:  # TFLite
            # Ensure features have correct shape for input
            input_shape = self.input_details[0]['shape']
            input_type = self.input_details[0]['dtype']
            is_quantized = input_type == np.int8
            
            if input_shape[0] == 1:  # Batch size of 1
                features = np.expand_dims(features, axis=0)
            
            # For quantized models, convert float values to int8
            if is_quantized:
                print("Detected quantized INT8 model, converting input to INT8")
                input_details = self.input_details[0]
                if 'quantization_parameters' in input_details:
                    quantization_params = input_details['quantization_parameters']
                    scale = quantization_params['scales'][0]
                    zero_point = quantization_params['zero_points'][0]
                    
                    # Convert float to int8 using scale and zero point
                    features = (features / scale + zero_point).astype(np.int8)
                    print(f"Quantized input range: {features.min()} to {features.max()}")
                else:
                    # If quantization parameters are not available, use a simpler estimation
                    # based on the typical range of MFCC features
                    print("Warning: Quantization parameters not found, using estimated scale")
                    features = np.clip(features * 10, -128, 127).astype(np.int8)
            else:
                # For float models, use float32
                features = features.astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # If output is quantized, dequantize it
            if len(self.output_details[0]['shape']) > 0 and 'quantization_parameters' in self.output_details[0]:
                quantization_params = self.output_details[0]['quantization_parameters']
                if quantization_params['scales'] and quantization_params['zero_points']:
                    scale = quantization_params['scales'][0]
                    zero_point = quantization_params['zero_points'][0]
                    predictions = scale * (predictions.astype(np.float32) - zero_point)
            
            return predictions[0]
    
    def test_file(self, file_path, plot_results=DEFAULT_SHOW_PLOTS):
        """
        Test model on a single audio file.
        
        Args:
            file_path: Path to audio file
            plot_results: Whether to plot waveform and spectrogram (defaults to config.DEFAULT_SHOW_PLOTS)
            
        Returns:
            result: Dictionary with prediction results
        """
        print(f"Testing file: {file_path}")
        
        # Load audio
        audio, sr = load_audio(file_path, target_sr=self.sample_rate)
        
        # Extract features
        features = self.extract_features(audio)
        
        # Make prediction
        predictions = self.predict(features)
        
        # Get predicted class
        predicted_class = np.argmax(predictions)
        predicted_label = self.class_names[predicted_class] if hasattr(self, 'class_names') else str(predicted_class)
        
        # Create result dictionary
        threshold_exceeded = predictions[predicted_class] >= self.threshold
        
        result = {
            'file': os.path.basename(file_path),
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': float(predictions[predicted_class]),
            'threshold': float(self.threshold),
            'threshold_exceeded': threshold_exceeded,
            'predictions': {self.class_names[i]: float(predictions[i]) for i in range(len(predictions))}
        }
        
        # Print result
        print(f"Predicted: {predicted_label} (Class {predicted_class}) with confidence {predictions[predicted_class]:.4f}")
        print(f"Detection threshold: {self.threshold}")
        
        # Determine if prediction exceeds threshold
        threshold_exceeded = predictions[predicted_class] >= self.threshold
        status = "DETECTED" if threshold_exceeded else "BELOW THRESHOLD"
        print(f"Status: {status}")
        
        print("All predictions:")
        for i, prob in enumerate(predictions):
            class_name = self.class_names[i] if hasattr(self, 'class_names') else f"Class {i}"
            print(f"  {class_name}: {prob:.4f}")
        
        # Plot waveform and spectrogram
        if plot_results:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plot_waveform(audio, sr=self.sample_rate, title="Waveform")
            
            plt.subplot(2, 1, 2)
            librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=self.sample_rate), ref=np.max),
                                   sr=self.sample_rate, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            
            plt.tight_layout()
            plt.show()
            plt.close()  # Explicitly close the figure
        
        return result
    
    def batch_test(self, directory, samples=10):
        """
        Test model on multiple files from a directory.
        
        Args:
            directory: Directory containing audio files
            samples: Maximum number of samples to test
            
        Returns:
            results: List of test results
        """
        # Find audio files
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.ogg']):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"No audio files found in {directory}")
            return []
        
        # Limit number of samples if requested
        if samples and samples < len(audio_files):
            audio_files = audio_files[:samples]
        
        print(f"Testing {len(audio_files)} audio files")
        
        # Test each file
        results = []
        for file_path in tqdm(audio_files):
            try:
                result = self.test_file(file_path, plot_results=False)
                results.append(result)
                time.sleep(0.1)  # Brief pause between tests
            except Exception as e:
                print(f"Error testing {file_path}: {str(e)}")
        
        # Calculate accuracy - for non-keywords, we want to see them classified as 'negative'
        # A correct classification for a non-keyword is predicted_class = 0 (negative)
        correct_class = 0
        correct_threshold = 0
        total = len(results)
        
        for result in results:
            if result['predicted_class'] == 0:  # 0 = negative class
                correct_class += 1
            
            # Check if prediction is below threshold for non-negative classes
            if result['predicted_class'] != 0 and result['confidence'] < self.threshold:
                correct_threshold += 1
        
        accuracy_class = correct_class / total * 100 if total > 0 else 0
        accuracy_with_threshold = (correct_class + correct_threshold) / total * 100 if total > 0 else 0
        
        print(f"\nNon-Keyword Test Results:")
        print(f"Detection threshold: {self.threshold}")
        print(f"Correctly classified as negative: {correct_class}/{total} ({accuracy_class:.2f}%)")
        print(f"Correct with threshold applied: {correct_class + correct_threshold}/{total} ({accuracy_with_threshold:.2f}%)")
        
        # Print detailed results
        misclassified = []
        for result in results:
            if result['predicted_class'] != 0:
                misclassified.append((result['file'], result['predicted_label'], result['confidence']))
        
        if misclassified:
            print("\nMisclassified non-keywords:")
            for file, label, conf in misclassified:
                print(f"  {file}: falsely detected as '{label}' with confidence {conf:.4f}")
        
        return results

def test_model_non_keywords_exist():
    """Test if non-keywords directory and metadata exist."""
    
    # Path to non-keywords directory
    non_keywords_dir = os.path.join(KEYWORDS_DIR, 'non_keywords')
    
    # Check if directory exists
    if not os.path.exists(non_keywords_dir):
        print("❌ Non-keywords directory not found:", non_keywords_dir)
        return False
    
    # Check if metadata file exists
    metadata_path = os.path.join(KEYWORDS_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        print("❌ Metadata file not found:", metadata_path)
        return False
    
    # Check if metadata contains non-keywords
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'non_keywords' not in metadata:
        print("❌ Non-keywords not found in metadata")
        return False
    
    if 'samples' not in metadata['non_keywords'] or not metadata['non_keywords']['samples']:
        print("❌ No non-keyword samples found in metadata")
        return False
    
    print(f"✅ Found {metadata['non_keywords']['count']} non-keyword samples")
    
    # Check for sample files
    sample_count = 0
    for sample in metadata['non_keywords']['samples']:
        file_path = os.path.join(non_keywords_dir, sample['file'])
        if os.path.exists(file_path):
            sample_count += 1
    
    print(f"✅ {sample_count} non-keyword sample files exist out of {len(metadata['non_keywords']['samples'])} in metadata")
    
    return sample_count > 0

def main():
    parser = argparse.ArgumentParser(description='Test keyword detection model using non-keyword samples')
    parser.add_argument('--model', type=str,
                        help='Path to trained model file (.keras or .tflite format)')
    parser.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD,
                        help=f'Keyword to find the latest model for (default: "{DEFAULT_KEYWORD}")')
    parser.add_argument('--file', type=str, 
                        help='Path to a single non-keyword audio file to test (WAV format)')
    parser.add_argument('--dir', type=str, 
                        help='Directory containing non-keyword audio files to test')
    parser.add_argument('--samples', type=int, default=DEFAULT_TEST_SAMPLES, 
                        help=f'Maximum number of samples to test in batch mode (default: {DEFAULT_TEST_SAMPLES})')
    parser.add_argument('--keywords-dir', type=str, default=KEYWORDS_DIR, 
                        help=f'Directory containing keyword and non-keyword samples (default: {KEYWORDS_DIR})')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available trained models and exit')
    parser.add_argument('--check-exist', action='store_true',
                        help='Only check if non-keyword samples exist and exit without testing')
    parser.add_argument('--threshold', type=float, default=DEFAULT_DETECTION_THRESHOLD, 
                        help=f'Detection confidence threshold (0.0-1.0) (default: {DEFAULT_DETECTION_THRESHOLD})')
    args = parser.parse_args()
    
    # Just check if non-keywords exist if --check-exist is specified
    if args.check_exist:
        print("Checking for non-keywords setup...")
        if test_model_non_keywords_exist():
            print("✅ Non-keywords setup passed!")
            sys.exit(0)
        else:
            print("❌ Non-keywords setup failed!")
            sys.exit(1)
    
    # Convert relative paths to absolute paths if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List available models if requested
    if args.list_models:
        models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
        tflite_files = glob.glob(os.path.join(models_dir, '*.tflite'))
        if tflite_files:
            print("Available models:")
            for model in tflite_files:
                print(f"  {os.path.basename(model)}")
        else:
            print("No models found.")
        return
    
    # Get model path - either from --model parameter or by finding latest model for keyword
    model_path = None
    if args.model:
        model_path = normalize_model_path(args.model, script_dir)
    elif args.keyword:
        model_path = find_latest_model_by_keyword(args.keyword)
        if not model_path:
            print(f"Error: No model found for keyword '{args.keyword}'")
            return
    
    # Create the tester with the model
    tester = NonKeywordTester(model_path, args.keywords_dir, threshold=args.threshold)
    
    if args.file:
        # Test single file
        tester.test_file(args.file)
    
    elif args.dir:
        # Batch test
        tester.batch_test(args.dir, args.samples)
    
    else:
        # If no file or dir specified, use the non-keywords directory
        non_keywords_dir = os.path.join(args.keywords_dir, 'non_keywords')
        if os.path.exists(non_keywords_dir):
            print(f"Using non-keywords directory: {non_keywords_dir}")
            tester.batch_test(non_keywords_dir, args.samples)
        else:
            print(f"Error: Non-keywords directory not found: {non_keywords_dir}")

if __name__ == "__main__":
    main()
