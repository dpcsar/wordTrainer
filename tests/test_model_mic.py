"""
Test keyword detection model using microphone input.
"""

import os
import argparse
import numpy as np
import json
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
import time
import sys
import threading
import glob
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (MODELS_DIR, DEFAULT_DETECTION_THRESHOLD, SAMPLE_RATE, 
                    DEFAULT_AUDIO_DURATION, DEFAULT_BUFFER_DURATION, DEFAULT_KEYWORD,
                    BASE_DIR)
from src.audio_utils import extract_features
from src.path_utils import normalize_model_path, find_latest_model_by_keyword

import sounddevice as sd
import queue

class MicrophoneDetector:
    def __init__(self, model_path, threshold=None, sample_rate=SAMPLE_RATE):
        """
        Initialize MicrophoneDetector.
        
        Args:
            model_path: Path to trained model (.h5 or .tflite)
            threshold: Detection threshold (None to use default from config)
            sample_rate: Audio sample rate
        """
        self.model_path = model_path
        # Use threshold from config if none provided
        self.threshold = threshold if threshold is not None else DEFAULT_DETECTION_THRESHOLD
        self.sample_rate = sample_rate
        
        # Audio settings
        self.audio_duration = DEFAULT_AUDIO_DURATION  # seconds
        self.num_samples = int(self.audio_duration * sample_rate)
        # Define a smaller buffer just for processing
        self.buffer_duration = DEFAULT_BUFFER_DURATION  # seconds for audio processing
        self.buffer_samples = int(self.buffer_duration * sample_rate)
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        
        # Sample window (sliding window over audio buffer)
        self.window = np.zeros(self.num_samples, dtype=np.float32)
        
        # Queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # No longer saving recordings
        
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
            model_path: Path to .h5 model file
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
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio input.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time info
            status: Status flags
        """
        if status:
            print(f"Status: {status}")
        
        # Put audio data in queue
        self.audio_queue.put(indata.copy())
    
    def process_audio(self):
        """Process audio data from queue and run detection."""
        print("Processing audio... Press Ctrl+C to stop")
        
        try:
            while True:
                # Get audio chunk from queue
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    
                    # Convert to mono if needed
                    if chunk.shape[1] > 1:
                        chunk = np.mean(chunk, axis=1)
                    
                    # Add to audio buffer (shift and append)
                    chunk_flat = chunk.flatten()
                    self.audio_buffer = np.roll(self.audio_buffer, -len(chunk_flat))
                    self.audio_buffer[-len(chunk_flat):] = chunk_flat
                    
                    # Update detection window
                    self.window = self.audio_buffer[-self.num_samples:]
                    
                    # Detect keywords
                    self.detect_keyword(self.window)
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("Stopping audio processing")
    
    def extract_features(self, audio_data):
        """
        Extract features from audio data.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            features: Extracted features
        """
        # Extract MFCCs
        n_mfcc = self.feature_params.get('n_mfcc', 13)
        n_fft = self.feature_params.get('n_fft', 512)
        hop_length = self.feature_params.get('hop_length', 160)
        
        mfccs = extract_features(
            audio_data, 
            sr=self.sample_rate, 
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Transpose to get time on the first axis
        mfccs = mfccs.T
        
        # Get input shape from model if using TFLite
        expected_time_steps = None
        if self.model_type == 'tflite' and hasattr(self, 'input_details'):
            expected_time_steps = self.input_details[0]['shape'][1]
        
        # Reshape if necessary
        if expected_time_steps and mfccs.shape[0] != expected_time_steps:
            # Resize features to match expected input shape
            from scipy import signal
            
            # Use resampling to resize the time dimension
            if mfccs.shape[0] > 0 and expected_time_steps > 0:
                resampled_mfccs = np.zeros((expected_time_steps, mfccs.shape[1]))
                for i in range(mfccs.shape[1]):
                    resampled_mfccs[:, i] = signal.resample(mfccs[:, i], expected_time_steps)
                mfccs = resampled_mfccs
                # Feature resizing done silently
        
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
            # Get expected input shape and data type
            input_shape = self.input_details[0]['shape']
            input_type = self.input_details[0]['dtype']
            is_quantized = input_type == np.int8
            
            # Check if dimensions match and reshape if needed
            if features.shape[0] != input_shape[1] or features.shape[1] != input_shape[2]:
                # If we can't reshape, we need to extract features again at the correct size
                # Silently handle feature shape mismatch
                if hasattr(self, 'extract_features'):
                    # This would have already been handled in extract_features()
                    pass
            
            # Ensure features have correct shape for input (add batch dimension)
            features_batch = np.expand_dims(features, axis=0)
            
            # For quantized models, convert float values to int8
            if is_quantized:
                if 'quantization_parameters' in self.input_details[0]:
                    quantization_params = self.input_details[0]['quantization_parameters']
                    scale = quantization_params['scales'][0]
                    zero_point = quantization_params['zero_points'][0]
                    
                    # Convert float to int8 using scale and zero point
                    features_batch = (features_batch / scale + zero_point).astype(np.int8)
                else:
                    # If quantization parameters are not available, use a simpler estimation
                    features_batch = np.clip(features_batch * 10, -128, 127).astype(np.int8)
            else:
                features_batch = features_batch.astype(np.float32)
            
            try:
                # Set input tensor
                self.interpreter.set_tensor(self.input_details[0]['index'], features_batch)
                
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
            except Exception as e:
                print(f"Error during prediction: {e}")
                # Return empty predictions (all zeros) as fallback
                num_classes = self.output_details[0]['shape'][-1]
                return np.zeros(num_classes)
    
    def detect_keyword(self, audio_data):
        """
        Detect keyword in audio data.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            result: Detection result
        """
        # Extract features
        features = self.extract_features(audio_data)
        
        # Make prediction
        predictions = self.predict(features)
        
        # Get predicted class
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        predicted_label = self.class_names[predicted_class] if hasattr(self, 'class_names') else str(predicted_class)
        
        # Show debug info for all predictions
        keyword_confidences = {}
        for i in range(len(predictions)):
            class_label = self.class_names[i] if hasattr(self, 'class_names') else f"Class {i}"
            keyword_confidences[class_label] = float(predictions[i])
        
        # Print the confidence for both negative and positive classes for debugging
        neg_confidence = predictions[0] if len(predictions) > 0 else 0
        pos_confidences = {self.class_names[i]: predictions[i] for i in range(1, len(predictions))}
        
        # Only report keywords (not negative class) above threshold
        if predicted_class > 0 and confidence >= self.threshold:
            print(f"Detected: {predicted_label} (Confidence: {confidence:.4f}, Negative: {neg_confidence:.4f})")
        elif neg_confidence >= self.threshold:
            print(f"No keyword detected (Negative confidence: {neg_confidence:.4f})")
        
        # Return result
        return {
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'negative_confidence': float(neg_confidence),
            'predictions': keyword_confidences
        }
    
    def start_detection(self, device=None):
        """
        Start keyword detection from microphone.
        
        Args:
            device: Audio device index
        """
        print(f"Starting keyword detection with keywords: {self.keywords}")
        print(f"Detection threshold: {self.threshold}")
        
        # List available devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device_info in enumerate(devices):
            print(f"  {i}: {device_info['name']} (in: {device_info['max_input_channels']}, out: {device_info['max_output_channels']})")
        
        # Use default device if none specified
        if device is not None:
            sd.default.device = device
            print(f"Using device {device}: {sd.query_devices(device)['name']}")
        
        # Start audio processing thread
        audio_thread = threading.Thread(target=self.process_audio)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Start audio stream
        with sd.InputStream(callback=self.audio_callback, channels=1, 
                           samplerate=self.sample_rate, blocksize=int(self.sample_rate * 0.1)):
            print("\nListening... Press Ctrl+C to stop")
            
            try:
                while True:
                    time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("Stopping...")

def main():
    parser = argparse.ArgumentParser(description='Test keyword detection model using live microphone input')
    parser.add_argument('--model', type=str,
                        help='Path to trained model file (.keras or .tflite format)')
    parser.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD,
                        help=f'Keyword to find the latest model for (default: "{DEFAULT_KEYWORD}")')
    parser.add_argument('--threshold', type=float, default=DEFAULT_DETECTION_THRESHOLD, 
                        help=f'Detection confidence threshold (0.0-1.0) (default: {DEFAULT_DETECTION_THRESHOLD})')
    parser.add_argument('--device', type=int, 
                        help='Audio input device index (use system default if not specified)')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available trained models and exit')
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        tflite_files = glob.glob(os.path.join(MODELS_DIR, '*.tflite'))
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
        model_path = normalize_model_path(args.model)
    elif args.keyword:
        model_path = find_latest_model_by_keyword(args.keyword)
        if not model_path:
            print(f"Error: No model found for keyword '{args.keyword}'")
            return
    
    detector = MicrophoneDetector(model_path, args.threshold)
    detector.start_detection(args.device)

if __name__ == "__main__":
    main()
