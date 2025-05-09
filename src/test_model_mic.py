"""
Test keyword detection model using microphone input.
"""

import os
import argparse
import numpy as np
import json
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import soundfile as sf
import queue
import sys
import threading
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.audio_utils import extract_features

class MicrophoneDetector:
    def __init__(self, model_path, threshold=0.5, sample_rate=16000):
        """
        Initialize MicrophoneDetector.
        
        Args:
            model_path: Path to trained model (.h5 or .tflite)
            threshold: Detection threshold
            sample_rate: Audio sample rate
        """
        self.model_path = model_path
        self.threshold = threshold
        self.sample_rate = sample_rate
        
        # Audio settings
        self.audio_duration = 2  # seconds
        self.buffer_duration = 10  # seconds for visualization
        self.num_samples = int(self.audio_duration * sample_rate)
        self.buffer_samples = int(self.buffer_duration * sample_rate)
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        
        # Sample window (sliding window over audio buffer)
        self.window = np.zeros(self.num_samples, dtype=np.float32)
        
        # Queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Create directory for saving recordings
        self.recordings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'recordings')
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Load model metadata
        model_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            # Find this specific model in the metadata
            model_name = os.path.basename(model_path).split('.')[0]
            model_info = None
            
            for model in self.model_metadata['models']:
                if model['name'] == model_name:
                    model_info = model
                    break
            
            if model_info:
                self.keywords = model_info['keywords']
                self.feature_params = model_info.get('feature_params', {})
                self.class_names = ['negative'] + self.keywords
                print(f"Model keywords: {self.keywords}")
            else:
                print(f"Warning: Model {model_name} not found in metadata")
                self.keywords = []
                self.feature_params = {}
                # Try to infer from model
                if model_path.endswith('.tflite'):
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                    output_details = interpreter.get_output_details()
                    num_classes = output_details[0]['shape'][-1]
                    self.keywords = [f"keyword_{i}" for i in range(1, num_classes)]
                    self.class_names = ['negative'] + self.keywords
        else:
            print(f"Warning: Model metadata not found at {metadata_path}")
            self.keywords = []
            self.feature_params = {}
            # Try to infer from model
            if model_path.endswith('.tflite'):
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                output_details = interpreter.get_output_details()
                num_classes = output_details[0]['shape'][-1]
                self.keywords = [f"keyword_{i}" for i in range(1, num_classes)]
                self.class_names = ['negative'] + self.keywords
        
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
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Loaded TFLite model from {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
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
            if input_shape[0] == 1:  # Batch size of 1
                features = np.expand_dims(features, axis=0).astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            return predictions[0]
    
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
        
        # Only report keywords (not negative class) above threshold
        if predicted_class > 0 and confidence >= self.threshold:
            print(f"Detected: {predicted_label} (Confidence: {confidence:.4f})")
            
            # Save recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{predicted_label}_{confidence:.2f}_{timestamp}.wav"
            filepath = os.path.join(self.recordings_dir, filename)
            
            sf.write(filepath, audio_data, self.sample_rate)
            print(f"Saved recording to {filepath}")
        
        # Return result for visualization
        return {
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'predictions': {
                self.class_names[i] if hasattr(self, 'class_names') else f"Class {i}": 
                float(predictions[i]) for i in range(len(predictions))
            }
        }
    
    def start_detection(self, device=None, visualization=True):
        """
        Start keyword detection from microphone.
        
        Args:
            device: Audio device index
            visualization: Whether to show visualization
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
        
        # Set up visualization
        if visualization:
            self.setup_visualization()
        
        # Start audio stream
        with sd.InputStream(callback=self.audio_callback, channels=1, 
                           samplerate=self.sample_rate, blocksize=int(self.sample_rate * 0.1)):
            print("\nListening... Press Ctrl+C to stop")
            
            try:
                if visualization:
                    plt.show()
                else:
                    while True:
                        time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("Stopping...")
    
    def setup_visualization(self):
        """Set up real-time visualization of audio and detections."""
        # Create figure with two subplots
        self.fig, (self.ax_wave, self.ax_pred) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('Keyword Detection')
        
        # Set up waveform plot
        self.x_wave = np.arange(self.buffer_samples) / self.sample_rate
        self.line_wave, = self.ax_wave.plot(self.x_wave, np.zeros_like(self.x_wave))
        
        # Highlight the current detection window
        window_start = (self.buffer_samples - self.num_samples) / self.sample_rate
        window_end = self.buffer_duration
        self.ax_wave.axvspan(window_start, window_end, color='yellow', alpha=0.2)
        
        self.ax_wave.set_xlim(0, self.buffer_duration)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_title('Audio Waveform')
        self.ax_wave.set_xlabel('Time (s)')
        self.ax_wave.set_ylabel('Amplitude')
        
        # Set up prediction bar plot
        self.ax_pred.set_title('Keyword Detection Confidence')
        self.ax_pred.set_xlabel('Class')
        self.ax_pred.set_ylabel('Confidence')
        
        # Initialize bar plot with class names
        num_classes = len(self.class_names) if hasattr(self, 'class_names') else 2
        self.x_pred = np.arange(num_classes)
        self.bar_pred = self.ax_pred.bar(
            self.x_pred, 
            np.zeros(num_classes),
            tick_label=self.class_names if hasattr(self, 'class_names') else [f"Class {i}" for i in range(num_classes)]
        )
        
        self.ax_pred.set_ylim(0, 1)
        self.ax_pred.axhline(y=self.threshold, color='r', linestyle='-', alpha=0.5)
        
        # Add threshold label
        self.ax_pred.text(
            0.95, self.threshold + 0.05, 
            f'Threshold: {self.threshold}', 
            ha='right', va='bottom', color='red'
        )
        
        plt.tight_layout()
        
        # Set up animation
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=100, blit=False)
    
    def update_plots(self, frame):
        """Update visualization plots."""
        # Update waveform plot
        self.line_wave.set_ydata(self.audio_buffer)
        
        # Run detection on current window
        result = self.detect_keyword(self.window)
        
        # Update prediction bars
        for i, bar in enumerate(self.bar_pred):
            class_name = self.class_names[i] if hasattr(self, 'class_names') else f"Class {i}"
            bar.set_height(result['predictions'][class_name])
            
            # Color based on threshold
            if i > 0 and result['predictions'][class_name] >= self.threshold:
                bar.set_color('green')
            else:
                bar.set_color('blue')
        
        # Return artists that were updated
        return [self.line_wave] + list(self.bar_pred)

def main():
    parser = argparse.ArgumentParser(description='Test keyword detection model using microphone')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to trained model (.h5 or .tflite)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Detection threshold')
    parser.add_argument('--device', type=int, 
                        help='Audio device index')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths if needed
    if not os.path.isabs(args.model):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model = os.path.abspath(os.path.join(script_dir, args.model))
    
    detector = MicrophoneDetector(args.model, args.threshold)
    detector.start_detection(args.device, not args.no_viz)

if __name__ == "__main__":
    main()
