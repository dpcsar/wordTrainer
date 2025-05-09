"""
Train a keyword detection model using TensorFlow Lite.
"""

import os
import argparse
import numpy as np
import json
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import sys
import random

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.audio_utils import load_audio, extract_features, augment_audio

class KeywordDetectionModelTrainer:
    def __init__(self, data_dir, model_dir, sample_rate=16000, feature_params=None):
        """
        Initialize KeywordDetectionModelTrainer.
        
        Args:
            data_dir: Directory containing audio data
            model_dir: Directory to save trained models
            sample_rate: Audio sample rate
            feature_params: Dictionary of feature extraction parameters
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        
        # Set default feature parameters if not provided
        self.feature_params = feature_params or {
            'n_mfcc': 13,
            'n_fft': 512,
            'hop_length': 160,
            'window_length_ms': 30,
            'window_step_ms': 10,
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Path to model metadata file
        self.metadata_path = os.path.join(model_dir, 'model_metadata.json')
        
        # Load existing metadata if available
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        else:
            self.model_metadata = {
                'models': [],
                'feature_params': self.feature_params,
            }
    
    def _save_metadata(self):
        """Save model metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
    
    def prepare_dataset(self, keywords, negative_samples_ratio=1.0, validation_split=0.2):
        """
        Prepare dataset for training.
        
        Args:
            keywords: List of keywords to include in the model
            negative_samples_ratio: Ratio of negative samples to include (relative to positives)
            validation_split: Fraction of data to use for validation
            
        Returns:
            dataset: Dictionary containing training and validation data
        """
        mixed_data_dir = os.path.join(self.data_dir, 'mixed')
        
        # Check if mixed data directory exists
        if not os.path.exists(mixed_data_dir):
            raise ValueError(f"Mixed data directory not found: {mixed_data_dir}")
        
        # Load mixed data metadata
        mixed_metadata_path = os.path.join(mixed_data_dir, 'metadata.json')
        if not os.path.exists(mixed_metadata_path):
            raise ValueError(f"Mixed data metadata not found: {mixed_metadata_path}")
        
        with open(mixed_metadata_path, 'r') as f:
            mixed_metadata = json.load(f)
        
        # Prepare dataset
        dataset = {
            'train': {
                'features': [],
                'labels': [],
                'filenames': [],
            },
            'validation': {
                'features': [],
                'labels': [],
                'filenames': [],
            }
        }
        
        # Map keywords to indices
        self.keyword_to_index = {}
        for i, keyword in enumerate(keywords):
            self.keyword_to_index[keyword] = i + 1  # Reserve 0 for negative class
        
        # Load positive samples (keywords)
        all_samples = []
        
        for keyword in keywords:
            if keyword not in mixed_metadata:
                print(f"Warning: Keyword '{keyword}' not found in mixed data")
                continue
            
            keyword_samples = mixed_metadata[keyword]['samples']
            for sample in keyword_samples:
                sample_path = os.path.join(mixed_data_dir, keyword, sample['file'])
                if os.path.exists(sample_path):
                    all_samples.append({
                        'path': sample_path,
                        'label': self.keyword_to_index[keyword],
                        'keyword': keyword,
                        'metadata': sample
                    })
        
        # Shuffle all samples
        random.shuffle(all_samples)
        
        # Split into training and validation
        split_idx = int(len(all_samples) * (1 - validation_split))
        train_samples = all_samples[:split_idx]
        validation_samples = all_samples[split_idx:]
        
        print(f"Preparing dataset with {len(train_samples)} training and {len(validation_samples)} validation samples")
        
        # Extract features for training samples
        for sample in tqdm(train_samples, desc="Processing training samples"):
            features = self._extract_features_from_file(sample['path'])
            if features is not None:
                dataset['train']['features'].append(features)
                dataset['train']['labels'].append(sample['label'])
                dataset['train']['filenames'].append(os.path.basename(sample['path']))
        
        # Extract features for validation samples
        for sample in tqdm(validation_samples, desc="Processing validation samples"):
            features = self._extract_features_from_file(sample['path'])
            if features is not None:
                dataset['validation']['features'].append(features)
                dataset['validation']['labels'].append(sample['label'])
                dataset['validation']['filenames'].append(os.path.basename(sample['path']))
        
        # Convert to numpy arrays
        for split in ['train', 'validation']:
            dataset[split]['features'] = np.array(dataset[split]['features'])
            dataset[split]['labels'] = np.array(dataset[split]['labels'])
        
        print(f"Dataset prepared with {len(dataset['train']['features'])} training and "
              f"{len(dataset['validation']['features'])} validation samples")
              
        return dataset
    
    def _extract_features_from_file(self, file_path):
        """
        Extract features from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            features: Extracted features
        """
        try:
            audio, sr = load_audio(file_path, target_sr=self.sample_rate)
            
            # Extract MFCCs
            mfccs = extract_features(
                audio, 
                sr=self.sample_rate,
                n_mfcc=self.feature_params['n_mfcc'],
                n_fft=self.feature_params['n_fft'],
                hop_length=self.feature_params['hop_length']
            )
            
            # Transpose to get time on the first axis
            mfccs = mfccs.T
            
            return mfccs
        
        except Exception as e:
            print(f"Error extracting features from {file_path}: {str(e)}")
            return None
    
    def build_model(self, input_shape, num_classes):
        """
        Build a keyword detection model.
        
        Args:
            input_shape: Shape of input features (time_steps, n_mfcc)
            num_classes: Number of output classes (keywords + 1 for negative)
            
        Returns:
            model: TensorFlow model
        """
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),
            
            # Convolutional layers
            tf.keras.layers.Reshape((*input_shape, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model(self, keywords, epochs=50, batch_size=32, validation_split=0.2, learning_rate=0.001):
        """
        Train a keyword detection model.
        
        Args:
            keywords: List of keywords to include in the model
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            learning_rate: Initial learning rate
            
        Returns:
            model: Trained TensorFlow model
            history: Training history
        """
        # Prepare dataset
        dataset = self.prepare_dataset(keywords, validation_split=validation_split)
        
        # Get input shape and number of classes
        input_shape = dataset['train']['features'][0].shape
        num_classes = len(keywords) + 1  # Add one for negative class
        
        print(f"Building model with input shape {input_shape} and {num_classes} output classes")
        
        # Build model
        model = self.build_model(input_shape, num_classes)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
        
        # Train model
        print(f"Training model for {epochs} epochs with batch size {batch_size}")
        
        history = model.fit(
            dataset['train']['features'], dataset['train']['labels'],
            validation_data=(dataset['validation']['features'], dataset['validation']['labels']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_accuracy = model.evaluate(dataset['validation']['features'], dataset['validation']['labels'])
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"keyword_detection_{'-'.join(keywords)}_{timestamp}"
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        model.save(model_path)
        
        # Save TFLite model
        self._convert_to_tflite(model, model_name)
        
        # Plot training history
        self._plot_training_history(history, model_name)
        
        # Evaluate model in detail
        self._evaluate_model(model, dataset['validation'], keywords, model_name)
        
        # Update metadata
        model_info = {
            'name': model_name,
            'keywords': keywords,
            'input_shape': list(input_shape),
            'num_classes': num_classes,
            'validation_accuracy': float(val_accuracy),
            'validation_loss': float(val_loss),
            'feature_params': self.feature_params,
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
            },
            'timestamp': timestamp,
            'h5_path': model_path,
            'tflite_path': os.path.join(self.model_dir, f"{model_name}.tflite"),
        }
        
        self.model_metadata['models'].append(model_info)
        self._save_metadata()
        
        return model, history
    
    def _convert_to_tflite(self, model, model_name):
        """
        Convert TensorFlow model to TFLite format.
        
        Args:
            model: TensorFlow model
            model_name: Name of the model
        """
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_path = os.path.join(self.model_dir, f"{model_name}.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {tflite_path}")
        
        # Create a quantized version for even smaller size
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Create a representative dataset from random noise
        def representative_dataset():
            for _ in range(100):
                data = np.random.random((1, *model.input.shape[1:]))
                yield [data.astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        
        # Convert to quantized model
        try:
            quantized_model = converter.convert()
            
            # Save the quantized model
            quantized_path = os.path.join(self.model_dir, f"{model_name}_quantized.tflite")
            with open(quantized_path, 'wb') as f:
                f.write(quantized_model)
            
            print(f"Quantized TFLite model saved to {quantized_path}")
        
        except Exception as e:
            print(f"Error creating quantized model: {str(e)}")
    
    def _plot_training_history(self, history, model_name):
        """
        Plot training history.
        
        Args:
            history: Training history
            model_name: Name of the model
        """
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(self.model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, f"{model_name}_history.png")
        plt.savefig(plot_path)
        plt.close()
    
    def _evaluate_model(self, model, validation_data, keywords, model_name):
        """
        Evaluate model in detail.
        
        Args:
            model: TensorFlow model
            validation_data: Validation dataset
            keywords: List of keywords
            model_name: Name of the model
        """
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(self.model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get predictions
        predictions = model.predict(validation_data['features'])
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Create class names
        class_names = ['negative'] + keywords
        
        # Confusion matrix
        cm = confusion_matrix(validation_data['labels'], predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix
        cm_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        # Classification report
        report = classification_report(
            validation_data['labels'], 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        # Save report
        report_path = os.path.join(self.model_dir, f"{model_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Train a keyword detection model')
    parser.add_argument('--keywords', type=str, nargs='+', required=True, 
                        help='Keywords to detect')
    parser.add_argument('--data-dir', type=str, default='../data', 
                        help='Directory containing audio data')
    parser.add_argument('--model-dir', type=str, default='../models', 
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, 
                        help='Initial learning rate')
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.abspath(os.path.join(script_dir, args.model_dir))
    
    trainer = KeywordDetectionModelTrainer(args.data_dir, args.model_dir)
    trainer.train_model(
        args.keywords,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()
