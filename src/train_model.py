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
# Import from config and audio utils
from config import (VALIDATION_SPLIT, DEFAULT_NEGATIVE_SAMPLES_RATIO, SAMPLE_RATE,
                   FEATURE_PARAMS, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
                   EARLY_STOPPING_PATIENCE, DATA_DIR, MODELS_DIR, DEFAULT_KEYWORD)
from src.audio_utils import load_audio, extract_features

class KeywordDetectionModelTrainer:
    def __init__(self, data_dir, model_dir, sample_rate=SAMPLE_RATE, feature_params=None):
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
        self.feature_params = feature_params or FEATURE_PARAMS
        
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
    
    def prepare_dataset(self, keywords, negative_samples_ratio=DEFAULT_NEGATIVE_SAMPLES_RATIO, validation_split=VALIDATION_SPLIT):
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
        positive_samples = []
        
        for keyword in keywords:
            if keyword not in mixed_metadata:
                print(f"Warning: Keyword '{keyword}' not found in mixed data")
                continue
            
            keyword_samples = mixed_metadata[keyword]['samples']
            for sample in keyword_samples:
                # Sanitize keyword directory name
                sanitized_keyword = keyword.replace(' ', '_')
                sample_path = os.path.join(mixed_data_dir, sanitized_keyword, sample['file'])
                if os.path.exists(sample_path):
                    positive_samples.append({
                        'path': sample_path,
                        'label': self.keyword_to_index[keyword],
                        'keyword': keyword,
                        'metadata': sample
                    })
        
        negative_samples = []
        num_negative_samples = int(len(positive_samples) * negative_samples_ratio)
        
        mixed_non_keywords_added = 0
        
        # Load negative samples (non-keywords) in a similar way to positive samples
        if "non_keywords" in mixed_metadata:
            print("Adding mixed non-keyword samples as negative examples...")
            
            # Similar to how we handle keywords, get all non_keyword samples
            non_keyword_samples = mixed_metadata["non_keywords"]["samples"]
            for sample in non_keyword_samples:
                sample_path = os.path.join(mixed_data_dir, "non_keywords", sample['file'])
                if os.path.exists(sample_path):
                    # Extract the actual non-keyword name from metadata or filename
                    non_keyword_name = None
                    if 'non_keyword' in sample:
                        non_keyword_name = sample['non_keyword']
                    else:
                        # Try to parse from filename (non_keyword_play_propeller_neg1p4db_abab6342.wav)
                        parts = os.path.basename(sample_path).split('_')
                        if len(parts) > 2 and parts[0] == 'non' and parts[1] == 'keyword':
                            non_keyword_name = parts[2]
                    
                    negative_samples.append({
                        'path': sample_path,
                        'label': 0,  # 0 is for negative class
                        'keyword': None,
                        'non_keyword': non_keyword_name,
                        'metadata': sample,
                        'type': 'mixed-non-keyword'
                    })
                    mixed_non_keywords_added += 1
            
            if mixed_non_keywords_added > 0:
                print(f"Added {mixed_non_keywords_added} mixed non-keyword samples as negative examples")
        
        # Check if we have enough negative samples (at least 30% of requested amount)
        if len(negative_samples) < num_negative_samples * 0.3:
            raise ValueError(f"Not enough negative samples found. Found {len(negative_samples)} but need at least {int(num_negative_samples * 0.3)} (30% of the requested {num_negative_samples}). Please generate more non-keywords or use a lower negative samples ratio.")
            
        # Shuffle and limit negative samples to the desired ratio
        random.shuffle(negative_samples)
        negative_samples = negative_samples[:num_negative_samples]
        
        # Ensure we have at least some negative samples for validation
        min_negative_for_validation = max(2, int(len(negative_samples) * validation_split))
        if len(negative_samples) < min_negative_for_validation + 2:
            raise ValueError(f"Not enough negative samples for validation. Found {len(negative_samples)} but need at least {min_negative_for_validation + 2}. Please generate more non-keywords or use a lower validation split.")
        
        # Split positive and negative samples separately to ensure balanced validation set
        pos_split_idx = int(len(positive_samples) * (1 - validation_split))
        neg_split_idx = int(len(negative_samples) * (1 - validation_split))
        
        pos_train = positive_samples[:pos_split_idx]
        pos_val = positive_samples[pos_split_idx:]
        neg_train = negative_samples[:neg_split_idx]
        neg_val = negative_samples[neg_split_idx:]
        
        # Combine samples
        train_samples = pos_train + neg_train
        validation_samples = pos_val + neg_val
        
        # Shuffle training and validation samples
        random.shuffle(train_samples)
        random.shuffle(validation_samples)
        
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
        
        # Count sample types in training set
        negative_count = 0
        positive_count = 0
        
        for i, label in enumerate(dataset['train']['labels']):
            if label == 0:  # negative sample
                negative_count += 1
            else:
                positive_count += 1
        
        # Count validation samples        
        val_negative_count = sum(1 for label in dataset['validation']['labels'] if label == 0)
        val_positive_count = len(dataset['validation']['labels']) - val_negative_count
                
        # Print dataset statistics
        print(f"Dataset prepared with {len(dataset['train']['features'])} training and "
              f"{len(dataset['validation']['features'])} validation samples")
        print(f"Training set: {positive_count} positive samples, {negative_count} negative samples")
        print(f"Validation set: {val_positive_count} positive samples, {val_negative_count} negative samples")
        
        # Count negative sample types
        negative_types_count = {}
        for sample in negative_samples:
            sample_type = sample.get('type', 'unknown')
            if sample_type not in negative_types_count:
                negative_types_count[sample_type] = 0
            negative_types_count[sample_type] += 1
        
        print("Negative samples breakdown:")
        for sample_type, count in negative_types_count.items():
            print(f"  - {sample_type}: {count} samples")
              
        return dataset
    
    def _extract_features_from_file(self, file_path):
        """
        Extract features from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            features: Extracted features with consistent shape
        """
        try:
            audio, sr = load_audio(file_path, target_sr=self.sample_rate)
            
            # For negative samples (background noise), if the file is long, 
            # take a random segment of appropriate length
            if len(audio) > self.sample_rate * 2:  # If longer than 2 seconds
                # For background noise, take a random segment
                start = np.random.randint(0, len(audio) - self.sample_rate)
                audio = audio[start:start + self.sample_rate]
            
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
            
            # Define a fixed length for all features (adjust as needed)
            # For keyword detection, ~1 second of audio should be sufficient
            fixed_length = int(self.sample_rate / self.feature_params['hop_length'])
            
            # Pad or truncate to fixed length
            if mfccs.shape[0] > fixed_length:
                # Truncate longer sequences
                mfccs = mfccs[:fixed_length, :]
            elif mfccs.shape[0] < fixed_length:
                # Pad shorter sequences with zeros
                padding = np.zeros((fixed_length - mfccs.shape[0], mfccs.shape[1]))
                mfccs = np.vstack((mfccs, padding))
            
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
    
    def train_model(self, keywords, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, validation_split=VALIDATION_SPLIT, learning_rate=DEFAULT_LEARNING_RATE, negative_samples_ratio=DEFAULT_NEGATIVE_SAMPLES_RATIO):
        """
        Train a keyword detection model.
        
        Args:
            keywords: List of keywords to include in the model
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            learning_rate: Initial learning rate
            negative_samples_ratio: Ratio of negative samples to include (relative to positives)
            
        Returns:
            model: Trained TensorFlow model
            history: Training history
        """
        # Prepare dataset
        dataset = self.prepare_dataset(keywords, negative_samples_ratio=negative_samples_ratio, validation_split=validation_split)
        
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
                monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
        
        # Train model
        print(f"Training model for {epochs} epochs with batch size {batch_size}")
        
        # Add class weights to handle imbalanced datasets
        class_weights = None
        if negative_samples_ratio != 1.0:
            total_samples = len(dataset['train']['labels'])
            
            # Count samples per class
            class_counts = {}
            for label in dataset['train']['labels']:
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
            
            # Calculate weights for all classes
            class_weights = {}
            n_classes = len(class_counts)
            for label, count in class_counts.items():
                class_weights[label] = total_samples / (n_classes * count) if count > 0 else 1.0
            
            print(f"Using class weights: {class_weights}")
        
        history = model.fit(
            dataset['train']['features'], dataset['train']['labels'],
            validation_data=(dataset['validation']['features'], dataset['validation']['labels']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_accuracy = model.evaluate(dataset['validation']['features'], dataset['validation']['labels'])
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Replace spaces with underscores in keywords for filename
        sanitized_keywords = [kw.replace(' ', '_') for kw in keywords]
        model_name = f"keyword_detection_{'-'.join(sanitized_keywords)}_{timestamp}"
        model_path = os.path.join(self.model_dir, f"{model_name}.keras")
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
            'keras_path': model_path,
            'tflite_path': os.path.join(self.model_dir, f"{model_name}.tflite"),
            'optimized_tflite_path': os.path.join(self.model_dir, f"{model_name}_optimized.tflite"),
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
        # For quantization, we need a representative dataset of real inputs
        # Collect audio files for dataset
        audio_files = []
        keyword_dir = os.path.join(DATA_DIR, 'keywords')
        mixed_dir = os.path.join(DATA_DIR, 'mixed')
        
        # Get some keyword samples
        for keyword in os.listdir(keyword_dir):
            if os.path.isdir(os.path.join(keyword_dir, keyword)) and keyword != "metadata.json":
                keyword_files = [os.path.join(keyword_dir, keyword, f) 
                               for f in os.listdir(os.path.join(keyword_dir, keyword))[:20] 
                               if f.endswith('.wav')]
                audio_files.extend(keyword_files)
        
        # Get some mixed samples if available
        if os.path.exists(mixed_dir):
            for keyword in os.listdir(mixed_dir):
                if os.path.isdir(os.path.join(mixed_dir, keyword)) and keyword != "metadata.json":
                    mixed_files = [os.path.join(mixed_dir, keyword, f) 
                                 for f in os.listdir(os.path.join(mixed_dir, keyword))[:20] 
                                 if f.endswith('.wav')]
                    audio_files.extend(mixed_files)
        
        # Shuffle to get a good mix
        random.shuffle(audio_files)
        audio_files = audio_files[:100]  # Limit to 100 samples for efficiency
        
        print(f"Using {len(audio_files)} audio samples for model quantization")
        
        # Ensure the model has been built with concrete input shape before conversion
        if hasattr(model, 'input_shape') and model.input_shape[1:] is not None:
            expected_shape = tuple(model.input_shape[1:])
        elif hasattr(model, 'inputs') and model.inputs:
            expected_shape = tuple(model.inputs[0].shape[1:])
        else:
            # If we couldn't determine shape, use a default
            print("WARNING: Could not determine input shape from model, using default.")
            expected_shape = (98, 13)  # Common MFCC shape
            
        print(f"Model expects input shape: {expected_shape}")
            
        # Create an input tensor with exactly the right shape
        sample_input = np.random.random((1, *expected_shape)).astype(np.float32)
            
        # Run inference to ensure model is built with concrete input shape
        _ = model(sample_input)
        print("Model warmed up successfully with correct input shape")
            
        # Ensure the model has been built with concrete input shape
        # Extract features from at least one sample to use as calibration data
        calibration_data = []
        for file_path in audio_files[:5]:  # Just need a few samples
            try:
                features = self._extract_features_from_file(file_path)
                if features is not None:
                    # Add batch dimension
                    features_batch = np.expand_dims(features, axis=0).astype(np.float32)
                    calibration_data.append(features_batch)
                    # Call the model to ensure it's built with concrete input shape
                    _ = model(features_batch)
            except Exception as e:
                print(f"Error processing calibration sample: {str(e)}")
                continue
        
        # If we couldn't process any real samples, create a random input tensor
        if not calibration_data:
            input_shape = model.input_shape[1:]  # Get input shape excluding batch dimension
            random_input = np.random.random((1, *input_shape)).astype(np.float32)
            # Call the model to ensure it's built with concrete input shape
            _ = model(random_input)
        
        # Try to determine the model's expected input shape
        expected_shape = None
        
        # Try different methods to get the input shape
        if hasattr(model, 'input_shape') and model.input_shape[1:] is not None:
            expected_shape = tuple(model.input_shape[1:])
        elif hasattr(model, 'inputs') and model.inputs:
            expected_shape = tuple(model.inputs[0].shape[1:])
        else:
            # Try to get input shape from layers
            for layer in model.layers:
                if hasattr(layer, 'input_shape') and layer.input_shape is not None:
                    expected_shape = tuple(layer.input_shape[1:])
                    break
        
        # If shape still not determined, use a default
        if expected_shape is None:
            print("WARNING: Could not determine input shape, using default shape")
            expected_shape = (98, 13)  # Common MFCC shape
            
        print(f"Using input shape {expected_shape} for representative dataset")
        
        # Create a representative dataset generator that matches the expected input shape
        def representative_dataset():
            # First provide purely synthetic data with exact expected shape
            for _ in range(50):  # Always provide at least 50 synthetic samples
                data = np.random.random((1, *expected_shape)).astype(np.float32)
                yield [data.astype(np.float32)]
                
            # Then try to provide some real data samples if available
            real_samples_count = 0
            for file_path in audio_files[:50]:  # Limit to first 50 audio files
                try:
                    # Extract features using the same method as during training
                    features = self._extract_features_from_file(file_path)
                    if features is not None:
                        # Check if shapes match
                        if features.shape == expected_shape:
                            # Add batch dimension and ensure float32 format
                            features_batch = np.expand_dims(features, axis=0).astype(np.float32)
                            yield [features_batch]
                            real_samples_count += 1
                        else:
                            # Skip samples with wrong shape to avoid errors
                            pass
                except Exception as e:
                    # Just skip problematic samples quietly
                    continue
                    
            print(f"Provided {real_samples_count} real samples in representative dataset")
        
        # First create a standard (non-optimized) TFLite model
        print("Creating standard TFLite model...")
        standard_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        standard_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        standard_tflite_model = standard_converter.convert()
        
        # Save the standard TFLite model
        tflite_path = os.path.join(self.model_dir, f"{model_name}.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(standard_tflite_model)
        
        print(f"Standard TFLite model saved to {tflite_path}")
            
        # Now create fully optimized model for Android
        print("Creating optimized TFLite model for Android...")
        
        # Define the path for our optimized model
        optimized_tflite_path = os.path.join(self.model_dir, f"{model_name}_optimized.tflite")
        optimized_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        optimized_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        optimized_converter.representative_dataset = representative_dataset
        optimized_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        optimized_converter.inference_input_type = tf.int8
        optimized_converter.inference_output_type = tf.int8
            
        # Set experimental options for better compatibility
        optimized_converter._experimental_lower_tensor_list_ops = False
            
        optimized_tflite_model = optimized_converter.convert()
            
        # Save the optimized TFLite model
        with open(optimized_tflite_path, 'wb') as f:
            f.write(optimized_tflite_model)
            
        print(f"Fully optimized INT8 TFLite model saved to {optimized_tflite_path}")
        optimization_note = "INT8 quantization"
            
        # Update metadata to include optimized model path and note
        if hasattr(self, 'model_metadata'):
            # Find this model in metadata
            for model_info in self.model_metadata['models']:
                if model_info['name'] == model_name:
                    model_info['optimized_tflite_path'] = optimized_tflite_path
                    model_info['optimization_note'] = optimization_note
                    break
    
    def _plot_training_history(self, history, model_name):
        """
        Plot training history.
        
        Args:
            history: Training history
            model_name: Name of the model
        """
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(self.model_dir, 'plots')
        plots_dir = plots_dir.replace(' ', '_')  # Replace spaces with underscores in directory name
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
        # Create plots directory if it doesn't exist (use sanitized name)
        plots_dir = os.path.join(self.model_dir, 'plots')
        plots_dir = plots_dir.replace(' ', '_')  # Replace spaces with underscores in directory name
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get predictions
        predictions = model.predict(validation_data['features'])
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Create class names
        class_names = ['negative'] + keywords
        
        # Check if we have both negative and positive samples in validation data
        has_negative = 0 in validation_data['labels']
        has_positive = any(label > 0 for label in validation_data['labels'])
        
        if not has_negative:
            print("Warning: No negative samples in validation data!")
        
        if not has_positive:
            print("Warning: No positive samples in validation data!")
        
        # Confusion matrix
        cm = confusion_matrix(validation_data['labels'], predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Calculate and display metrics for each class
        for i in range(len(class_names)):
            if i in np.unique(validation_data['labels']):
                true_positive = cm[i, i]
                false_negative = np.sum(cm[i, :]) - true_positive
                false_positive = np.sum(cm[:, i]) - true_positive
                true_negative = np.sum(cm) - true_positive - false_negative - false_positive
                
                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"Class: {class_names[i]}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  Specificity: {specificity:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  Support: {np.sum(validation_data['labels'] == i)}")
        
        # Save confusion matrix
        cm_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        # Get unique classes present in validation data
        unique_labels = np.unique(np.concatenate([validation_data['labels'], predicted_classes]))
        
        # Classification report
        try:
            # Check if we have more than one class in the actual data
            if len(unique_labels) == 1:
                print(f"Warning: Only one class ({class_names[unique_labels[0]]}) present in validation data.")
                # Create a simplified report manually
                report = {
                    f"{class_names[unique_labels[0]]}": {
                        "precision": 1.0 if np.all(validation_data['labels'] == predicted_classes) else 0.0,
                        "recall": 1.0 if np.all(validation_data['labels'] == predicted_classes) else 0.0,
                        "f1-score": 1.0 if np.all(validation_data['labels'] == predicted_classes) else 0.0,
                        "support": len(validation_data['labels'])
                    },
                    "accuracy": np.mean(validation_data['labels'] == predicted_classes),
                    "macro avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": len(validation_data['labels'])},
                    "weighted avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": len(validation_data['labels'])}
                }
            else:
                report = classification_report(
                    validation_data['labels'], 
                    predicted_classes, 
                    labels=unique_labels,
                    target_names=[class_names[i] for i in unique_labels],
                    output_dict=True
                )
            
            # Save report
            report_path = os.path.join(self.model_dir, f"{model_name}_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            print(f"Error generating classification report: {str(e)}")
            # Log basic accuracy
            accuracy = np.mean(validation_data['labels'] == predicted_classes)
            print(f"Validation accuracy: {accuracy:.4f}")
            
            # Create a simplified JSON report for consistency
            simple_report = {
                "accuracy": float(accuracy),
                "error": str(e)
            }
            report_path = os.path.join(self.model_dir, f"{model_name}_report.json")
            with open(report_path, 'w') as f:
                json.dump(simple_report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Train a neural network model for keyword detection')
    parser.add_argument('--keywords', type=str, nargs='+', default=[DEFAULT_KEYWORD], 
                        help=f'Keywords to train the model to detect (default: "{DEFAULT_KEYWORD}")')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, 
                        help=f'Directory containing mixed audio training data (default: {DATA_DIR})')
    parser.add_argument('--model-dir', type=str, default=MODELS_DIR, 
                        help=f'Directory to save trained models (default: {MODELS_DIR})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, 
                        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, 
                        help=f'Training batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, 
                        help=f'Initial learning rate for optimizer (default: {DEFAULT_LEARNING_RATE})')
    parser.add_argument('--negative-samples-ratio', type=float, default=DEFAULT_NEGATIVE_SAMPLES_RATIO, 
                        help=f'Ratio of negative samples to include relative to positives (default: {DEFAULT_NEGATIVE_SAMPLES_RATIO}x)')
    args = parser.parse_args()
    
    trainer = KeywordDetectionModelTrainer(args.data_dir, args.model_dir)
    trainer.train_model(
        args.keywords,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        negative_samples_ratio=args.negative_samples_ratio
    )

if __name__ == "__main__":
    main()
