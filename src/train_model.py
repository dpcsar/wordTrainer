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
from config import VALIDATION_SPLIT, DEFAULT_NEGATIVE_SAMPLES_RATIO
from src.audio_utils import load_audio, extract_features

class KeywordDetectionModelTrainer:
    def __init__(self, data_dir, model_dir, sample_rate=None, feature_params=None):
        """
        Initialize KeywordDetectionModelTrainer.
        
        Args:
            data_dir: Directory containing audio data
            model_dir: Directory to save trained models
            sample_rate: Audio sample rate
            feature_params: Dictionary of feature extraction parameters
        """
        from config import SAMPLE_RATE, FEATURE_PARAMS
        
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.sample_rate = sample_rate if sample_rate is not None else SAMPLE_RATE
        
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
        backgrounds_dir = os.path.join(self.data_dir, 'backgrounds')
        non_keywords_dir = os.path.join(self.data_dir, 'keywords', 'non_keywords')
        
        # Check if mixed data directory exists
        if not os.path.exists(mixed_data_dir):
            raise ValueError(f"Mixed data directory not found: {mixed_data_dir}")
        
        # Load mixed data metadata
        mixed_metadata_path = os.path.join(mixed_data_dir, 'metadata.json')
        if not os.path.exists(mixed_metadata_path):
            raise ValueError(f"Mixed data metadata not found: {mixed_metadata_path}")
        
        with open(mixed_metadata_path, 'r') as f:
            mixed_metadata = json.load(f)
        
        # Load background noise metadata
        backgrounds_metadata_path = os.path.join(backgrounds_dir, 'metadata.json')
        if os.path.exists(backgrounds_metadata_path):
            with open(backgrounds_metadata_path, 'r') as f:
                backgrounds_metadata = json.load(f)
        else:
            print(f"Warning: Background metadata not found at {backgrounds_metadata_path}")
            backgrounds_metadata = {}
            
        # Load non-keywords metadata if available
        non_keywords_metadata = {}
        non_keywords_metadata_path = os.path.join(self.data_dir, 'keywords', 'metadata.json')
        if os.path.exists(non_keywords_metadata_path):
            with open(non_keywords_metadata_path, 'r') as f:
                keywords_metadata = json.load(f)
                if "non_keywords" in keywords_metadata:
                    non_keywords_metadata = {"non_keywords": keywords_metadata["non_keywords"]}
                    print(f"Loaded {keywords_metadata['non_keywords']['count']} non-keyword samples")
        else:
            print(f"Warning: Non-keywords metadata not found at {non_keywords_metadata_path}")
        
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
                sample_path = os.path.join(mixed_data_dir, keyword, sample['file'])
                if os.path.exists(sample_path):
                    positive_samples.append({
                        'path': sample_path,
                        'label': self.keyword_to_index[keyword],
                        'keyword': keyword,
                        'metadata': sample
                    })
        
        # Create negative samples using background noise and non-keywords
        negative_samples = []
        num_negative_samples = int(len(positive_samples) * negative_samples_ratio)
        print(f"Creating {num_negative_samples} negative samples based on ratio {negative_samples_ratio}")
        
        # Add non-keyword samples as negative examples (these are actual words, better for discrimination)
        if "non_keywords" in non_keywords_metadata:
            print("Adding non-keyword samples as negative examples...")
            for sample in non_keywords_metadata["non_keywords"]["samples"]:
                file_path = os.path.join(self.data_dir, 'keywords', 'non_keywords', sample['file'])
                if os.path.exists(file_path):
                    negative_samples.append({
                        'path': file_path,
                        'label': 0,  # 0 is for negative class
                        'keyword': None,
                        'non_keyword': sample.get('non_keyword'),
                        'metadata': sample,
                        'type': 'non-keyword'
                    })
        
        # Add background noise as negative samples
        print("Adding background noise samples as negative examples...")
        for noise_type in ['propeller', 'jet', 'cockpit']:
            if noise_type in backgrounds_metadata:
                for sample in backgrounds_metadata[noise_type]['samples']:
                    file_path = os.path.join(backgrounds_dir, noise_type, sample['file'])
                    if os.path.exists(file_path):
                        negative_samples.append({
                            'path': file_path,
                            'label': 0,  # 0 is for negative class
                            'keyword': None,
                            'metadata': sample,
                            'type': 'background'
                        })
        
        # If we don't have enough negative samples, create more by using keyword samples as negatives
        # for other keywords (e.g., "hello" can be a negative example for "activate")
        if len(negative_samples) < num_negative_samples and len(mixed_metadata) > 1:
            print("Adding other keyword samples as negative examples...")
            for keyword in mixed_metadata:
                if keyword not in keywords:  # Only use non-target keywords as negatives
                    keyword_samples = mixed_metadata[keyword]['samples']
                    for sample in keyword_samples:
                        sample_path = os.path.join(mixed_data_dir, keyword, sample['file'])
                        if os.path.exists(sample_path):
                            negative_samples.append({
                                'path': sample_path,
                                'label': 0,  # 0 is for negative class
                                'keyword': keyword,
                                'metadata': sample,
                                'type': 'other-keyword'
                            })
        
        # If we still don't have enough negative samples, use segments from positive samples
        # that don't contain the keyword (e.g., silence/background parts)
        # Note: This would require more advanced audio processing

        # Shuffle and limit negative samples to the desired ratio
        random.shuffle(negative_samples)
        negative_samples = negative_samples[:num_negative_samples]
        
        # Combine positive and negative samples
        all_samples = positive_samples + negative_samples
        
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
        
        # Count sample types in training set
        negative_types = {}
        negative_count = 0
        positive_count = 0
        
        for i, label in enumerate(dataset['train']['labels']):
            if label == 0:  # negative sample
                negative_count += 1
            else:
                positive_count += 1
                
        # Print dataset statistics
        print(f"Dataset prepared with {len(dataset['train']['features'])} training and "
              f"{len(dataset['validation']['features'])} validation samples")
        print(f"Training set: {positive_count} positive samples, {negative_count} negative samples")
        
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
    
    def train_model(self, keywords, epochs=50, batch_size=32, validation_split=0.2, learning_rate=0.001, negative_samples_ratio=1.0):
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
            report_path = os.path.join(self.model_dir, f"{model_name}_accuracy.txt")
            with open(report_path, 'w') as f:
                f.write(f"Accuracy: {accuracy:.4f}\n")

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
    parser.add_argument('--negative-samples-ratio', type=float, default=1.0, 
                        help='Ratio of negative samples to include (relative to positives)')
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
        learning_rate=args.learning_rate,
        negative_samples_ratio=args.negative_samples_ratio
    )

if __name__ == "__main__":
    main()
