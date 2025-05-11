# Keyword Detection Model Trainer

A complete toolkit for training and testing keyword detection models for Android applications using TensorFlow Lite.

## Overview

This project provides tools to:

1. Generate keyword samples using Google Text-to-Speech (gTTS) with varying accents, ages, and genders
2. Generate non-keyword samples as negative training examples
3. Synthesize aircraft cockpit background noise (propeller aircraft, jet aircraft, cockpit ambience)
4. Mix keywords with background noise at various Signal-to-Noise Ratios (SNR)
5. Train TensorFlow Lite models for keyword detection
6. Test models using both pre-recorded samples and microphone input
7. Export models for use in Android Kotlin applications

## Project Structure

```
wordTrainer/
├── src/               # Python source code
│   ├── audio_utils.py                # Audio processing utility functions
│   ├── generate_keywords.py          # Generate keyword samples using gTTS
│   ├── generate_non_keywords.py      # Generate non-keyword samples for negative training
│   ├── generate_background_noise.py  # Generate cockpit background noise
│   ├── mix_audio_samples.py          # Mix keywords with noise at different SNRs
│   ├── prepare_for_android.py        # Export and prepare models for Android
│   └── train_model.py                # Train keyword detection model
│
├── data/              # Audio data
│   ├── keywords/      # Keyword samples
│   │   ├── metadata.json             # Metadata for keyword samples
│   │   ├── activate/                 # Specific keyword samples
│   │   └── non_keywords/             # Non-keyword samples
│   │
│   ├── backgrounds/   # Background noise samples
│   │   ├── metadata.json             # Metadata for background noise
│   │   ├── cockpit/                  # Cockpit ambient noise
│   │   ├── jet/                      # Jet aircraft noise
│   │   └── propeller/                # Propeller aircraft noise
│   │
│   └── mixed/         # Mixed audio for training
│       └── metadata.json             # Metadata for mixed samples
│
├── tests/             # Test scripts
│   ├── test_audio_utils.py           # Test audio utility functions
│   ├── test_model_gtts.py            # Test model using gTTS samples
│   ├── test_model_mic.py             # Test model using microphone input
│   └── test_non_keywords.py          # Test non-keyword generation
│
├── models/            # Trained models
│   ├── model_metadata.json           # Metadata for trained models
│   └── plots/                        # Training plots and visualizations
│
├── config.py          # Configuration settings
├── main.py            # Main entry point
├── run_tests.sh       # Script to run all tests
├── run_workflow.sh    # Script to run complete training workflow
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+ with pip
- VS Code with Python extension
- Internet connection (for gTTS)
- Audio output and input devices for testing

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd wordTrainer
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the project in VS Code:
   ```bash
   code .
   ```

## Usage

### 1. Generate Keyword Samples

Generate speech samples for your keyword using different accents, simulated ages, and genders:

```bash
python src/generate_keywords.py --keyword "activate" --samples 50
```

Options:
- `--keyword`: The keyword to generate samples for
- `--samples`: Number of samples to generate (default: 50)
- `--output-dir`: Output directory (default: data/keywords)
- `--silence`: Silence to add at beginning and end in ms (default: 500)

### 2. Generate Background Noise

Generate synthetic background noise for aircraft environments:

```bash
python src/generate_background_noise.py --type all --samples 20
```

Options:
- `--type`: Type of noise to generate ('propeller', 'jet', 'cockpit', or 'all')
- `--samples`: Number of samples to generate (default: 20)
- `--output-dir`: Output directory (default: data/backgrounds)
- `--min-duration`: Minimum duration in seconds (default: 3.0)
- `--max-duration`: Maximum duration in seconds (default: 10.0)

### 3. Generate Non-Keyword Samples

Generate non-keyword samples to improve model discrimination by providing realistic negative examples:

```bash
python src/generate_non_keywords.py --samples 50 --avoid-keyword "activate"
```

Options:
- `--samples`: Number of samples to generate (default: 50)
- `--output-dir`: Output directory (default: data/keywords)
- `--silence`: Silence to add at beginning and end in milliseconds (default: 500)
- `--avoid-keyword`: Keyword to avoid using as non-keyword samples (optional)

Using non-keywords improves model discrimination by providing actual words as negative examples, not just background noise. This helps the model learn to reject similar-sounding words, reducing false positives in real-world usage.

### 4. Mix Keyword Samples with Background Noise

Mix keyword samples with background noise at various SNR levels:

```bash
python src/mix_audio_samples.py --keyword "activate" --num-mixes 100 --min-snr -5 --max-snr 20
```

Options:
- `--keyword`: Keyword to mix
- `--noise-types`: Types of background noise to mix with ('propeller', 'jet', 'cockpit')
- `--num-mixes`: Number of mixed samples to generate (default: 100)
- `--min-snr`: Minimum SNR in dB (default: -5)
- `--max-snr`: Maximum SNR in dB (default: 20)

### 5. Train Model

Train a keyword detection model:

```bash
python src/train_model.py --keywords "activate" "shutdown" --epochs 50 --batch-size 32
```

The training process will use:
- Positive examples: The specified keywords
- Negative examples: 
  - Non-keywords (common words generated as negative examples)
  - Background noise samples
  - Other keywords not specified for detection

Options:
- `--keywords`: Keywords to detect (multiple keywords can be specified)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--negative-samples-ratio`: Ratio of negative samples to include (default: 1.0)

### 6. Test Model with gTTS Samples

Test the trained model using pre-recorded gTTS samples:

```bash
python tests/test_model_gtts.py --model "models/keyword_detection_<your_model_xxx>.tflite" --dir "data/keywords/activate"
```

Options:
- `--model`: Path to trained model (.h5 or .tflite)
- `--file`: Path to a single audio file to test
- `--dir`: Directory containing audio files to test
- `--samples`: Maximum number of samples to test in batch mode (default: 10)

### 7. Test Model with Microphone

Test the trained model using real-time microphone input:

```bash
python tests/test_model_mic.py --model "models/keyword_detection_<your_model_xxx>.tflite" --threshold 0.7
```

Options:
- `--model`: Path to trained model (.h5 or .tflite)
- `--threshold`: Detection threshold (default: 0.5)
- `--device`: Audio device index (optional)
- `--no-viz`: Disable visualization

## Integration with Android

The trained TFLite models can be integrated into Android applications. Use the prepare_for_android.py script to export your model:

```bash
python src/prepare_for_android.py --model "models/keyword_detection_<your_model_xxx>.h5" --output "models/android_ready" --optimize
```

Options:
- `--model`: Path to the trained model (.h5)
- `--output`: Output directory for Android-ready files
- `--optimize`: Apply additional optimization for mobile deployment (optional)

Key integration steps:
1. Copy the exported .tflite model file to your Android project's assets directory
2. Use the TensorFlow Lite Android library to load and run the model
3. Process audio from the device microphone in real-time
4. Apply the same feature extraction as used during training

## Complete Workflow

You can run the entire workflow using the provided shell script:

```bash
./run_workflow.sh --keyword "activate" --samples 50 --train-epochs 50
```

This script will:
1. Generate keyword samples
2. Generate background noise
3. Generate non-keywords
4. Mix samples with noise
5. Train the model
6. Test the model
7. Prepare for Android deployment

## Testing

Run the test suite to verify functionality:

```bash
./run_tests.sh
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Text-to-Speech API for synthetic speech generation
- TensorFlow and TensorFlow Lite
- Python audio libraries: librosa, pydub, sounddevice
