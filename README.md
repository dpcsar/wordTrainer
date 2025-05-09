# Keyword Detection Model Trainer

A complete toolkit for training and testing keyword detection models for Android applications using TensorFlow Lite.

## Overview

This project provides tools to:

1. Generate keyword samples using Google Text-to-Speech (gTTS) with varying accents, ages, and genders
2. Synthesize aircraft cockpit background noise (propeller aircraft, jet aircraft, cockpit ambience)
3. Mix keywords with background noise at various Signal-to-Noise Ratios (SNR)
4. Train TensorFlow Lite models for keyword detection
5. Test models using both pre-recorded samples and microphone input
6. Export models for use in Android Kotlin applications

## Project Structure

```
wordTrainer/
├── src/               # Python source code
│   ├── generate_keywords.py         # Generate keyword samples using gTTS
│   ├── generate_background_noise.py # Generate cockpit background noise
│   ├── mix_audio_samples.py         # Mix keywords with noise at different SNRs
│   ├── train_model.py               # Train keyword detection model
│   ├── test_model_gtts.py           # Test model using gTTS samples
│   └── test_model_mic.py            # Test model using microphone input
│
├── utils/             # Utility functions
│   └── audio_utils.py               # Audio processing utilities
│
├── data/              # Audio data
│   ├── keywords/      # Keyword samples
│   ├── backgrounds/   # Background noise samples
│   └── mixed/         # Mixed audio for training
│
├── models/            # Trained models
│   └── plots/         # Training plots and visualizations
│
├── recordings/        # Saved test recordings
│
├── .vscode/           # VSCode configuration
│   └── launch.json    # Launch configurations
│
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+ with pip
- VS Code with Python extension
- Internet connection (for gTTS)

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

### 3. Mix Keyword Samples with Background Noise

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

### 4. Train Model

Train a keyword detection model:

```bash
python src/train_model.py --keywords "activate" "shutdown" --epochs 50 --batch-size 32
```

Options:
- `--keywords`: Keywords to detect (multiple keywords can be specified)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Initial learning rate (default: 0.001)

### 5. Test Model with gTTS Samples

Test the trained model using pre-recorded gTTS samples:

```bash
python src/test_model_gtts.py --model "models/keyword_detection_activate_20250509_120000.tflite" --dir "data/keywords/activate"
```

Options:
- `--model`: Path to trained model (.h5 or .tflite)
- `--file`: Path to a single audio file to test
- `--dir`: Directory containing audio files to test
- `--num-samples`: Maximum number of samples to test in batch mode (default: 10)

### 6. Test Model with Microphone

Test the trained model using real-time microphone input:

```bash
python src/test_model_mic.py --model "models/keyword_detection_activate_20250509_120000.tflite" --threshold 0.7
```

Options:
- `--model`: Path to trained model (.h5 or .tflite)
- `--threshold`: Detection threshold (default: 0.5)
- `--device`: Audio device index (optional)
- `--no-viz`: Disable visualization

## Integration with Android

The trained TFLite models can be integrated into Android applications. Key steps:

1. Copy the .tflite model file to your Android project's assets directory
2. Use the TensorFlow Lite Android library to load and run the model
3. Process audio from the device microphone in real-time
4. Apply the same feature extraction as used during training

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Text-to-Speech API for synthetic speech generation
- TensorFlow and TensorFlow Lite
- Python audio libraries: librosa, pydub, sounddevice
