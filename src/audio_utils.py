"""
Audio utility functions for the keyword detection model.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import librosa.display
from pydub import AudioSegment
from scipy import signal
import matplotlib.pyplot as plt
import random
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SAMPLE_RATE

def load_audio(file_path, target_sr=SAMPLE_RATE):
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        audio: Audio samples
        sr: Sample rate
    """
    try:
        # Use librosa to load audio file (handles various formats)
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
        
    except Exception as e:
        # Fallback to pydub if librosa fails
        try:
            print(f"Librosa failed to load {file_path}, trying with pydub. Error: {str(e)}")
            audio_segment = AudioSegment.from_file(file_path)
            audio_segment = audio_segment.set_frame_rate(target_sr)
            audio_segment = audio_segment.set_channels(1)
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
            return audio, target_sr
            
        except Exception as e2:
            print(f"Failed to load {file_path} with error: {str(e2)}")
            raise e2

def extract_features(audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=160):
    """
    Extract MFCC features from audio.
    
    Args:
        audio: Audio samples
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients to extract
        n_fft: FFT window size
        hop_length: Hop length between frames
        
    Returns:
        mfccs: MFCC features
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Add MFCC deltas and delta-deltas for additional features
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Combine features
    features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    
    return features

def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio samples
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        resampled: Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    return resampled

def adjust_pitch_by_age(audio, age_group, reference_frame_rate=44100):
    """
    Adjust audio pitch based on age group.
    
    Args:
        audio: AudioSegment object to modify
        age_group: Age group ('child', 'young_adult', 'adult', 'senior')
        reference_frame_rate: Reference frame rate to reset to after modification
    
    Returns:
        AudioSegment: Audio with adjusted pitch
    """
    octaves = 0  # Default: no change
    
    # Adjust pitch based on age group
    if age_group == 'child':
        octaves = 0.2  # higher pitch for children
    elif age_group == 'young_adult':
        octaves = 0.0  # no change for young adults
    elif age_group == 'adult':
        octaves = -0.1  # slightly lower for adults
    elif age_group == 'senior':
        octaves = -0.3  # lower pitch for seniors
    
    # Only modify if there's a pitch change
    if octaves != 0:
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * (2.0 ** octaves))
        })
        audio = audio.set_frame_rate(reference_frame_rate)
    
    return audio

def normalize_audio(audio):
    """
    Normalize audio to have max amplitude of 1.
    
    Args:
        audio: Audio samples
        
    Returns:
        normalized: Normalized audio
    """
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio

# Keep this alias for backward compatibility
# It calls the new function, ignoring the gender parameter
def adjust_pitch_by_age_gender(audio, age_group, gender=None, reference_frame_rate=44100):
    """
    Legacy function that calls adjust_pitch_by_age, ignoring the gender parameter.
    Kept for backward compatibility.
    """
    return adjust_pitch_by_age(audio, age_group, reference_frame_rate)

def add_noise(audio, noise_level=0.005):
    """
    Add random noise to audio.
    
    Args:
        audio: Audio samples
        noise_level: Noise level (0.0 to 1.0)
        
    Returns:
        noisy_audio: Audio with added noise
    """
    noise = np.random.randn(len(audio)) * noise_level
    noisy_audio = audio + noise
    
    # Re-normalize
    return normalize_audio(noisy_audio)

def shift_audio(audio, shift_ms=100, sample_rate=SAMPLE_RATE):
    """
    Shift audio in time.
    
    Args:
        audio: Audio samples
        shift_ms: Shift in milliseconds
        sample_rate: Sample rate
        
    Returns:
        shifted_audio: Shifted audio
    """
    shift_samples = int(shift_ms * sample_rate / 1000)
    shifted_audio = np.roll(audio, shift_samples)
    
    # If we roll right, we need to zero out the samples that roll around
    if shift_samples > 0:
        shifted_audio[:shift_samples] = 0
    else:
        shifted_audio[shift_samples:] = 0
    
    return shifted_audio

def plot_waveform(audio, sr=SAMPLE_RATE, title="Waveform"):
    """
    Plot waveform of audio.
    
    Args:
        audio: Audio samples
        sr: Sample rate
        title: Plot title
    """
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(audio)) / sr, audio)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    plt.close()  # Explicitly close the figure

def plot_spectrogram(audio, sr=SAMPLE_RATE, title="Spectrogram"):
    """
    Plot spectrogram of audio.
    
    Args:
        audio: Audio samples
        sr: Sample rate
        title: Plot title
    """
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()  # Explicitly close the figure

def augment_audio(audio, sr=SAMPLE_RATE):
    """
    Apply random augmentation to audio.
    
    Args:
        audio: Audio samples
        sr: Sample rate
        
    Returns:
        augmented: Augmented audio
    """
    # Make a copy to avoid modifying original
    augmented = audio.copy()
    
    # Apply a random time shift
    if random.random() > 0.5:
        shift_ms = random.randint(-100, 100)  # Shift by up to 100ms in either direction
        augmented = shift_audio(augmented, shift_ms, sr)
    
    # Add random noise
    if random.random() > 0.5:
        noise_level = random.uniform(0.001, 0.01)  # Random noise level
        augmented = add_noise(augmented, noise_level)
    
    # Apply random pitch shift
    if random.random() > 0.5:
        pitch_steps = random.uniform(-1, 1)  # Shift by up to one semitone
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=pitch_steps)
    
    # Apply random time stretch (but not too extreme)
    if random.random() > 0.5:
        rate = random.uniform(0.9, 1.1)  # Stretch rate
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
        
        # Make sure the length matches original
        if len(augmented) > len(audio):
            augmented = augmented[:len(audio)]
        elif len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)))
    
    # Re-normalize
    return normalize_audio(augmented)

def save_audio(audio, file_path, sr=SAMPLE_RATE):
    """
    Save audio samples to a file.
    
    Args:
        audio: Audio samples
        file_path: Path to save audio file
        sr: Sample rate
        
    Returns:
        file_path: Path to saved file
    """
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save audio file using soundfile
        sf.write(file_path, audio, sr, subtype='PCM_16')
        
        return file_path
    except Exception as e:
        print(f"Error saving audio file {file_path}: {str(e)}")
        raise e

def mix_audio(foreground, background, snr_db):
    """
    Mix foreground audio with background audio at a given SNR level.
    
    Args:
        foreground: Foreground audio samples
        background: Background audio samples
        snr_db: Target Signal-to-Noise ratio in dB
        
    Returns:
        mixed_audio: Mixed audio samples
    """
    # Ensure background is at least as long as foreground
    if len(background) < len(foreground):
        # Loop background if necessary
        factor = int(np.ceil(len(foreground) / len(background)))
        background = np.tile(background, factor)
        
    # Trim background to match foreground length
    background = background[:len(foreground)]
    
    # Calculate foreground and background powers
    foreground_power = np.mean(foreground**2)
    background_power = np.mean(background**2)
    
    # Calculate background gain to achieve target SNR
    if background_power > 0 and foreground_power > 0:
        gain = np.sqrt(foreground_power / (background_power * 10**(snr_db/10)))
    else:
        gain = 0
        
    # Apply gain to background
    scaled_background = background * gain
    
    # Mix foreground and background
    mixed = foreground + scaled_background
    
    # Normalize to prevent clipping
    if np.max(np.abs(mixed)) > 0:
        mixed = mixed / np.max(np.abs(mixed)) * 0.95
        
    return mixed

def calculate_snr(signal, noise):
    """
    Calculate Signal-to-Noise Ratio in dB.
    
    Args:
        signal: Signal audio samples
        noise: Noise audio samples
        
    Returns:
        snr_db: Signal-to-Noise Ratio in dB
    """
    # Ensure noise is at least as long as signal
    if len(noise) < len(signal):
        factor = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, factor)
        
    # Trim noise to match signal length
    noise = noise[:len(signal)]
    
    # Calculate signal and noise powers
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    # Calculate SNR
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')  # No noise
        
    return snr