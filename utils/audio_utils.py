"""
Audio utility functions for the keyword detection model.
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from scipy import signal
import matplotlib.pyplot as plt

def load_audio(file_path, target_sr=16000):
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        audio_data: Audio samples as numpy array
        sr: Sample rate
    """
    audio_data, sr = librosa.load(file_path, sr=target_sr)
    return audio_data, sr

def save_audio(audio_data, file_path, sr=16000):
    """
    Save audio data to file.
    
    Args:
        audio_data: Audio samples as numpy array
        file_path: Output file path
        sr: Sample rate
    """
    sf.write(file_path, audio_data, sr)

def calculate_snr(signal, noise):
    """
    Calculate SNR in dB between signal and noise.
    
    Args:
        signal: Signal array
        noise: Noise array
        
    Returns:
        snr_db: SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = signal_power / noise_power if noise_power > 0 else float('inf')
    snr_db = 10 * np.log10(snr) if snr > 0 else float('-inf')
    return snr_db

def adjust_noise_for_snr(signal, noise, target_snr_db):
    """
    Adjust noise level to achieve target SNR.
    
    Args:
        signal: Signal array
        noise: Noise array
        target_snr_db: Target SNR in dB
        
    Returns:
        adjusted_noise: Adjusted noise array
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate the noise scaling factor to achieve target SNR
    current_snr = signal_power / noise_power if noise_power > 0 else float('inf')
    current_snr_db = 10 * np.log10(current_snr) if current_snr > 0 else float('-inf')
    
    adjustment_db = current_snr_db - target_snr_db
    adjustment_factor = 10 ** (adjustment_db / 10)
    
    adjusted_noise = noise * np.sqrt(adjustment_factor)
    return adjusted_noise

def mix_audio(signal, noise, target_snr_db):
    """
    Mix signal with noise at specified SNR.
    
    Args:
        signal: Signal array
        noise: Noise array
        target_snr_db: Target SNR in dB
        
    Returns:
        mixed: Signal + noise mixture
    """
    # Make sure noise is at least as long as signal
    if len(noise) < len(signal):
        # Tile the noise to make it long enough
        noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
        noise = noise[:len(signal)]
    else:
        # Take a random segment from noise that matches signal length
        start = np.random.randint(0, len(noise) - len(signal) + 1)
        noise = noise[start:start + len(signal)]
    
    # Adjust noise to achieve target SNR
    adjusted_noise = adjust_noise_for_snr(signal, noise, target_snr_db)
    
    # Mix signal and noise
    mixed = signal + adjusted_noise
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val * 0.95  # 0.95 to leave some headroom
        
    return mixed

def extract_features(audio, sr=16000, n_mfcc=13, n_fft=512, hop_length=160):
    """
    Extract MFCC features from audio.
    
    Args:
        audio: Audio samples
        sr: Sample rate
        n_mfcc: Number of MFCCs to extract
        n_fft: FFT window size
        hop_length: Hop length for FFT
        
    Returns:
        mfccs: MFCC features
    """
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=n_mfcc, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    # Normalize the MFCCs
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
    return mfccs

def plot_waveform(audio, sr=16000, title="Waveform"):
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

def plot_spectrogram(audio, sr=16000, title="Spectrogram"):
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

def augment_audio(audio, sr=16000):
    """
    Apply random augmentation to audio.
    
    Args:
        audio: Audio samples
        sr: Sample rate
        
    Returns:
        augmented_audio: Augmented audio
    """
    # Apply random time shift
    shift = np.random.randint(-sr//10, sr//10)
    if shift > 0:
        augmented_audio = np.pad(audio, (0, shift), mode='constant')[shift:]
    else:
        augmented_audio = np.pad(audio, (-shift, 0), mode='constant')[:len(audio)]
    
    # Apply random pitch shift
    n_steps = np.random.uniform(-1, 1)
    augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sr, n_steps=n_steps)
    
    # Apply random speed change
    speed_factor = np.random.uniform(0.9, 1.1)
    augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=speed_factor)
    
    # Trim or pad to original length
    if len(augmented_audio) > len(audio):
        augmented_audio = augmented_audio[:len(audio)]
    else:
        augmented_audio = np.pad(augmented_audio, (0, len(audio) - len(augmented_audio)), mode='constant')
    
    return augmented_audio
