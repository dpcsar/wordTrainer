"""
Unit tests for audio utility functions.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import (
    calculate_snr,
    adjust_noise_for_snr,
    mix_audio,
    extract_features
)

class TestAudioUtils(unittest.TestCase):
    def test_calculate_snr(self):
        # Create simple signal and noise
        signal = np.ones(1000)
        noise = np.ones(1000) * 0.1
        
        # Calculate SNR
        snr_db = calculate_snr(signal, noise)
        
        # Expected SNR: 10 * log10(1^2 / 0.1^2) = 20 dB
        self.assertAlmostEqual(snr_db, 20.0, delta=0.1)
    
    def test_adjust_noise_for_snr(self):
        # Create simple signal and noise
        signal = np.ones(1000)
        noise = np.ones(1000) * 0.1
        
        # Target SNR: 10 dB
        target_snr_db = 10.0
        
        # Adjust noise
        adjusted_noise = adjust_noise_for_snr(signal, noise, target_snr_db)
        
        # Calculate actual SNR
        actual_snr_db = calculate_snr(signal, adjusted_noise)
        
        # Check if actual SNR is close to target
        self.assertAlmostEqual(actual_snr_db, target_snr_db, delta=0.1)
    
    def test_mix_audio(self):
        # Create simple signal and noise
        signal = np.ones(1000)
        noise = np.ones(1000) * 0.1
        
        # Target SNR: 15 dB
        target_snr_db = 15.0
        
        # Mix audio
        mixed = mix_audio(signal, noise, target_snr_db)
        
        # Check if mixed audio has the right shape
        self.assertEqual(mixed.shape, signal.shape)
        
        # Calculate SNR between signal and (mixed - signal)
        actual_noise = mixed - signal
        actual_snr_db = calculate_snr(signal, actual_noise)
        
        # Check if actual SNR is close to target
        self.assertAlmostEqual(actual_snr_db, target_snr_db, delta=1.0)
    
    def test_extract_features(self):
        # Create simple audio signal (sine wave)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Extract features
        n_mfcc = 13
        mfccs = extract_features(audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Check if features have the right shape
        self.assertEqual(mfccs.shape[0], n_mfcc)
        
        # Check if features are not all zeros or NaNs
        self.assertFalse(np.all(mfccs == 0))
        self.assertFalse(np.any(np.isnan(mfccs)))

if __name__ == '__main__':
    unittest.main()
