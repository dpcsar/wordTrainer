"""
Generate and process background noise samples for aircraft environments.
"""

import os
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import json
import uuid
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import from audio_utils
from src.audio_utils import save_audio

class BackgroundNoiseGenerator:
    def __init__(self, output_dir, sample_rate=16000):
        """
        Initialize BackgroundNoiseGenerator.
        
        Args:
            output_dir: Directory to save generated noise samples
            sample_rate: Target sample rate for audio files
        """
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.metadata = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Path to metadata file
        self.metadata_path = os.path.join(output_dir, 'metadata.json')
        
        # Load existing metadata if available
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def generate_propeller_noise(self, duration_sec=10, rpm_range=(1500, 2700)):
        """
        Generate synthetic propeller aircraft noise.
        
        Args:
            duration_sec: Duration of noise in seconds
            rpm_range: Range of RPM values
            
        Returns:
            noise: Synthesized propeller noise
        """
        # Number of samples
        n_samples = int(duration_sec * self.sample_rate)
        
        # Time array
        t = np.linspace(0, duration_sec, n_samples)
        
        # Generate base white noise
        white_noise = np.random.normal(0, 0.1, n_samples)
        
        # Add propeller blade passing frequency components
        rpm = np.random.uniform(*rpm_range)  # Propeller RPM
        blades = np.random.randint(2, 5)  # Number of propeller blades
        
        # Blade passing frequency in Hz
        bpf = rpm * blades / 60
        
        # Generate propeller harmonics
        prop_noise = np.zeros_like(white_noise)
        for harmonic in range(1, 6):
            # Amplitude decreases with harmonic number
            amplitude = 0.8 / harmonic
            # Add harmonic component
            prop_noise += amplitude * np.sin(2 * np.pi * harmonic * bpf * t)
        
        # Mix with white noise
        mixed_noise = 0.7 * prop_noise + 0.3 * white_noise
        
        # Apply low-pass filter to simulate real-world noise
        b, a = signal.butter(4, 3000 / (self.sample_rate / 2), btype='low')
        filtered_noise = signal.filtfilt(b, a, mixed_noise)
        
        # Add engine rumble (low frequency components)
        rumble_freq = np.random.uniform(80, 120)  # Engine frequency in Hz
        rumble = 0.4 * np.sin(2 * np.pi * rumble_freq * t)
        
        # Mix in the rumble
        final_noise = 0.8 * filtered_noise + 0.2 * rumble
        
        # Normalize
        final_noise = final_noise / np.max(np.abs(final_noise)) * 0.95
        
        return final_noise
    
    def generate_jet_noise(self, duration_sec=10, intensity_range=(0.6, 1.0)):
        """
        Generate synthetic jet aircraft noise.
        
        Args:
            duration_sec: Duration of noise in seconds
            intensity_range: Range of intensity values
            
        Returns:
            noise: Synthesized jet noise
        """
        # Number of samples
        n_samples = int(duration_sec * self.sample_rate)
        
        # Generate pink noise (characteristic of jet engines)
        white_noise = np.random.normal(0, 1, n_samples)
        
        # Convert to pink noise by applying 1/f filter
        b, a = signal.butter(1, 4000 / (self.sample_rate / 2), btype='low')
        pink_noise = signal.filtfilt(b, a, white_noise)
        
        # Add high-frequency components for turbine whine
        t = np.linspace(0, duration_sec, n_samples)
        
        # Turbine whine frequency (varies randomly)
        whine_freq = np.random.uniform(5000, 8000)
        
        # Create slight frequency modulation for whine
        mod_freq = np.random.uniform(0.5, 2.0)
        freq_mod = whine_freq + 200 * np.sin(2 * np.pi * mod_freq * t)
        
        # Instantaneous phase is the integral of frequency
        phase = np.cumsum(freq_mod) / self.sample_rate
        
        # Generate whine with amplitude modulation
        whine_amplitude = np.random.uniform(*intensity_range)
        whine = whine_amplitude * 0.15 * np.sin(2 * np.pi * phase)
        
        # Add low frequency rumble
        rumble_freq = np.random.uniform(50, 100)
        rumble_amplitude = np.random.uniform(*intensity_range)
        rumble = rumble_amplitude * 0.4 * np.sin(2 * np.pi * rumble_freq * t)
        
        # Mix components
        mixed_noise = 0.7 * pink_noise + 0.2 * whine + 0.1 * rumble
        
        # Apply band-pass filter to shape spectrum
        nyquist = self.sample_rate / 2
        cutoff_low = 100 / nyquist
        cutoff_high = 8000 / nyquist
        
        # Ensure values are within valid range (0 < Wn < 1)
        cutoff_low = max(0.001, min(cutoff_low, 0.99))
        cutoff_high = max(cutoff_low + 0.001, min(cutoff_high, 0.99))
        
        b, a = signal.butter(2, [cutoff_low, cutoff_high], btype='band')
        filtered_noise = signal.filtfilt(b, a, mixed_noise)
        
        # Normalize
        filtered_noise = filtered_noise / np.max(np.abs(filtered_noise)) * 0.95
        
        return filtered_noise
    
    def generate_cockpit_ambience(self, duration_sec=10):
        """
        Generate synthetic cockpit ambience (mix of systems sounds).
        
        Args:
            duration_sec: Duration of noise in seconds
            
        Returns:
            noise: Synthesized cockpit ambience
        """
        # Number of samples
        n_samples = int(duration_sec * self.sample_rate)
        
        # Generate base noise (mix of white and pink)
        white_noise = np.random.normal(0, 0.1, n_samples)
        
        # Convert part to pink noise (lower frequencies)
        b, a = signal.butter(1, 1000 / (self.sample_rate / 2), btype='low')
        pink_noise = signal.filtfilt(b, a, white_noise)
        
        # Time array
        t = np.linspace(0, duration_sec, n_samples)
        
        # Add random electronic beeps and system sounds
        beeps = np.zeros_like(white_noise)
        
        # Add 2-5 random beeps
        for _ in range(np.random.randint(2, 6)):
            # Random start time and duration
            start_time = np.random.uniform(0, duration_sec * 0.8)
            beep_duration = np.random.uniform(0.1, 0.5)
            beep_freq = np.random.choice([800, 1000, 1200, 2000])
            
            # Convert times to sample indices
            start_idx = int(start_time * self.sample_rate)
            duration_idx = int(beep_duration * self.sample_rate)
            end_idx = min(start_idx + duration_idx, n_samples)
            
            # Generate beep tone
            t_beep = np.arange(duration_idx) / self.sample_rate
            beep_tone = 0.2 * np.sin(2 * np.pi * beep_freq * t_beep)
            
            # Apply envelope
            envelope = np.ones_like(beep_tone)
            attack = int(0.05 * duration_idx)
            decay = int(0.1 * duration_idx)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-decay:] = np.linspace(1, 0, decay)
            beep_tone = beep_tone * envelope
            
            # Add to beeps array
            beeps[start_idx:end_idx] += beep_tone[:end_idx-start_idx]
        
        # Add constant low-level hum (air systems)
        hum_freq = np.random.uniform(100, 200)
        hum = 0.05 * np.sin(2 * np.pi * hum_freq * t)
        
        # Mix all components
        mixed_noise = 0.6 * pink_noise + 0.3 * beeps + 0.1 * hum
        
        # Normalize
        mixed_noise = mixed_noise / np.max(np.abs(mixed_noise)) * 0.95
        
        return mixed_noise
    
    def generate_noise_samples(self, noise_type, num_samples=20, duration_range=(3, 10)):
        """
        Generate multiple noise samples and save them.
        
        Args:
            noise_type: Type of noise to generate ('propeller', 'jet', or 'cockpit')
            num_samples: Number of samples to generate
            duration_range: Range of durations in seconds
        """
        print(f"Generating {num_samples} {noise_type} noise samples")
        
        # Create noise type directory if it doesn't exist
        noise_dir = os.path.join(self.output_dir, noise_type)
        os.makedirs(noise_dir, exist_ok=True)
        
        # Initialize noise_type metadata if not exists
        if noise_type not in self.metadata:
            self.metadata[noise_type] = {
                'samples': [],
                'count': 0
            }
        
        # Generate samples
        for i in tqdm(range(num_samples)):
            # Generate a unique ID for this sample
            sample_id = str(uuid.uuid4())[:8]
            
            # Random duration
            duration = np.random.uniform(*duration_range)
            
            # Generate filename
            filename = f"{noise_type}_{sample_id}.wav"
            file_path = os.path.join(noise_dir, filename)
            
            # Generate noise based on type
            if noise_type == 'propeller':
                noise = self.generate_propeller_noise(duration)
            elif noise_type == 'jet':
                noise = self.generate_jet_noise(duration)
            elif noise_type == 'cockpit':
                noise = self.generate_cockpit_ambience(duration)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            # Save noise sample
            save_audio(noise, file_path, sr=self.sample_rate)
            
            # Add metadata
            sample_metadata = {
                'id': sample_id,
                'file': filename,
                'type': noise_type,
                'duration_sec': duration,
            }
            
            self.metadata[noise_type]['samples'].append(sample_metadata)
            self.metadata[noise_type]['count'] += 1
        
        # Save metadata
        self._save_metadata()
        print(f"Generated {num_samples} {noise_type} noise samples")

def main():
    parser = argparse.ArgumentParser(description='Generate background noise samples for aircraft environments')
    parser.add_argument('--type', type=str, choices=['propeller', 'jet', 'cockpit', 'all'], 
                        default='all', help='Type of noise to generate')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='../data/backgrounds', help='Output directory')
    parser.add_argument('--min-duration', type=float, default=3.0, help='Minimum duration in seconds')
    parser.add_argument('--max-duration', type=float, default=10.0, help='Maximum duration in seconds')
    args = parser.parse_args()
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    generator = BackgroundNoiseGenerator(args.output_dir)
    
    if args.type == 'all':
        for noise_type in ['propeller', 'jet', 'cockpit']:
            generator.generate_noise_samples(
                noise_type, 
                args.samples, 
                (args.min_duration, args.max_duration)
            )
    else:
        generator.generate_noise_samples(
            args.type, 
            args.samples, 
            (args.min_duration, args.max_duration)
        )

if __name__ == "__main__":
    main()
