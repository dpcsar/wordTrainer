"""
Mix keyword samples with background noise at various SNR levels.
"""

import os
import argparse
import numpy as np
import json
import librosa
import uuid
from tqdm import tqdm
import sys
import random

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.audio_utils import load_audio, save_audio, mix_audio, calculate_snr

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class AudioMixer:
    def __init__(self, keywords_dir, backgrounds_dir, output_dir, sample_rate=16000):
        """
        Initialize AudioMixer.
        
        Args:
            keywords_dir: Directory containing keyword samples
            backgrounds_dir: Directory containing background noise samples
            output_dir: Directory to save mixed samples
            sample_rate: Target sample rate for audio files
        """
        self.keywords_dir = keywords_dir
        self.backgrounds_dir = backgrounds_dir
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
        
        # Load keyword metadata
        keywords_metadata_path = os.path.join(keywords_dir, 'metadata.json')
        if os.path.exists(keywords_metadata_path):
            with open(keywords_metadata_path, 'r') as f:
                self.keywords_metadata = json.load(f)
        else:
            print(f"Warning: Keyword metadata not found at {keywords_metadata_path}")
            self.keywords_metadata = {}
        
        # Load background metadata
        backgrounds_metadata_path = os.path.join(backgrounds_dir, 'metadata.json')
        if os.path.exists(backgrounds_metadata_path):
            with open(backgrounds_metadata_path, 'r') as f:
                self.backgrounds_metadata = json.load(f)
        else:
            print(f"Warning: Background metadata not found at {backgrounds_metadata_path}")
            self.backgrounds_metadata = {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, cls=NumpyEncoder)
    
    def get_keyword_samples(self, keyword):
        """
        Get list of keyword samples.
        
        Args:
            keyword: Keyword to get samples for
            
        Returns:
            List of keyword sample paths
        """
        if keyword not in self.keywords_metadata:
            return []
        
        samples = []
        for sample in self.keywords_metadata[keyword]['samples']:
            file_path = os.path.join(self.keywords_dir, keyword, sample['file'])
            if os.path.exists(file_path):
                samples.append({
                    'path': file_path,
                    'metadata': sample
                })
        
        return samples
    
    def get_background_samples(self, noise_type=None):
        """
        Get list of background noise samples.
        
        Args:
            noise_type: Type of background noise ('propeller', 'jet', 'cockpit', or None for all)
            
        Returns:
            List of background sample paths
        """
        samples = []
        
        noise_types = [noise_type] if noise_type else ['propeller', 'jet', 'cockpit']
        
        for noise_type in noise_types:
            if noise_type not in self.backgrounds_metadata:
                continue
            
            for sample in self.backgrounds_metadata[noise_type]['samples']:
                file_path = os.path.join(self.backgrounds_dir, noise_type, sample['file'])
                if os.path.exists(file_path):
                    samples.append({
                        'path': file_path,
                        'metadata': sample
                    })
        
        return samples
    
    def mix_samples(self, keyword, noise_types=None, num_mixes=100, snr_range=(-5, 20)):
        """
        Mix keyword samples with background noise at various SNR levels.
        
        Args:
            keyword: Keyword to mix
            noise_types: Types of background noise to mix with ('propeller', 'jet', 'cockpit', or None for all)
            num_mixes: Number of mixed samples to generate
            snr_range: Range of SNR values in dB
        """
        # Get keyword samples
        keyword_samples = self.get_keyword_samples(keyword)
        if not keyword_samples:
            print(f"No samples found for keyword '{keyword}'")
            return
        
        # Get background samples
        background_samples = self.get_background_samples(
            noise_type=noise_types[0] if noise_types and len(noise_types) == 1 else None
        )
        if not background_samples:
            print("No background samples found")
            return
        
        print(f"Mixing {keyword} with background noise, creating {num_mixes} samples")
        
        # Create keyword directory in output
        keyword_dir = os.path.join(self.output_dir, keyword)
        os.makedirs(keyword_dir, exist_ok=True)
        
        # Initialize keyword metadata if not exists
        if keyword not in self.metadata:
            self.metadata[keyword] = {
                'samples': [],
                'count': 0
            }
        
        # Mix samples
        for i in tqdm(range(num_mixes)):
            # Select random keyword sample
            keyword_sample = random.choice(keyword_samples)
            
            # Select random background sample and type
            background_sample = random.choice(background_samples)
            
            # Select random SNR value
            snr = np.random.uniform(*snr_range)
            
            # Generate a unique ID for this mixed sample
            sample_id = str(uuid.uuid4())[:8]
            
            # Generate filename
            noise_type = background_sample['metadata']['type']
            snr_str = f"{snr:.1f}".replace('-', 'neg').replace('.', 'p')
            
            filename = f"{keyword}_{noise_type}_{snr_str}db_{sample_id}.wav"
            file_path = os.path.join(keyword_dir, filename)
            
            # Load audio files
            keyword_audio, _ = load_audio(keyword_sample['path'], target_sr=self.sample_rate)
            background_audio, _ = load_audio(background_sample['path'], target_sr=self.sample_rate)
            
            # Mix audio
            mixed_audio = mix_audio(keyword_audio, background_audio, snr)
            
            # Calculate actual SNR (for validation)
            actual_snr = calculate_snr(keyword_audio, mixed_audio - keyword_audio)
            
            # Save mixed audio
            save_audio(mixed_audio, file_path, sr=self.sample_rate)
            
            # Add metadata
            sample_metadata = {
                'id': sample_id,
                'file': filename,
                'keyword': keyword,
                'noise_type': noise_type,
                'target_snr_db': snr,
                'actual_snr_db': actual_snr,
                'keyword_sample': keyword_sample['metadata']['id'],
                'background_sample': background_sample['metadata']['id'],
            }
            
            self.metadata[keyword]['samples'].append(sample_metadata)
            self.metadata[keyword]['count'] += 1
            
            # Save metadata every 10 samples
            if i % 10 == 0:
                self._save_metadata()
        
        # Save final metadata
        self._save_metadata()
        print(f"Mixed {num_mixes} samples for keyword '{keyword}'")

def main():
    parser = argparse.ArgumentParser(description='Mix keyword samples with background noise')
    parser.add_argument('--keyword', type=str, required=True, help='Keyword to mix')
    parser.add_argument('--noise-types', type=str, nargs='+', choices=['propeller', 'jet', 'cockpit'], 
                        help='Types of background noise to mix with')
    parser.add_argument('--num-mixes', type=int, default=100, help='Number of mixed samples to generate')
    parser.add_argument('--min-snr', type=float, default=-5, help='Minimum SNR in dB')
    parser.add_argument('--max-snr', type=float, default=20, help='Maximum SNR in dB')
    parser.add_argument('--keywords-dir', type=str, default='../data/keywords', help='Keywords directory')
    parser.add_argument('--backgrounds-dir', type=str, default='../data/backgrounds', help='Backgrounds directory')
    parser.add_argument('--output-dir', type=str, default='../data/mixed', help='Output directory')
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.keywords_dir):
        args.keywords_dir = os.path.abspath(os.path.join(script_dir, args.keywords_dir))
    
    if not os.path.isabs(args.backgrounds_dir):
        args.backgrounds_dir = os.path.abspath(os.path.join(script_dir, args.backgrounds_dir))
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    mixer = AudioMixer(args.keywords_dir, args.backgrounds_dir, args.output_dir)
    mixer.mix_samples(
        args.keyword,
        args.noise_types,
        args.num_mixes,
        (args.min_snr, args.max_snr)
    )

if __name__ == "__main__":
    main()
