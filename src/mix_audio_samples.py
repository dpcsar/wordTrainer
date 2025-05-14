"""
Mix keyword samples with background noise at various SNR levels.
"""

import os
import argparse
import numpy as np
import json
import uuid
from tqdm import tqdm
import sys
import random

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import from config and audio_utils
from config import (SAMPLE_RATE, DEFAULT_NUM_MIXES_SAMPLES, DEFAULT_NUM_MIXES_NON_KEY_WORDS,
                   DEFAULT_SNR_RANGE, DEFAULT_KEYWORD, NOISE_TYPES, KEYWORDS_DIR, 
                   BACKGROUNDS_DIR, MIXED_DIR, NON_KEYWORDS_DIR)
from src.audio_utils import load_audio, save_audio, mix_audio, calculate_snr

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
    def __init__(self, keywords_dir, backgrounds_dir, output_dir, sample_rate=SAMPLE_RATE):
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
    
    def get_non_keyword_samples(self):
        """
        Get list of non-keyword samples.
        
        Returns:
            List of non-keyword sample paths
        """
        samples = []
        
        if NON_KEYWORDS_DIR not in self.keywords_metadata:
            return samples
        
        for sample in self.keywords_metadata[NON_KEYWORDS_DIR]["samples"]:
            file_path = os.path.join(self.keywords_dir, NON_KEYWORDS_DIR, sample['file'])
            if os.path.exists(file_path):
                samples.append({
                    'path': file_path,
                    'metadata': sample,
                    'non_keyword': sample.get('non_keyword')
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
        
        noise_types = [noise_type] if noise_type else NOISE_TYPES
        
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
    
    def mix_samples(self, keyword=None, noise_types=None, num_mixes=None, snr_range=DEFAULT_SNR_RANGE):
        """
        Mix keyword and non-keyword samples with background noise at various SNR levels.
        
        Args:
            keyword: Keyword to mix (required)
            noise_types: Types of background noise to mix with ('propeller', 'jet', 'cockpit', or None for all)
            num_mixes: Number of mixed samples to generate per keyword/non-keyword (if None, uses defaults)
            snr_range: Range of SNR values in dB
        """
        # Handle keyword samples
        keyword_samples = []
        if keyword:
            keyword_samples = self.get_keyword_samples(keyword)
            if not keyword_samples:
                print(f"No samples found for keyword '{keyword}'")
                return
        
        # Handle non-keyword samples (always required now)
        non_keyword_samples = []
        non_keyword_samples = self.get_non_keyword_samples()
        if not non_keyword_samples:
            print("No non-keyword samples found")
            return
                
        print(f"Found {len(keyword_samples)} keyword samples and {len(non_keyword_samples)} non-keyword samples")
        
        # Get background samples
        background_samples = self.get_background_samples(
            noise_type=noise_types[0] if noise_types and len(noise_types) == 1 else None
        )
        if not background_samples:
            print("No background samples found")
            return
            
        print(f"Found {len(background_samples)} background samples")
        
        # Use default values if num_mixes is None
        keyword_num_mixes = num_mixes if num_mixes is not None else DEFAULT_NUM_MIXES_SAMPLES
        non_keyword_num_mixes = num_mixes if num_mixes is not None else DEFAULT_NUM_MIXES_NON_KEY_WORDS
        
        # Calculate how many samples we need to create
        keyword_existing_count = 0
        non_keyword_existing_count = 0
        
        # Check existing sample counts
        if keyword in self.metadata:
            keyword_existing_count = self.metadata[keyword]['count']
        
        if NON_KEYWORDS_DIR in self.metadata:
            non_keyword_existing_count = self.metadata[NON_KEYWORDS_DIR]['count']
        
        # Calculate how many more we need
        keyword_needed = max(0, keyword_num_mixes - keyword_existing_count)
        non_keyword_needed = max(0, non_keyword_num_mixes - non_keyword_existing_count)
        
        # Process keyword samples only if more are needed
        if keyword_needed > 0:
            print(f"Creating {keyword_needed} more samples for {keyword} (existing: {keyword_existing_count}, target: {keyword_num_mixes})")
            self._mix_audio_category(keyword_samples, background_samples, keyword, keyword_needed, snr_range)
        else:
            print(f"Already have enough samples for {keyword}: {keyword_existing_count} >= {keyword_num_mixes}")
            
        # Process non-keyword samples only if more are needed
        if non_keyword_needed > 0:
            print(f"Creating {non_keyword_needed} more samples for non-keywords (existing: {non_keyword_existing_count}, target: {non_keyword_num_mixes})")
            self._mix_audio_category(non_keyword_samples, background_samples, NON_KEYWORDS_DIR, non_keyword_needed, snr_range, is_non_keyword=True)
        else:
            print(f"Already have enough samples for non-keywords: {non_keyword_existing_count} >= {non_keyword_num_mixes}")
            
        # Save final metadata
        self._save_metadata()
    
    def _mix_audio_category(self, samples, background_samples, category, num_mixes, snr_range, is_non_keyword=False):
        """
        Mix samples of a specific category with background noise.
        
        Args:
            samples: List of samples to mix
            background_samples: List of background noise samples
            category: Category name ('keyword' or non-keywords directory name)
            num_mixes: Number of mixed samples to generate
            snr_range: Range of SNR values in dB
            is_non_keyword: Whether we're processing non-keywords
        """
        print(f"Mixing {category} with background noise, creating {num_mixes} samples")
        
        # Create output directory if it doesn't exist (may be a relative path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Create category directory in output
        category_dir = os.path.join(self.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Print path for debugging
        print(f"Creating mixed samples in directory: {category_dir}")
        
        # Initialize category metadata if not exists
        if category not in self.metadata:
            self.metadata[category] = {
                'samples': [],
                'count': 0
            }
        
        # Mix samples
        for i in tqdm(range(num_mixes)):
            # Select random sample
            sample = random.choice(samples)
            
            # Select random background sample and type
            background_sample = random.choice(background_samples)
            
            # Select random SNR value
            snr = np.random.uniform(*snr_range)
            
            # Generate a unique ID for this mixed sample
            sample_id = str(uuid.uuid4())[:8]
            
            # Generate filename
            noise_type = background_sample['metadata']['type']
            snr_str = f"{snr:.1f}".replace('-', 'neg').replace('.', 'p')
            
            # Different filename format for keywords vs non-keywords
            if is_non_keyword:
                non_keyword_value = sample.get('non_keyword', sample['metadata'].get('non_keyword', 'unknown'))
                filename = f"non_keyword_{non_keyword_value}_{noise_type}_{snr_str}db_{sample_id}.wav"
            else:
                filename = f"{category}_{noise_type}_{snr_str}db_{sample_id}.wav"
                
            file_path = os.path.join(category_dir, filename)
            
            # Load audio files
            sample_audio, _ = load_audio(sample['path'], target_sr=self.sample_rate)
            background_audio, _ = load_audio(background_sample['path'], target_sr=self.sample_rate)
            
            # Mix audio
            mixed_audio = mix_audio(sample_audio, background_audio, snr)
            
            # Calculate actual SNR (for validation)
            actual_snr = calculate_snr(sample_audio, mixed_audio - sample_audio)
            
            # Save mixed audio
            save_audio(mixed_audio, file_path, sr=self.sample_rate)
            
            # Add metadata
            sample_metadata = {
                'id': sample_id,
                'file': filename,
                'noise_type': noise_type,
                'target_snr_db': snr,
                'actual_snr_db': actual_snr,
                'sample_id': sample['metadata']['id'],
                'background_sample': background_sample['metadata']['id'],
            }
            
            # Add keyword or non_keyword specific fields
            if is_non_keyword:
                sample_metadata['non_keyword'] = sample.get('non_keyword')
            else:
                sample_metadata['keyword'] = category
            
            self.metadata[category]['samples'].append(sample_metadata)
            self.metadata[category]['count'] += 1
            
            # Save metadata every 10 samples
            if i % 10 == 0:
                self._save_metadata()
        
        print(f"Mixed {num_mixes} samples for {category}")

def main():
    parser = argparse.ArgumentParser(description='Mix keyword and non-keyword samples with background noise')
    parser.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD, help=f'Keyword to mix (default: {DEFAULT_KEYWORD})')
    parser.add_argument('--noise-types', type=str, nargs='+', choices=NOISE_TYPES, 
                        help=f'Types of background noise to mix with (default: all)')
    parser.add_argument('--num-mixes', type=int, default=None, 
                        help=f'Number of mixed samples to generate (default: {DEFAULT_NUM_MIXES_SAMPLES} for keywords, {DEFAULT_NUM_MIXES_NON_KEY_WORDS} for non-keywords)')
    parser.add_argument('--min-snr', type=float, default=DEFAULT_SNR_RANGE[0], help=f'Minimum SNR in dB (default: {DEFAULT_SNR_RANGE[0]})')
    parser.add_argument('--max-snr', type=float, default=DEFAULT_SNR_RANGE[1], help=f'Maximum SNR in dB (default: {DEFAULT_SNR_RANGE[1]})')
    
    # Use directory paths from config for better consistency
    parser.add_argument('--keywords-dir', type=str, default=KEYWORDS_DIR, help='Keywords directory')
    parser.add_argument('--backgrounds-dir', type=str, default=BACKGROUNDS_DIR, help='Backgrounds directory')
    parser.add_argument('--output-dir', type=str, default=MIXED_DIR, help='Output directory')
    args = parser.parse_args()
    
    mixer = AudioMixer(args.keywords_dir, args.backgrounds_dir, args.output_dir)
    mixer.mix_samples(
        args.keyword,
        args.noise_types,
        args.num_mixes,
        (args.min_snr, args.max_snr)
    )

if __name__ == "__main__":
    main()
