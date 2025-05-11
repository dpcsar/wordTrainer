#!/usr/bin/env python3
"""
Generate non-keyword speech samples using gTTS with different accents and variations.
These serve as negative examples for training that are actual words (unlike background noise).
"""

import os
import sys
import argparse
import random
import numpy as np
import time
from gtts import gTTS
from pydub import AudioSegment
import uuid
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACCENTS, AGE_GROUPS, NON_KEYWORDS, SAMPLE_RATE, DEFAULT_NON_KEYWORD_SAMPLES, DEFAULT_SILENCE_MS, KEYWORDS_DIR, DEFAULT_KEYWORD

class NonKeywordGenerator:
    def __init__(self, output_dir, sample_rate=SAMPLE_RATE):
        """
        Initialize NonKeywordGenerator.
        
        Args:
            output_dir: Directory to save generated samples
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
    
    def generate_non_keyword_samples(self, num_samples=DEFAULT_NON_KEYWORD_SAMPLES, silence_ms=DEFAULT_SILENCE_MS, keyword_to_avoid=DEFAULT_KEYWORD):
        """
        Generate non-keyword speech samples with different words, accents, ages, and genders.
        
        Args:
            num_samples: Number of samples to generate (default: DEFAULT_NON_KEYWORD_SAMPLES)
            silence_ms: Silence to add at beginning and end (milliseconds) (default: DEFAULT_SILENCE_MS)
            keyword_to_avoid: Keyword to avoid using as non-keyword samples (default: DEFAULT_KEYWORD)
        """
        print(f"Generating {num_samples} non-keyword samples")
        
        # Create non-keywords directory if it doesn't exist
        non_keywords_dir = os.path.join(self.output_dir, "non_keywords")
        os.makedirs(non_keywords_dir, exist_ok=True)
        
        # Initialize non-keywords metadata if not exists
        if "non_keywords" not in self.metadata:
            self.metadata["non_keywords"] = {
                'samples': [],
                'count': 0
            }
        
        # Filter out the keyword to avoid if provided
        available_words = [word for word in NON_KEYWORDS if word != keyword_to_avoid]
        if len(available_words) == 0:
            available_words = NON_KEYWORDS
        
        # Generate samples
        for i in tqdm(range(num_samples)):
            # Select random non-keyword word
            non_keyword = random.choice(available_words)
            
            # Randomly select accent, age, gender
            accent_info = random.choice(ACCENTS)
            age_group = random.choice(AGE_GROUPS)
            gender = accent_info["gender"]  # Use the gender from the accent info
            
            # Generate a unique ID for this sample
            sample_id = str(uuid.uuid4())[:8]
            
            # Extract TLD for filename (e.g., "com" becomes "us", "co.uk" becomes "uk")
            tld = accent_info["tld"]
            # Convert TLD to a short country code
            if tld == "com":
                country_code = "us"
            elif tld.startswith("co."):
                country_code = tld.split(".")[-1]
            elif tld.startswith("com."):
                country_code = tld.split(".")[-1]
            else:
                country_code = tld
                
            # Generate filename
            filename = f"nonkw_{non_keyword}_{country_code}_{age_group}_{gender}_{sample_id}.wav"
            file_path = os.path.join(non_keywords_dir, filename)
            
            # Generate TTS audio
            try:
                tts = gTTS(text=non_keyword, lang=accent_info["lang"], tld=accent_info["tld"])
                mp3_path = file_path.replace('.wav', '.mp3')
                tts.save(mp3_path)
                
                # Convert mp3 to wav and apply transformations
                audio = AudioSegment.from_mp3(mp3_path)
                
                # Add silence at beginning and end
                silence = AudioSegment.silent(duration=silence_ms)
                audio = silence + audio + silence
                
                # Adjust pitch and speed for age and gender simulation
                if age_group == 'child':
                    octaves = 0.3  # higher pitch for children
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * (2.0 ** octaves))
                    })
                    audio = audio.set_frame_rate(44100)
                elif age_group == 'senior':
                    octaves = -0.2  # lower pitch for seniors
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * (2.0 ** octaves))
                    })
                    audio = audio.set_frame_rate(44100)
                
                if gender == 'male' and (age_group == 'adult' or age_group == 'young_adult'):
                    octaves = -0.1  # slightly lower for adult males
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * (2.0 ** octaves))
                    })
                    audio = audio.set_frame_rate(44100)
                
                # Export wav file
                audio = audio.set_channels(1)  # mono
                audio = audio.set_frame_rate(self.sample_rate)  # resample
                audio.export(file_path, format="wav")
                
                # Remove the temporary mp3 file
                os.remove(mp3_path)
                
                # Add metadata
                sample_metadata = {
                    'id': sample_id,
                    'file': filename,
                    'non_keyword': non_keyword,
                    'accent': accent_info["tld"],
                    'accent_name': accent_info["name"],
                    'age_group': age_group,
                    'gender': gender,
                    'duration_ms': len(audio),
                }
                
                self.metadata["non_keywords"]['samples'].append(sample_metadata)
                self.metadata["non_keywords"]['count'] += 1
                
                # Save metadata every 10 samples
                if i % 10 == 0:
                    self._save_metadata()
                
                # Add a small delay to prevent rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
        
        # Save final metadata
        self._save_metadata()
        print(f"Generated {num_samples} non-keyword samples")

def main():
    parser = argparse.ArgumentParser(description='Generate non-keyword speech samples using gTTS')
    parser.add_argument('--samples', type=int, default=DEFAULT_NON_KEYWORD_SAMPLES, 
                        help=f'Number of samples to generate (default: {DEFAULT_NON_KEYWORD_SAMPLES})')
    parser.add_argument('--output-dir', type=str, default=KEYWORDS_DIR, help='Output directory')
    parser.add_argument('--silence', type=int, default=DEFAULT_SILENCE_MS, 
                        help=f'Silence to add at beginning and end in milliseconds (default: {DEFAULT_SILENCE_MS})')
    parser.add_argument('--avoid-keyword', type=str, default=DEFAULT_KEYWORD,
                        help=f'Keyword to avoid using as non-keyword (default: {DEFAULT_KEYWORD})')
    args = parser.parse_args()
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    generator = NonKeywordGenerator(args.output_dir)
    generator.generate_non_keyword_samples(args.samples, args.silence, args.avoid_keyword)

if __name__ == "__main__":
    main()
