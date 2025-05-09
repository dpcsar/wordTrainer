"""
Generate keyword speech samples using gTTS with different accents and variations.
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
from utils.config import ACCENTS, AGE_GROUPS, GENDERS, TLD_TO_COUNTRY

class KeywordGenerator:
    def __init__(self, output_dir, sample_rate=16000):
        """
        Initialize KeywordGenerator.
        
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
    
    def generate_keyword_samples(self, keyword, num_samples=100, silence_ms=500):
        """
        Generate keyword speech samples with different accents, ages, and genders.
        
        Args:
            keyword: The keyword to generate samples for
            num_samples: Number of samples to generate
            silence_ms: Silence to add at beginning and end (milliseconds)
        """
        print(f"Generating {num_samples} samples for keyword: '{keyword}'")
        
        # Create keyword directory if it doesn't exist
        keyword_dir = os.path.join(self.output_dir, keyword)
        os.makedirs(keyword_dir, exist_ok=True)
        
        # Initialize keyword metadata if not exists
        if keyword not in self.metadata:
            self.metadata[keyword] = {
                'samples': [],
                'count': 0
            }
        
        # Generate samples
        for i in tqdm(range(num_samples)):
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
            filename = f"{keyword}_{country_code}_{age_group}_{gender}_{sample_id}.wav"
            file_path = os.path.join(keyword_dir, filename)
            
            # Generate TTS audio
            try:
                tts = gTTS(text=keyword, lang=accent_info["lang"], tld=accent_info["tld"])
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
                    'keyword': keyword,
                    'accent': accent_info["tld"],
                    'accent_name': accent_info["name"],
                    'age_group': age_group,
                    'gender': gender,
                    'duration_ms': len(audio),
                }
                
                self.metadata[keyword]['samples'].append(sample_metadata)
                self.metadata[keyword]['count'] += 1
                
                # Save metadata every 10 samples
                if i % 10 == 0:
                    self._save_metadata()
                
                # Add a small delay to prevent rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
        
        # Save final metadata
        self._save_metadata()
        print(f"Generated {num_samples} samples for keyword '{keyword}'")

def main():
    parser = argparse.ArgumentParser(description='Generate keyword speech samples using gTTS')
    parser.add_argument('--keyword', type=str, required=True, help='Keyword to generate samples for')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='../data/keywords', help='Output directory')
    parser.add_argument('--silence', type=int, default=500, help='Silence to add at beginning and end (milliseconds)')
    args = parser.parse_args()
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    generator = KeywordGenerator(args.output_dir)
    generator.generate_keyword_samples(args.keyword, args.samples, args.silence)

if __name__ == "__main__":
    main()
