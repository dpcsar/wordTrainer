"""
Generate keyword speech samples using Google Cloud Text-to-Speech with different voices and variations.
"""

import os
import sys
import argparse
import random
import numpy as np
import time
import io
from google.cloud import texttospeech
from pydub import AudioSegment
import uuid
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import from config
from config import (ACCENTS, AGE_GROUPS, SAMPLE_RATE, DEFAULT_KEYWORD_SAMPLES,
                    DEFAULT_SILENCE_MS, KEYWORDS_DIR, DEFAULT_KEYWORD,
                    GOOGLE_TTS_AUDIO_CONFIG)
from src.audio_utils import adjust_pitch_by_age

class KeywordGenerator:
    def __init__(self, output_dir, sample_rate=SAMPLE_RATE):
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
    
    def generate_keyword_samples(self, keyword, num_samples=DEFAULT_KEYWORD_SAMPLES, silence_ms=DEFAULT_SILENCE_MS):
        """
        Generate keyword speech samples with different accents, ages, and genders.
        
        Args:
            keyword: The keyword to generate samples for
            num_samples: Number of samples to generate (default: from config.DEFAULT_KEYWORD_SAMPLES)
            silence_ms: Silence to add at beginning and end in milliseconds (default: from config.DEFAULT_SILENCE_MS)
        """
        # Create keyword directory if it doesn't exist
        keyword_dir = os.path.join(self.output_dir, keyword)
        os.makedirs(keyword_dir, exist_ok=True)
        
        # Initialize keyword metadata if not exists
        if keyword not in self.metadata:
            self.metadata[keyword] = {
                'samples': [],
                'count': 0
            }
        
        # Check if we already have enough samples for this keyword
        existing_count = self.metadata[keyword]['count']
        samples_needed = max(0, num_samples - existing_count)
        
        if samples_needed <= 0:
            print(f"Already have {existing_count} samples for keyword '{keyword}', no additional samples needed.")
            return
        
        print(f"Generating {samples_needed} additional samples for keyword: '{keyword}' (already have {existing_count})")
        
        # Generate only the needed samples
        for i in tqdm(range(samples_needed)):
            # Randomly select accent, age, gender
            accent_info = random.choice(ACCENTS)
            age_group = random.choice(AGE_GROUPS)
            gender = accent_info["gender"]  # Use the gender from the accent info
            
            # Generate a unique ID for this sample
            sample_id = str(uuid.uuid4())[:8]
            
            # Get accent code for filename (e.g., "us", "uk", "au")
            country_code = accent_info["accent"]
                
            # Generate filename
            filename = f"{keyword}_{country_code}_{age_group}_{gender}_{sample_id}.wav"
            file_path = os.path.join(keyword_dir, filename)
            
            # Generate TTS audio
            try:
                # Initialize Google Cloud TTS client
                client = texttospeech.TextToSpeechClient()
                
                # Set the text input
                synthesis_input = texttospeech.SynthesisInput(text=keyword)
                
                # Select the voice based on accent info
                voice = texttospeech.VoiceSelectionParams(
                    language_code=accent_info["voice_name"].split("-")[0]+"-"+accent_info["voice_name"].split("-")[1],
                    name=accent_info["voice_name"],
                    ssml_gender=(texttospeech.SsmlVoiceGender.MALE if accent_info["gender"] == "male" 
                               else texttospeech.SsmlVoiceGender.FEMALE)
                )
                
                # Configure audio output using settings from config
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    speaking_rate=GOOGLE_TTS_AUDIO_CONFIG["speaking_rate"],
                    pitch=GOOGLE_TTS_AUDIO_CONFIG["pitch"],
                    volume_gain_db=GOOGLE_TTS_AUDIO_CONFIG["volume_gain_db"],
                    effects_profile_id=GOOGLE_TTS_AUDIO_CONFIG["effects_profile_id"]
                )
                
                # Generate speech
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                # Directly create AudioSegment from bytes
                audio = AudioSegment.from_wav(io.BytesIO(response.audio_content))
                
                # Add silence at beginning and end
                silence = AudioSegment.silent(duration=silence_ms)
                audio = silence + audio + silence
                
                # Adjust pitch based on age group only
                audio = adjust_pitch_by_age(audio, age_group)
                
                # Export wav file
                audio = audio.set_channels(1)  # mono
                audio = audio.set_frame_rate(self.sample_rate)  # resample
                audio.export(file_path, format="wav")
                
                # Add metadata
                sample_metadata = {
                    'id': sample_id,
                    'file': filename,
                    'keyword': keyword,
                    'accent': accent_info["accent"],
                    'accent_name': accent_info["accent_name"],
                    'voice_name': accent_info["voice_name"],
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
        print(f"Generated {samples_needed} additional samples for keyword '{keyword}', total now: {self.metadata[keyword]['count']}")

def main():
    parser = argparse.ArgumentParser(description='Generate keyword speech samples using Google Cloud TTS')
    parser.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD, help=f'Keyword to generate samples for (default: {DEFAULT_KEYWORD})')
    parser.add_argument('--samples', type=int, default=DEFAULT_KEYWORD_SAMPLES, 
                        help=f'Total number of samples desired (will only generate what is needed to reach this number) (default: {DEFAULT_KEYWORD_SAMPLES})')
    parser.add_argument('--output-dir', type=str, default=KEYWORDS_DIR, help='Output directory')
    parser.add_argument('--silence', type=int, default=DEFAULT_SILENCE_MS, 
                        help=f'Silence to add at beginning and end in milliseconds (default: {DEFAULT_SILENCE_MS})')
    args = parser.parse_args()
    
    generator = KeywordGenerator(args.output_dir)
    generator.generate_keyword_samples(args.keyword, args.samples, args.silence)

if __name__ == "__main__":
    main()
