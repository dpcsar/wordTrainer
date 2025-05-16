#!/usr/bin/env python3
"""
Generate non-keyword speech samples using Google Cloud Text-to-Speech with different accents and variations.
These serve as negative examples for training that are actual words (unlike background noise).
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ACCENTS, NON_KEYWORDS, SAMPLE_RATE, DEFAULT_NON_KEYWORD_SAMPLES, 
                    DEFAULT_SILENCE_MS, KEYWORDS_DIR, DEFAULT_KEYWORD, GOOGLE_TTS_AUDIO_CONFIG)

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
        # Create non-keywords directory if it doesn't exist
        non_keywords_dir = os.path.join(self.output_dir, "non_keywords")
        os.makedirs(non_keywords_dir, exist_ok=True)
        
        # Initialize non-keywords metadata if not exists
        if "non_keywords" not in self.metadata:
            self.metadata["non_keywords"] = {
                'samples': [],
                'count': 0
            }
        
        # Check if we already have enough non-keyword samples
        existing_count = self.metadata["non_keywords"]['count']
        samples_needed = max(0, num_samples - existing_count)
        
        if samples_needed <= 0:
            print(f"Already have {existing_count} non-keyword samples, no additional samples needed.")
            return
        
        print(f"Generating {samples_needed} additional non-keyword samples (already have {existing_count})")
        
        # Filter out the keyword to avoid if provided
        available_words = [word for word in NON_KEYWORDS if word != keyword_to_avoid]
        if len(available_words) == 0:
            available_words = NON_KEYWORDS
        
        # Generate only the needed samples
        for i in tqdm(range(samples_needed)):
            # Select random non-keyword word
            non_keyword = random.choice(available_words)
            
            # Randomly select accent, age, gender
            accent_info = random.choice(ACCENTS)
            gender = accent_info["gender"]  # Use the gender from the accent info
            
            # Generate a unique ID for this sample
            sample_id = str(uuid.uuid4())[:8]
            
            # Get accent code for filename (e.g., "us", "uk", "au")
            country_code = accent_info["accent"]
                
            # Generate filename (replace spaces with underscores)
            filename = f"nonkw_{non_keyword}_{country_code}_{gender}_{sample_id}.wav".replace(' ', '_')
            file_path = os.path.join(non_keywords_dir, filename)
            # Also replace spaces with underscores in the path
            file_path = file_path.replace(' ', '_')
            
            # Generate TTS audio
            try:
                # Initialize Google Cloud TTS client
                client = texttospeech.TextToSpeechClient()
                
                # Set the text input
                synthesis_input = texttospeech.SynthesisInput(text=non_keyword)
                
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
                
                # Export wav file
                audio = audio.set_channels(1)  # mono
                audio = audio.set_frame_rate(self.sample_rate)  # resample
                audio.export(file_path, format="wav")
                
                # Add metadata
                sample_metadata = {
                    'id': sample_id,
                    'file': filename,
                    'non_keyword': non_keyword,
                    'accent': accent_info["accent"],
                    'accent_name': accent_info["accent_name"],
                    'voice_name': accent_info["voice_name"],
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
        print(f"Generated {samples_needed} non-keyword samples")

def main():
    parser = argparse.ArgumentParser(description='Generate non-keyword speech samples using Google Cloud Text-to-Speech API')
    parser.add_argument('--samples', type=int, default=DEFAULT_NON_KEYWORD_SAMPLES, 
                        help=f'Number of non-keyword samples to generate for training (default: {DEFAULT_NON_KEYWORD_SAMPLES})')
    parser.add_argument('--output-dir', type=str, default=KEYWORDS_DIR, 
                        help=f'Output directory for generated non-keyword samples (default: {KEYWORDS_DIR})')
    parser.add_argument('--silence', type=int, default=DEFAULT_SILENCE_MS, 
                        help=f'Silence padding to add at beginning and end in milliseconds (default: {DEFAULT_SILENCE_MS}ms)')
    parser.add_argument('--avoid-keyword', type=str, default=DEFAULT_KEYWORD,
                        help=f'Target keyword to avoid using in non-keyword phrases (default: "{DEFAULT_KEYWORD}")')
    args = parser.parse_args()
    
    generator = NonKeywordGenerator(args.output_dir)
    generator.generate_non_keyword_samples(args.samples, args.silence, args.avoid_keyword)

if __name__ == "__main__":
    main()
