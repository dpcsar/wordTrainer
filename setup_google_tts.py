#!/usr/bin/env python3
"""
Setup script for Google Cloud Text-to-Speech in the wordTrainer project.
This script verifies that Google Cloud TTS is properly set up and functional.
"""

import os
import sys
import argparse
from google.cloud import texttospeech
from pydub import AudioSegment
import io

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import GOOGLE_TTS_VOICES, GOOGLE_TTS_AUDIO_CONFIG, SAMPLE_RATE

def setup_google_tts(credentials_path, test_phrase="Testing Google Cloud Text-to-Speech"):
    """
    Set up and test Google Cloud Text-to-Speech service.
    
    Args:
        credentials_path: Path to Google Cloud service account JSON credentials file
        test_phrase: Phrase to test TTS with
    """
    # Check if credentials file exists
    if not os.path.exists(credentials_path):
        print(f"Error: Credentials file not found at {credentials_path}")
        return False
    
    # Set the environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    
    try:
        # Initialize the client
        client = texttospeech.TextToSpeechClient()
        
        # Print authentication success
        print(f"✓ Successfully authenticated with Google Cloud")
        
        # Test a sample text-to-speech request
        print(f"Testing TTS with phrase: '{test_phrase}'")
        
        # Set the text input
        synthesis_input = texttospeech.SynthesisInput(text=test_phrase)
        
        # Build the voice request - using the first US male voice from our config
        us_male_voice = next(voice for voice in GOOGLE_TTS_VOICES 
                          if voice["accent"] == "us" and voice["gender"] == "male")
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=us_male_voice["name"].split("-")[0] + "-" + us_male_voice["name"].split("-")[1],
            name=us_male_voice["name"],
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        
        # Select the type of audio file to return - using our config settings
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            speaking_rate=GOOGLE_TTS_AUDIO_CONFIG["speaking_rate"],
            pitch=GOOGLE_TTS_AUDIO_CONFIG["pitch"],
            volume_gain_db=GOOGLE_TTS_AUDIO_CONFIG["volume_gain_db"],
            effects_profile_id=GOOGLE_TTS_AUDIO_CONFIG["effects_profile_id"]
        )
        
        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save the audio to a test file
        test_output = "google_tts_test.wav"
        with open(test_output, "wb") as out:
            out.write(response.audio_content)
            print(f"✓ Audio content written to '{test_output}'")
        
        # List available voices
        voices = client.list_voices().voices
        english_voices = [v for v in voices if "en-" in v.name]
        
        print(f"\nDetected {len(english_voices)} English voices available in your Google Cloud account.")
        print("Sample of available voices:")
        
        # Print a subset of voices to verify
        sample_size = min(10, len(english_voices))
        for i in range(sample_size):
            voice = english_voices[i]
            gender = "Male" if voice.ssml_gender == texttospeech.SsmlVoiceGender.MALE else "Female"
            print(f"  - {voice.name} ({gender}): {voice.language_codes[0]}")
        
        print("\n✓ Google Cloud Text-to-Speech is successfully set up!")
        print("\nTo use this in the wordTrainer project:")
        print(f"1. Add this to your environment: export GOOGLE_APPLICATION_CREDENTIALS=\"{credentials_path}\"")
        print("2. Run the generator: python main.py generate-keywords --keyword \"your keyword\"")
        
        return True
        
    except Exception as e:
        print(f"Error: Failed to initialize Google Cloud Text-to-Speech: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Set up Google Cloud Text-to-Speech for wordTrainer')
    parser.add_argument('--credentials', type=str, required=True, 
                        help='Path to Google Cloud service account JSON credentials file')
    parser.add_argument('--test-phrase', type=str, default="Testing Google Cloud Text-to-Speech", 
                        help='Phrase to test TTS with')
    
    args = parser.parse_args()
    
    setup_google_tts(args.credentials, args.test_phrase)

if __name__ == "__main__":
    main()
