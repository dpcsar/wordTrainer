#!/usr/bin/env python3
"""
Test that non-keywords are correctly loaded in the training process.
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import KEYWORDS_DIR

def test_non_keywords_exist():
    """Test if non-keywords directory and metadata exist."""
    
    # Path to non-keywords directory
    non_keywords_dir = os.path.join(KEYWORDS_DIR, 'non_keywords')
    
    # Check if directory exists
    if not os.path.exists(non_keywords_dir):
        print("❌ Non-keywords directory not found:", non_keywords_dir)
        return False
    
    # Check if metadata file exists
    metadata_path = os.path.join(KEYWORDS_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        print("❌ Metadata file not found:", metadata_path)
        return False
    
    # Check if metadata contains non-keywords
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'non_keywords' not in metadata:
        print("❌ Non-keywords not found in metadata")
        return False
    
    if 'samples' not in metadata['non_keywords'] or not metadata['non_keywords']['samples']:
        print("❌ No non-keyword samples found in metadata")
        return False
    
    print(f"✅ Found {metadata['non_keywords']['count']} non-keyword samples")
    
    # Check for sample files
    sample_count = 0
    for sample in metadata['non_keywords']['samples']:
        file_path = os.path.join(non_keywords_dir, sample['file'])
        if os.path.exists(file_path):
            sample_count += 1
    
    print(f"✅ {sample_count} non-keyword sample files exist out of {len(metadata['non_keywords']['samples'])} in metadata")
    
    return sample_count > 0

if __name__ == "__main__":
    print("Testing non-keywords setup...")
    if test_non_keywords_exist():
        print("✅ Non-keywords setup passed!")
        sys.exit(0)
    else:
        print("❌ Non-keywords setup failed!")
        sys.exit(1)
