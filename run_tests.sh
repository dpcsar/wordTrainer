#!/bin/bash

# Script to run the test keyword detection
# Usage: ./run_test.sh [keyword]

# Extract defaults from config.py
DEFAULT_KEYWORD=$(grep "DEFAULT_KEYWORD" config.py | cut -d '=' -f 2 | cut -d '#' -f 1 | tr -d ' ' | tr -d '"')

# Use provided keyword or default
KEYWORD=${1:-$DEFAULT_KEYWORD}

# Extract other defaults from config.py
SAMPLES_TO_TEST=$(grep "DEFAULT_TEST_SAMPLES" config.py | cut -d '=' -f 2 | tr -d ' ')
NON_KEYWORDS_SAMPLES_TO_TEST=$(grep "DEFAULT_TEST_SAMPLES" config.py | cut -d '=' -f 2 | tr -d ' ')
THRESHOLD=$(grep "DEFAULT_DETECTION_THRESHOLD" config.py | cut -d '=' -f 2 | tr -d ' ')

echo "======================================================="
echo "Starting keyword detection tests for keyword: $KEYWORD"
echo "======================================================="

# Find the latest model
LATEST_MODEL=$(find models -name "keyword_detection_${KEYWORD}*.tflite" | sort | tail -n 1)
if [ -z "$LATEST_MODEL" ]; then
  echo "No trained model found. Exiting."
  exit 1
fi

echo "Latest model: $LATEST_MODEL"

# Step 1: Test model with gTTS samples
echo -e "\n[Step 1/2] Testing model with gTTS samples..."
python main.py test-gtts --model "$LATEST_MODEL" --samples $SAMPLES_TO_TEST
if [ $? -ne 0 ]; then
  echo "Error testing model with gTTS samples."
  exit 1
fi

# Step 2: Test model with non-keywords samples
echo -e "\n[Step 2/2] Testing model with gTTS samples..."
python main.py test-non-keywords --model "$LATEST_MODEL" --samples $NON_KEYWORDS_SAMPLES_TO_TEST
if [ $? -ne 0 ]; then
  echo "Error testing model with gTTS samples."
  exit 1
fi

echo -e "\n======================================================="
echo "Tests completed successfully!"
echo "Trained model: $LATEST_MODEL"
echo -e "\nYou can now test the model using microphone input:"
echo "python main.py test-mic --model \"$LATEST_MODEL\" --threshold $THRESHOLD"
echo -e "=======================================================\n"

# Ask if user wants to test with microphone
read -p "Do you want to test the model with microphone input now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  python main.py test-mic --model "$LATEST_MODEL" --threshold $THRESHOLD
fi
