#!/bin/bash

# Script to run the test keyword detection
# Usage: ./run_workflow.sh
#   python scripts get defauts from config.py

# Extract default word from config.py
KEYWORD=$(grep "^DEFAULT_KEYWORD =" config.py | cut -d '=' -f 2 | cut -d '#' -f 1 | tr -d ' ' | sed 's/"//g')

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

# Step 1: Test model with TTS samples
echo -e "\n[Step 1/2] Testing model with TTS samples..."
python main.py test-tts
if [ $? -ne 0 ]; then
  echo "Error testing model with TTS samples."
  exit 1
fi

# Step 2: Test model with non-keywords samples
echo -e "\n[Step 2/2] Testing model with non-keyword samples..."
python main.py test-non-keywords
if [ $? -ne 0 ]; then
  echo "Error testing model with non-keyword samples."
  exit 1
fi

echo -e "\n======================================================="
echo "Tests completed successfully!"
echo "Trained model: $LATEST_MODEL"
echo -e "\nYou can now test the model using microphone input:"
echo "python main.py test-mic --model \"$LATEST_MODEL\""
echo -e "=======================================================\n"

# Ask if user wants to test with microphone
read -p "Do you want to test the model with microphone input now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  python main.py test-mic --model "$LATEST_MODEL"
fi
