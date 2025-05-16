#!/bin/bash

# Script to run the complete keyword detection workflow
# Usage: ./run_workflow.sh
#   python scripts get defauts from config.py

# Extract default word from config.py
KEYWORD=$(grep "^DEFAULT_KEYWORD =" config.py | cut -d '=' -f 2 | cut -d '#' -f 1 | tr -d ' ' | sed 's/"//g')

echo "======================================================="
echo "Starting keyword detection workflow for keyword: $KEYWORD"
echo "======================================================="

# Step 1: Generate keyword samples
echo -e "\n[Step 1/7] Generating keyword samples using TTS..."
python main.py generate-keywords
if [ $? -ne 0 ]; then
  echo "Error generating keyword samples. Exiting."
  exit 1
fi

# Step 2: Generate background noise if needed
echo -e "\n[Step 2/7] Checking for background noise samples..."
if [ ! -f "data/backgrounds/metadata.json" ]; then
  echo "Generating background noise samples..."
  python main.py generate-noise
  if [ $? -ne 0 ]; then
    echo "Error generating background noise. Exiting."
    exit 1
  fi
else
  echo "Background noise samples already exist. Skipping generation."
fi

# Step 3: Generate non-keywords for negative training
echo -e "\n[Step 3/7] Generating non-keyword samples..."
python main.py generate-non-keywords
if [ $? -ne 0 ]; then
  echo "Error generating non-keyword samples. Exiting."
  exit 1
fi

# Step 4: Mix keyword samples with background noise
echo -e "\n[Step 4/7] Mixing keyword samples with background noise..."
python main.py mix-audio
if [ $? -ne 0 ]; then
  echo "Error mixing audio samples. Exiting."
  exit 1
fi

# Step 5: Train model
echo -e "\n[Step 5/7] Training keyword detection model..."
python main.py train
if [ $? -ne 0 ]; then
  echo "Error training model. Exiting."
  exit 1
fi

# Find the latest model
# Replace spaces with underscores in keyword for finding the model
SANITIZED_KEYWORD=${KEYWORD// /_}
LATEST_MODEL=$(find models -name "keyword_detection_${SANITIZED_KEYWORD}*.tflite" | sort | tail -n 1)
if [ -z "$LATEST_MODEL" ]; then
  echo "No trained model found. Exiting."
  exit 1
fi

echo "Latest model: $LATEST_MODEL"

# Step 6: Test model with TTS samples
echo -e "\n[Step 6/7] Testing model with TTS samples..."
python main.py test-tts
if [ $? -ne 0 ]; then
  echo "Error testing model with TTS samples."
  exit 1
fi

# Step 7: Test model with non-keywords samples
echo -e "\n[Step 7/7] Testing model with non-keyword samples..."
python main.py test-non-keywords
if [ $? -ne 0 ]; then
  echo "Error testing model with non-keyword samples."
  exit 1
fi

echo -e "\n======================================================="
echo "Workflow completed successfully!"
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
