#!/bin/bash

# Script to run the complete keyword detection workflow
# Usage: ./run_workflow.sh <keyword>

# Check if keyword is provided
if [ -z "$1" ]; then
  echo "Usage: ./run_workflow.sh <keyword>"
  exit 1
fi

KEYWORD=$1
SAMPLES=50
SAMPLES_TO_TEST=5
NON_KEYWORDS_SAMPLES=50
NON_KEYWORDS_SAMPLES_TO_TEST=5
BG_SAMPLES=200
NUM_MIXES=100
MIN_SNR=-5
MAX_SNR=20
EPOCHS=50
BATCH_SIZE=32
THRESHOLD=0.6

echo "======================================================="
echo "Starting keyword detection workflow for keyword: $KEYWORD"
echo "======================================================"

# Step 1: Generate keyword samples
echo -e "\n[Step 1/7] Generating keyword samples using gTTS..."
python main.py generate-keywords --keyword "$KEYWORD" --samples $SAMPLES
if [ $? -ne 0 ]; then
  echo "Error generating keyword samples. Exiting."
  exit 1
fi

# Step 2: Generate background noise if needed
echo -e "\n[Step 2/7] Checking for background noise samples..."
if [ ! -f "data/backgrounds/metadata.json" ]; then
  echo "Generating background noise samples..."
  python main.py generate-noise --type all --samples $BG_SAMPLES
  if [ $? -ne 0 ]; then
    echo "Error generating background noise. Exiting."
    exit 1
  fi
else
  echo "Background noise samples already exist. Skipping generation."
fi

# Step 3: Generate non-keywords for negative training
echo -e "\n[Step 3/7] Generating non-keyword samples..."
python main.py generate-non-keywords --samples $NON_KEYWORDS_SAMPLES --avoid-keyword "$KEYWORD"
if [ $? -ne 0 ]; then
  echo "Error generating non-keyword samples. Exiting."
  exit 1
fi

# Step 4: Mix keyword samples with background noise
echo -e "\n[Step 4/7] Mixing keyword samples with background noise..."
python main.py mix-audio --keyword "$KEYWORD" --num-mixes $NUM_MIXES --min-snr $MIN_SNR --max-snr $MAX_SNR
if [ $? -ne 0 ]; then
  echo "Error mixing audio samples. Exiting."
  exit 1
fi

# Step 5: Train model
echo -e "\n[Step 5/7] Training keyword detection model..."
python main.py train --keywords "$KEYWORD" --epochs $EPOCHS --batch-size $BATCH_SIZE
if [ $? -ne 0 ]; then
  echo "Error training model. Exiting."
  exit 1
fi

# Find the latest model
LATEST_MODEL=$(find models -name "keyword_detection_${KEYWORD}*.tflite" | sort | tail -n 1)
if [ -z "$LATEST_MODEL" ]; then
  echo "No trained model found. Exiting."
  exit 1
fi

echo "Latest model: $LATEST_MODEL"

# Step 6: Test model with gTTS samples
echo -e "\n[Step 6/7] Testing model with gTTS samples..."
python main.py test-gtts --model "$LATEST_MODEL" --samples $SAMPLES_TO_TEST
if [ $? -ne 0 ]; then
  echo "Error testing model with gTTS samples."
  exit 1
fi

# Step 7: Test model with non-keywords samples
echo -e "\n[Step 7/7] Testing model with non-keyword samples..."
python main.py test-non-keywords --model "$LATEST_MODEL" --samples $NON_KEYWORDS_SAMPLES_TO_TEST
if [ $? -ne 0 ]; then
  echo "Error testing model with non-keyword samples."
  exit 1
fi

echo -e "\n======================================================="
echo "Workflow completed successfully!"
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
