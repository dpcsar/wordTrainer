# Google Cloud Text-to-Speech Setup for wordTrainer

This guide explains how to set up Google Cloud Text-to-Speech for generating keyword samples in the wordTrainer project.

> Note: All Google TTS configuration is now located in the main `config.py` file. This allows for easy customization of voices and audio settings in one central location.

## Prerequisites

1. A Google Cloud account
2. Google Cloud SDK (gcloud) installed
3. A Google Cloud project with the Text-to-Speech API enabled
4. Service account credentials for your project

## Setting Up Google Cloud Text-to-Speech

### 1. Install Google Cloud SDK (gcloud)

#### For Linux:
```bash
# Add the Cloud SDK distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# Update and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk

# Initialize the SDK
gcloud init
```

#### For macOS:
```bash
# Install using Homebrew
brew install --cask google-cloud-sdk

# Initialize the SDK
gcloud init
```

#### For Windows:
1. Download the Google Cloud SDK installer from [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
2. Run the installer and follow the prompts
3. After installation, open a command prompt and run:
   ```
   gcloud init
   ```

After installation, enable the Text-to-Speech API for your project:
```bash
gcloud services enable texttospeech.googleapis.com
```

### 2. Create a Google Cloud Account and Project

If you don't have a Google Cloud account, sign up at [cloud.google.com](https://cloud.google.com/).

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your project ID - you'll need it later

### 3. Enable the Text-to-Speech API

1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Text-to-Speech"
3. Click on "Cloud Text-to-Speech API"
4. Click "Enable"

### 4. Create Service Account Credentials

1. In the Google Cloud Console, go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" and select "Service Account"
3. Fill in the details and create the account
4. Click on the newly created service account
5. Go to the "Keys" tab
6. Click "Add Key" > "Create new key"
7. Select JSON and click "Create"
8. Save the downloaded JSON file securely (this is your credentials file)

### 5. Verify Setup in wordTrainer

1. Place your credentials file in the project directory (e.g., `google_credentials.json`)
2. Run the setup verification script:

```bash
python setup_google_tts.py --credentials path/to/your/google_credentials.json
```

This script will:
- Check if your credentials are valid
- Test generating a sample audio file
- Show available voices in your account

### 6. Set Environment Variable

Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your credentials file:

```bash
# For temporary use in current terminal
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google_credentials.json"

# For permanent use, add to your .bashrc or .zshrc
echo 'export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google_credentials.json"' >> ~/.zshrc
```

## Using Google Cloud Text-to-Speech

Once set up, you can generate keyword samples using:

```bash
python main.py generate-keywords --keyword "your keyword" --samples 50
```

## Free Tier Information

Google Cloud Text-to-Speech offers a generous free tier:
- 1 million characters per month for standard voices
- 4 million characters per month for Neural/WaveNet voices
- This is sufficient for most training needs

## Additional Resources

- [Google Cloud Text-to-Speech Documentation](https://cloud.google.com/text-to-speech)
- [Python Client Documentation](https://googleapis.dev/python/texttospeech/latest/index.html)
