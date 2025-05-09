#!/usr/bin/env python3
"""
Prepare trained model for Android integration.
"""

import os
import sys
import argparse
import json
import shutil
import tensorflow as tf
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import MODELS_DIR

def prepare_for_android(model_path, output_dir=None):
    """
    Prepare a trained model for Android integration.
    
    Args:
        model_path: Path to trained model (.tflite)
        output_dir: Output directory for Android assets
        
    Returns:
        output_path: Path to prepared model
    """
    if not model_path.endswith('.tflite'):
        raise ValueError("Model must be in TFLite format (.tflite)")
    
    # Get model name
    model_name = os.path.basename(model_path).split('.')[0]
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), 'android')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model file
    output_path = os.path.join(output_dir, f"{model_name}.tflite")
    shutil.copy(model_path, output_path)
    
    # Load model metadata
    model_dir = os.path.dirname(model_path)
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        
        # Find this specific model in the metadata
        model_info = None
        for model in model_metadata['models']:
            if model['name'] == model_name:
                model_info = model
                break
        
        if model_info:
            # Create config file for Android
            android_config = {
                'model_name': model_name,
                'keywords': model_info['keywords'],
                'input_shape': model_info['input_shape'],
                'num_classes': model_info['num_classes'],
                'feature_params': model_info['feature_params'],
                'sample_rate': 16000,
                'detection_threshold': 0.5,
                'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            config_path = os.path.join(output_dir, f"{model_name}_config.json")
            with open(config_path, 'w') as f:
                json.dump(android_config, f, indent=2)
            
            print(f"Created Android configuration file: {config_path}")
    
    print(f"Prepared model for Android: {output_path}")
    return output_path

def create_android_template(output_dir, package_name=None):
    """
    Create a template for Android integration.
    
    Args:
        output_dir: Output directory
        package_name: Android package name
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default package name if not provided
    if package_name is None:
        package_name = "com.example.keyworddetection"
    
    # Create KeywordDetector.kt
    kotlin_code = f"""package {package_name}

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt

class KeywordDetector(
    private val context: Context,
    private val modelPath: String,
    private val configPath: String,
    private val detectionListener: DetectionListener
) {{
    interface DetectionListener {{
        fun onKeywordDetected(keyword: String, confidence: Float)
        fun onError(exception: Exception)
    }}
    
    // Audio constants
    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT
    
    // Detection parameters
    private var keywords = listOf<String>()
    private var detectionThreshold = 0.5f
    private var isRunning = false
    
    // TFLite Interpreter
    private var interpreter: Interpreter? = null
    private var featureParams: Map<String, Any> = mapOf()
    private var inputShape: IntArray = intArrayOf()
    
    // Audio processing
    private var audioRecord: AudioRecord? = null
    private var audioBuffer: FloatArray? = null
    private var recordingThread: Thread? = null
    
    init {{
        try {{
            loadModel()
            loadConfig()
            setupAudio()
        }} catch (e: Exception) {{
            detectionListener.onError(e)
        }}
    }}
    
    private fun loadModel() {{
        val tfliteModel = loadModelFile(context, modelPath)
        val options = Interpreter.Options()
        interpreter = Interpreter(tfliteModel, options)
        
        // Get input shape from model
        val inputTensor = interpreter?.getInputTensor(0)
        inputShape = inputTensor?.shape() ?: intArrayOf(1, 0, 0)
    }}
    
    private fun loadConfig() {{
        try {{
            context.assets.open(configPath).use {{ inputStream ->
                val jsonString = inputStream.bufferedReader().use {{ it.readText() }}
                val config = org.json.JSONObject(jsonString)
                
                // Load keywords
                val keywordsArray = config.getJSONArray("keywords")
                val keywordsList = mutableListOf<String>()
                for (i in 0 until keywordsArray.length()) {{
                    keywordsList.add(keywordsArray.getString(i))
                }}
                keywords = keywordsList
                
                // Load detection threshold
                detectionThreshold = config.optDouble("detection_threshold", 0.5).toFloat()
                
                // Load feature parameters
                val featureParamsObj = config.optJSONObject("feature_params") ?: org.json.JSONObject()
                val params = mutableMapOf<String, Any>()
                val iter = featureParamsObj.keys()
                while (iter.hasNext()) {{
                    val key = iter.next()
                    params[key] = featureParamsObj.get(key)
                }}
                featureParams = params
            }}
        }} catch (e: Exception) {{
            throw RuntimeException("Error loading config file: $configPath", e)
        }}
    }}
    
    private fun setupAudio() {{
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT
        )
        
        // Use a buffer large enough to hold at least 2 seconds of audio
        val bufferSize = maxOf(minBufferSize, SAMPLE_RATE * 2)
        audioBuffer = FloatArray(bufferSize)
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        )
    }}
    
    fun startDetection() {{
        if (isRunning) return
        
        isRunning = true
        audioRecord?.startRecording()
        
        recordingThread = Thread(Runnable {{
            processAudioStream()
        }}, "AudioProcessingThread")
        
        recordingThread?.start()
    }}
    
    fun stopDetection() {{
        isRunning = false
        
        try {{
            recordingThread?.join(1000)
            audioRecord?.stop()
        }} catch (e: Exception) {{
            detectionListener.onError(e)
        }}
    }}
    
    fun release() {{
        stopDetection()
        audioRecord?.release()
        interpreter?.close()
    }}
    
    private fun processAudioStream() {{
        try {{
            while (isRunning) {{
                val audioChunk = readAudioChunk()
                
                if (audioChunk != null) {{
                    val features = extractFeatures(audioChunk)
                    
                    if (features != null) {{
                        // Run inference
                        val result = runInference(features)
                        
                        // Process results
                        processResults(result)
                    }}
                }}
                
                // Small delay to prevent CPU overuse
                Thread.sleep(20)
            }}
        }} catch (e: Exception) {{
            isRunning = false
            detectionListener.onError(e)
        }}
    }}
    
    private fun readAudioChunk(): FloatArray? {{
        val buffer = audioBuffer ?: return null
        
        // Read audio data
        val readSize = audioRecord?.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING) ?: 0
        
        if (readSize <= 0) {{
            return null
        }}
        
        // Return a copy of the read data
        return buffer.copyOfRange(0, readSize)
    }}
    
    private fun extractFeatures(audioData: FloatArray): FloatArray? {{
        try {{
            // This is a simplified feature extraction
            // In a real app, implement MFCC extraction here
            // using the feature parameters from the config
            
            // For now, just return normalized audio as placeholder
            val normalizedAudio = normalizeAudio(audioData)
            
            // Resize to match model input shape if needed
            val inputSize = inputShape[1]
            return if (normalizedAudio.size > inputSize) {{
                normalizedAudio.copyOfRange(0, inputSize)
            }} else if (normalizedAudio.size < inputSize) {{
                val padded = FloatArray(inputSize)
                System.arraycopy(normalizedAudio, 0, padded, 0, normalizedAudio.size)
                padded
            }} else {{
                normalizedAudio
            }}
        }} catch (e: Exception) {{
            detectionListener.onError(e)
            return null
        }}
    }}
    
    private fun normalizeAudio(audioData: FloatArray): FloatArray {{
        // Calculate RMS value
        var sum = 0f
        for (sample in audioData) {{
            sum += sample * sample
        }}
        val rms = sqrt(sum / audioData.size)
        
        // Skip normalization if audio is too quiet
        if (rms < 0.01f) {{
            return audioData
        }}
        
        // Normalize
        val normalized = FloatArray(audioData.size)
        for (i in audioData.indices) {{
            normalized[i] = audioData[i] / rms
        }}
        
        return normalized
    }}
    
    private fun runInference(features: FloatArray): FloatArray {{
        val outputSize = keywords.size + 1  // +1 for negative class
        val output = Array(1) {{ FloatArray(outputSize) }}
        
        // Reshape input to match model input shape
        val inputArray = Array(1) {{
            Array(inputShape[1]) {{
                FloatArray(inputShape[2]) {{ features[it] }}
            }}
        }}
        
        // Run inference
        interpreter?.run(inputArray, output)
        
        return output[0]
    }}
    
    private fun processResults(results: FloatArray) {{
        // Find highest scoring class
        var maxIndex = 0
        var maxValue = results[0]
        
        for (i in 1 until results.size) {{
            if (results[i] > maxValue) {{
                maxValue = results[i]
                maxIndex = i
            }}
        }}
        
        // Skip if it's the negative class (index 0) or below threshold
        if (maxIndex > 0 && maxValue >= detectionThreshold) {{
            val keywordIndex = maxIndex - 1  // Adjust for negative class
            
            if (keywordIndex < keywords.size) {{
                val keyword = keywords[keywordIndex]
                detectionListener.onKeywordDetected(keyword, maxValue)
            }}
        }}
    }}
    
    companion object {{
        private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {{
            val fileDescriptor = context.assets.openFd(modelPath)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }}
    }}
}}
"""

    # Create directory structure
    package_path = package_name.replace('.', '/')
    kotlin_dir = os.path.join(output_dir, 'app/src/main/java', package_path)
    os.makedirs(kotlin_dir, exist_ok=True)
    
    # Write KeywordDetector.kt
    with open(os.path.join(kotlin_dir, 'KeywordDetector.kt'), 'w') as f:
        f.write(kotlin_code)
    
    # Create README with integration instructions
    readme_content = """# Android Integration Guide

## Setup

1. Copy the model file(s) and configuration to your Android project's `assets` folder.

2. Add the following dependencies to your app's `build.gradle` file:

```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.8.0'
    
    // For audio processing (optional)
    implementation 'com.jlibrosa:jlibrosa:1.1.1'
}
```

3. Copy the `KeywordDetector.kt` file to your project's source directory.

## Usage

```kotlin
// Initialize the detector
val detector = KeywordDetector(
    context = applicationContext,
    modelPath = "keyword_detection_model.tflite",
    configPath = "keyword_detection_model_config.json",
    detectionListener = object : KeywordDetector.DetectionListener {
        override fun onKeywordDetected(keyword: String, confidence: Float) {
            // Handle keyword detection
            Log.d("KeywordDetection", "Detected: $keyword with confidence: $confidence")
        }
        
        override fun onError(exception: Exception) {
            // Handle errors
            Log.e("KeywordDetection", "Error", exception)
        }
    }
)

// Start detection
detector.startDetection()

// Stop detection when no longer needed
detector.stopDetection()

// Release resources when completely done
detector.release()
```

## Permissions

Add the following permission to your AndroidManifest.xml:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Don't forget to request the permission at runtime for Android 6.0+.
"""

    # Write README
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"Created Android template in {output_dir}")
    print(f"KeywordDetector.kt written to {os.path.join(kotlin_dir, 'KeywordDetector.kt')}")
    print(f"README with integration instructions written to {os.path.join(output_dir, 'README.md')}")

def main():
    parser = argparse.ArgumentParser(description='Prepare trained model for Android integration')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to trained model (.tflite)')
    parser.add_argument('--output-dir', type=str, 
                        help='Output directory for Android assets')
    parser.add_argument('--package-name', type=str, default="com.example.keyworddetection",
                        help='Android package name')
    parser.add_argument('--create-template', action='store_true',
                        help='Create Android integration template')
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths if needed
    if not os.path.isabs(args.model):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model = os.path.abspath(os.path.join(script_dir, args.model))
    
    if args.output_dir and not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # Prepare model for Android
    output_path = prepare_for_android(args.model, args.output_dir)
    
    # Create Android template if requested
    if args.create_template:
        template_dir = os.path.dirname(output_path) if args.output_dir is None else args.output_dir
        create_android_template(template_dir, args.package_name)

if __name__ == '__main__':
    main()
