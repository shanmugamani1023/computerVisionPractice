import os
print(os.environ.get('CONDA_DEFAULT_ENV'))
import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Path to the audio file (use the WAV version I provided)
audio_file_path = "bin/activation_function_revision.m4a"

# Transcribe the audio
with sr.AudioFile(audio_file_path) as source:
    audio_data = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio_data)
        print("Transcript:", transcript)
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print(f"API error: {str(e)}")
