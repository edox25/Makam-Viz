import librosa
import numpy as np
import json
import sys
import os

def extract_audio_features(audio_path, output_path):
    """
    Extract audio features for visualization:
    - RMS (Root Mean Square) for orb pulsation
    - Spectral Centroid for color intensity
    - Onset Events for rotation speed
    """
    
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Extract RMS (for orb pulsation)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    
    # Extract Spectral Centroid (for color intensity)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Extract Onset Events (for rotation speed)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    
    # Create time array for synchronization
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    
    # Normalize features for visualization
    rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    centroid_normalized = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid))
    
    # Create feature data structure
    features = {
        'duration': float(librosa.get_duration(y=y, sr=sr)),
        'sample_rate': int(sr),
        'rms': {
            'values': rms_normalized.tolist(),
            'times': times.tolist()
        },
        'spectral_centroid': {
            'values': centroid_normalized.tolist(),
            'times': times.tolist()
        },
        'onsets': onset_frames.tolist()
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    
    print(f"Audio features extracted and saved to {output_path}")
    print(f"Duration: {features['duration']:.2f} seconds")
    print(f"RMS frames: {len(rms)}")
    print(f"Spectral centroid frames: {len(spectral_centroid)}")
    print(f"Onset events: {len(onset_frames)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_audio_features.py <input_audio_path> <output_json_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
    
    extract_audio_features(audio_path, output_path) 