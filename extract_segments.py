import os
import pandas as pd
from scipy.io import wavfile
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
DATA_DIR = BASE_DIR / "25108016/DataSubmission/DataSubmission"
OUTPUT_DIR = BASE_DIR / "extracted_segments"

# Create output directory 
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_segments_from_wav(wav_path, detections_path, output_folder):
    """
    Extract segments from a WAV file based on detection times.
    
    Args:
        wav_path: Path to the WAV file
        detections_path: Path to the detections txt file
        output_folder: Folder to save extracted segments
    """
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(wav_path)
    
    # Read the detections file
    detections = pd.read_csv(detections_path, sep='\t')
    
    # Filter to only keep Waveform views (remove duplicate Spectrogram rows)
    detections = detections[detections['View'] == 'Waveform 1'].reset_index(drop=True)
    
    # Create output folder for this recording
    recording_name = wav_path.stem
    recording_output_dir = output_folder / recording_name
    recording_output_dir.mkdir(exist_ok=True)
    
    # Track statistics
    valid_segments = 0
    skipped_segments = 0
    
    # Extract each segment
    for idx, row in detections.iterrows():
        # Get start and end times in seconds
        start_time = float(row['Begin Time (s)'])
        end_time = float(row['End Time (s)'])
        
        # Calculate duration
        duration = end_time - start_time
        
        # Skip if duration is too short (less than 10ms)
        if duration < 0.01:
            skipped_segments += 1
            print(f"  Skipping segment {idx}: duration={duration:.6f}s")
            continue
        
        # Convert to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure we don't go beyond the audio length
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        # Extract segment
        segment = audio_data[start_sample:end_sample]
        
        # Save segment
        segment_filename = f"{recording_name}_segment_{valid_segments:04d}_t{start_time:.3f}-{end_time:.3f}_dur{duration:.3f}s.wav"
        segment_path = recording_output_dir / segment_filename
        
        wavfile.write(segment_path, sample_rate, segment)
        valid_segments += 1
        
        # Print info for first few segments
        if valid_segments <= 3:
            print(f"  Segment {valid_segments}: start={start_time:.3f}s, end={end_time:.3f}s, duration={duration:.3f}s, samples={len(segment)}")
    
    print(f"Extracted {valid_segments} valid segments from {recording_name} (skipped {skipped_segments})")
    return valid_segments

def check_extracted_segments():
    """
    Check the extracted segments to verify they have proper duration.
    """
    print("\nChecking extracted segments...")
    
    # Check first few segments from each recording
    for recording_folder in OUTPUT_DIR.iterdir():
        if recording_folder.is_dir():
            print(f"\nRecording: {recording_folder.name}")
            wav_files = list(recording_folder.glob("*.wav"))[:5]  # Check first 5
            
            for wav_file in wav_files:
                try:
                    sample_rate, audio_data = wavfile.read(wav_file)
                    duration = len(audio_data) / sample_rate
                    print(f"  {wav_file.name}: duration={duration:.3f}s, samples={len(audio_data)}, sr={sample_rate}")
                except Exception as e:
                    print(f"  Error reading {wav_file.name}: {e}")

def process_all_recordings():
    """
    Process all recordings in the data directory.
    """
    total_segments = 0
    
    # Clear output directory first
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Iterate through all date folders
    for date_folder in DATA_DIR.iterdir():
        if date_folder.is_dir() and date_folder.name.startswith('2023'):
            print(f"\nProcessing folder: {date_folder.name}")
            
            # Find all WAV files in this folder
            wav_files = list(date_folder.glob("*.WAV"))
            
            for wav_file in wav_files:
                # Find corresponding detections file
                detections_file = wav_file.with_suffix('.Detections.selections.txt')
                
                if detections_file.exists():
                    print(f"Processing: {wav_file.name}")
                    num_segments = extract_segments_from_wav(
                        wav_file, 
                        detections_file, 
                        OUTPUT_DIR
                    )
                    total_segments += num_segments
                else:
                    print(f"Warning: No detections file found for {wav_file.name}")
    
    print(f"\nTotal segments extracted: {total_segments}")

if __name__ == "__main__":
    print("Starting segment extraction...")
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process all recordings
    process_all_recordings()
    
    # Check the extracted segments
    check_extracted_segments()