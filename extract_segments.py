import os
import pandas as pd
from scipy.io import wavfile
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
DATA_DIR = BASE_DIR / "25108016/DataSubmission/DataSubmission"
OUTPUT_DIR = BASE_DIR / "extracted_segments_padded"

# Create output directory 
OUTPUT_DIR.mkdir(exist_ok=True)

# Extraction parameters
EXTRACTION_PARAMS = {
    'pre_padding': 0.3,   # Add 300ms before detection start
    'post_padding': 0.3,  # Add 300ms after detection end
    'min_duration': 0.5,  # Minimum segment duration after padding
    'target_duration': 1.0,  # Target duration for consistent segments
    'max_duration': 1.6,  # Maximum duration (as mentioned in paper)
}

def extract_segments_with_padding(wav_path, detections_path, output_folder, params):
    """
    Extract segments from a WAV file with padding around detections.
    
    Args:
        wav_path: Path to the WAV file
        detections_path: Path to the detections txt file
        output_folder: Folder to save extracted segments
        params: Extraction parameters dictionary
    """
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(wav_path)
    audio_duration = len(audio_data) / sample_rate
    
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
    merged_segments = 0
    
    # Sort detections by start time
    detections = detections.sort_values('Begin Time (s)')
    
    # Merge overlapping or close detections
    merged_detections = []
    i = 0
    while i < len(detections):
        current_start = float(detections.iloc[i]['Begin Time (s)'])
        current_end = float(detections.iloc[i]['End Time (s)'])
        
        # Add padding
        padded_start = max(0, current_start - params['pre_padding'])
        padded_end = min(audio_duration, current_end + params['post_padding'])
        
        # Check for overlapping or close subsequent detections
        j = i + 1
        while j < len(detections):
            next_start = float(detections.iloc[j]['Begin Time (s)'])
            next_end = float(detections.iloc[j]['End Time (s)'])
            next_padded_start = max(0, next_start - params['pre_padding'])
            
            # If next detection overlaps or is very close, merge them
            if next_padded_start <= padded_end + 0.1:  # 100ms gap tolerance
                padded_end = min(audio_duration, next_end + params['post_padding'])
                merged_segments += 1
                j += 1
            else:
                break
        
        merged_detections.append({
            'start': padded_start,
            'end': padded_end,
            'original_start': current_start,
            'original_end': detections.iloc[j-1]['End Time (s)'] if j > i + 1 else current_end,
            'merged_count': j - i
        })
        i = j
    
    print(f"  Merged {merged_segments} overlapping detections into {len(merged_detections)} segments")
    
    # Extract each merged segment
    for idx, detection in enumerate(merged_detections):
        start_time = detection['start']
        end_time = detection['end']
        duration = end_time - start_time
        
        # Apply duration constraints
        if duration < params['min_duration']:
            # Extend to minimum duration, centered on detection
            center = (start_time + end_time) / 2
            start_time = max(0, center - params['min_duration'] / 2)
            end_time = min(audio_duration, start_time + params['min_duration'])
            duration = end_time - start_time
        
        elif duration > params['max_duration']:
            # Trim to maximum duration, keeping center
            center = (start_time + end_time) / 2
            start_time = max(0, center - params['max_duration'] / 2)
            end_time = min(audio_duration, start_time + params['max_duration'])
            duration = end_time - start_time
        
        # For consistency, we can optionally pad/trim to target duration
        if params.get('force_target_duration', False):
            if duration < params['target_duration']:
                # Pad to target duration
                padding_needed = params['target_duration'] - duration
                start_time = max(0, start_time - padding_needed / 2)
                end_time = min(audio_duration, end_time + padding_needed / 2)
                duration = end_time - start_time
        
        # Convert to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Extract segment
        segment = audio_data[start_sample:end_sample]
        
        # Save segment with informative filename
        segment_filename = (f"{recording_name}_segment_{valid_segments:04d}_"
                          f"t{detection['original_start']:.3f}-{detection['original_end']:.3f}_"
                          f"padded_{start_time:.3f}-{end_time:.3f}_"
                          f"dur{duration:.3f}s.wav")
        segment_path = recording_output_dir / segment_filename
        
        wavfile.write(segment_path, sample_rate, segment)
        valid_segments += 1
        
        # Print info for first few segments
        if valid_segments <= 3:
            print(f"    Segment {valid_segments}: "
                  f"original=[{detection['original_start']:.3f}, {detection['original_end']:.3f}]s, "
                  f"padded=[{start_time:.3f}, {end_time:.3f}]s, "
                  f"duration={duration:.3f}s, merged={detection['merged_count']} calls")
    
    print(f"  Extracted {valid_segments} segments from {recording_name}")
    return valid_segments

def analyze_detection_patterns(data_dir):
    """
    Analyze detection patterns to determine optimal padding parameters.
    """
    print("\nAnalyzing detection patterns...")
    
    all_durations = []
    all_gaps = []
    
    for date_folder in data_dir.iterdir():
        if date_folder.is_dir() and date_folder.name.startswith('2023'):
            for wav_file in date_folder.glob("*.WAV"):
                detections_file = wav_file.with_suffix('.Detections.selections.txt')
                if detections_file.exists():
                    detections = pd.read_csv(detections_file, sep='\t')
                    detections = detections[detections['View'] == 'Waveform 1']
                    
                    # Calculate durations
                    durations = detections['End Time (s)'] - detections['Begin Time (s)']
                    all_durations.extend(durations.tolist())
                    
                    # Calculate gaps between consecutive detections
                    if len(detections) > 1:
                        sorted_detections = detections.sort_values('Begin Time (s)')
                        for i in range(len(sorted_detections) - 1):
                            gap = (sorted_detections.iloc[i+1]['Begin Time (s)'] - 
                                  sorted_detections.iloc[i]['End Time (s)'])
                            if gap > 0:
                                all_gaps.append(gap)
    
    if all_durations:
        print(f"  Detection durations: min={min(all_durations):.3f}s, "
              f"max={max(all_durations):.3f}s, "
              f"mean={np.mean(all_durations):.3f}s, "
              f"median={np.median(all_durations):.3f}s")
    
    if all_gaps:
        print(f"  Gaps between detections: min={min(all_gaps):.3f}s, "
              f"max={max(all_gaps):.3f}s, "
              f"mean={np.mean(all_gaps):.3f}s, "
              f"median={np.median(all_gaps):.3f}s")
        
        # Suggest padding based on gap analysis
        typical_gap = np.percentile(all_gaps, 25)  # 25th percentile of gaps
        suggested_padding = min(0.5, typical_gap / 2)  # Don't exceed 500ms padding
        print(f"  Suggested padding: {suggested_padding:.3f}s")
    
    return all_durations, all_gaps

def check_extracted_segments(output_dir):
    """
    Check the extracted segments to verify they have proper duration.
    """
    print("\nChecking extracted segments...")
    
    all_durations = []
    
    for recording_folder in output_dir.iterdir():
        if recording_folder.is_dir():
            wav_files = list(recording_folder.glob("*.wav"))
            
            for wav_file in wav_files:
                try:
                    sample_rate, audio_data = wavfile.read(wav_file)
                    duration = len(audio_data) / sample_rate
                    all_durations.append(duration)
                except Exception as e:
                    print(f"  Error reading {wav_file.name}: {e}")
    
    if all_durations:
        print(f"  Total segments: {len(all_durations)}")
        print(f"  Duration stats: min={min(all_durations):.3f}s, "
              f"max={max(all_durations):.3f}s, "
              f"mean={np.mean(all_durations):.3f}s, "
              f"std={np.std(all_durations):.3f}s")
        
        # Duration distribution
        bins = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        hist, _ = np.histogram(all_durations, bins=bins)
        print("  Duration distribution:")
        for i in range(len(hist)):
            if i < len(bins) - 1:
                print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]} segments")

def process_all_recordings(params):
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
                    print(f"  Processing: {wav_file.name}")
                    num_segments = extract_segments_with_padding(
                        wav_file, 
                        detections_file, 
                        OUTPUT_DIR,
                        params
                    )
                    total_segments += num_segments
                else:
                    print(f"  Warning: No detections file found for {wav_file.name}")
    
    print(f"\nTotal segments extracted: {total_segments}")
    return total_segments

if __name__ == "__main__":
    print("Starting improved segment extraction with padding...")
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # First, analyze detection patterns
    durations, gaps = analyze_detection_patterns(DATA_DIR)
    
    print(f"\nExtraction parameters:")
    for key, value in EXTRACTION_PARAMS.items():
        print(f"  {key}: {value}")
    
    # Process all recordings with padding
    total_segments = process_all_recordings(EXTRACTION_PARAMS)
    
    # Check the extracted segments
    check_extracted_segments(OUTPUT_DIR)
    
    print("\n✓ Segment extraction complete!")
    print(f"Segments saved to: {OUTPUT_DIR}")