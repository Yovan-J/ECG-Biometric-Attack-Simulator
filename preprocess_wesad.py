import pickle
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
# !!! IMPORTANT !!!
# YOU MUST UPDATE THIS PATH to point to the location of your WESAD dataset folder.
WESAD_PATH = Path("/home/sauce/Documents/Cybersecurity/WESAD") # Replace this with the actual path

# --- Constants for WESAD dataset ---
CHEST_DEVICE_KEY = 'chest'
ECG_SIGNAL_KEY = 'ECG'
SAMPLING_RATE_HZ = 700

# --- Constants for data segmentation ---
SEGMENT_DURATION_SECONDS = 10
SEGMENT_SAMPLES = SEGMENT_DURATION_SECONDS * SAMPLING_RATE_HZ

def process_subject_data(subject_path: Path):
    """
    Processes a single subject's data file from the WESAD dataset,
    extracting raw ECG segments.
    
    Args:
        subject_path (Path): Path to a subject's .pkl file.

    Returns:
        list: A list of raw, normalized ECG signal segments.
    """
    subject_segments = []
    
    try:
        with open(subject_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    
        raw_ecg = data['signal'][CHEST_DEVICE_KEY][ECG_SIGNAL_KEY].flatten()
        
        num_segments = len(raw_ecg) // SEGMENT_SAMPLES
        
        for i in range(num_segments):
            start = i * SEGMENT_SAMPLES
            end = start + SEGMENT_SAMPLES
            segment = raw_ecg[start:end]
            
            # Normalize the segment to have zero mean and unit variance
            segment_mean = np.mean(segment)
            segment_std = np.std(segment)
            
            # Avoid division by zero for flat segments
            if segment_std > 0:
                normalized_segment = (segment - segment_mean) / segment_std
                subject_segments.append(normalized_segment)
                
    except Exception as e:
        print(f"Could not process {subject_path.name}: {e}")
        
    return subject_segments


def main():
    """
    Main function to iterate through all WESAD subjects, process their data,
    and save the raw ECG segments and labels to a compressed NumPy file.
    """
    if not WESAD_PATH.exists() or "path/to/your/WESAD" in str(WESAD_PATH):
        print("Error: WESAD_PATH is not configured correctly.")
        print("Please update the WESAD_PATH variable in this script.")
        return

    all_segments = []
    
    print("Starting WESAD data preprocessing for CNN model...")
    
    subject_folders = sorted([d for d in WESAD_PATH.iterdir() if d.is_dir() and d.name.startswith('S')])
    
    for subject_dir in subject_folders:
        subject_id = subject_dir.name
        data_file = subject_dir / f"{subject_id}.pkl"
        
        if data_file.exists():
            print(f"Processing subject: {subject_id}...")
            segments = process_subject_data(data_file)
            all_segments.extend(segments)
        else:
            print(f"Info: Data file for {subject_id} not found, skipping.")

    if not all_segments:
        print("No ECG segments were extracted. Aborting.")
        return

    # Convert the list of segments to a NumPy array
    segments_array = np.array(all_segments)
    
    # Create a corresponding array of labels (0 for genuine)
    labels_array = np.zeros(len(segments_array), dtype=int)
    
    print(f"\nPreprocessing complete. Extracted {len(segments_array)} total segments.")

    # Save both arrays to a single compressed .npz file
    output_path = Path("ecg_segments.npz")
    np.savez_compressed(output_path, segments=segments_array, labels=labels_array)
    
    print(f"Successfully saved data to '{output_path}'")


if __name__ == "__main__":
    main()