import numpy as np
from scipy.signal import find_peaks

def simulate_replay_attack(genuine_signal: np.ndarray, fs: int = 700, loop_type: str = 'half'):
    """
    Simulates a replay attack by creating a looped pattern from the genuine signal.
    
    This represents an attacker who:
    1. Records a genuine ECG segment
    2. Extracts a good quality portion
    3. Loops it to create a longer signal for replay
    
    Args:
        genuine_signal (np.ndarray): The legitimate ECG signal (10 seconds at 700Hz = 7000 samples).
        fs (int): The sampling frequency. Default is 700 Hz for WESAD.
        loop_type (str): Type of looping:
            - 'half': Loop the first 5 seconds twice (default, easiest to detect)
            - 'third': Loop the first 3.33 seconds three times
            - 'full': Return full signal with minimal noise (hardest to detect)
    
    Returns:
        np.ndarray: The replayed ECG signal with looping pattern.
    """
    if loop_type == 'half':
        # Loop the first half of the signal twice
        # This creates a 5-second pattern that repeats
        segment_length = len(genuine_signal) // 2
        segment = genuine_signal[:segment_length]
        looped_signal = np.tile(segment, 2)[:len(genuine_signal)]
        
    elif loop_type == 'third':
        # Loop the first third three times
        # This creates a 3.33-second pattern that repeats
        segment_length = len(genuine_signal) // 3
        segment = genuine_signal[:segment_length]
        looped_signal = np.tile(segment, 3)[:len(genuine_signal)]
        
    elif loop_type == 'full':
        # Just return the full signal (no looping)
        # This is the hardest to detect - requires temporal freshness checks
        looped_signal = genuine_signal.copy()
        
    else:
        raise ValueError(f"Unknown loop_type: {loop_type}. Use 'half', 'third', or 'full'.")
    
    # Add minimal noise to simulate real-world transmission
    # Real attackers might add slight noise to avoid exact matching
    noise = np.random.normal(0, 0.01, looped_signal.shape)
    
    return looped_signal + noise


def simulate_synthetic_attack(genuine_signal: np.ndarray, fs: int):
    """
    Generates a synthetic ECG signal with no heart rate variability (HRV).

    This simulates an attacker creating a signal from a single heartbeat
    template, repeating it at a perfectly constant rate. This lack of
    natural variation is a key indicator of a spoofed signal.

    Args:
        genuine_signal (np.ndarray): A real ECG signal to use as a base.
        fs (int): The sampling frequency of the signal.

    Returns:
        np.ndarray: The generated synthetic signal, or None if it fails.
    """
    # Find a prominent R-peak to use as a template for a single heartbeat
    peaks, _ = find_peaks(genuine_signal, prominence=0.5, distance=fs * 0.5)

    if len(peaks) < 2:
        print("Warning: Not enough peaks found to create a synthetic signal.")
        return None

    # Create a template of a single heartbeat (P-QRS-T complex)
    # We'll grab the signal from 0.2s before the first peak to 0.4s after.
    template_start = max(0, peaks[0] - int(0.2 * fs))
    template_end = min(len(genuine_signal), peaks[0] + int(0.4 * fs))
    heartbeat_template = genuine_signal[template_start:template_end]

    # Calculate the average interval between heartbeats from the real signal
    avg_rr_interval = int(np.mean(np.diff(peaks)))
    
    # Create a new signal by repeating the template at this fixed interval
    num_beats = len(genuine_signal) // avg_rr_interval
    synthetic_signal = np.zeros_like(genuine_signal)

    for i in range(num_beats):
        start_idx = i * avg_rr_interval
        end_idx = start_idx + len(heartbeat_template)
        if end_idx < len(synthetic_signal):
            synthetic_signal[start_idx:end_idx] = heartbeat_template
    
    return synthetic_signal
