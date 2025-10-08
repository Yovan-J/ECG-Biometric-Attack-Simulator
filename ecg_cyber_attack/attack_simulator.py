import numpy as np
from scipy.signal import find_peaks

def simulate_replay_attack(genuine_signal: np.ndarray):
    """
    Simulates a replay attack by returning a copy of the genuine signal.

    In a real-world scenario, this would be a previously captured signal segment
    that the attacker is "replaying" to the sensor.

    Args:
        genuine_signal (np.ndarray): The legitimate ECG signal.

    Returns:
        np.ndarray: The replayed ECG signal.
    """
    return genuine_signal.copy()

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