import numpy as np
from scipy.signal import find_peaks

def extract_features(signal: np.ndarray, fs: int):
    """
    Extracts physiological features from an ECG signal, focusing on HRV.

    Args:
        signal (np.ndarray): The input ECG signal.
        fs (int): The sampling frequency of the signal.

    Returns:
        tuple: A tuple containing the mean RR interval and the HRV (SDNN),
               or (0, 0) if not enough heartbeats are detected.
    """
    # Find R-peaks, the most prominent peaks in the ECG signal
    # The prominence and distance parameters help filter out noise
    peaks, _ = find_peaks(signal, prominence=0.5, distance=fs * 0.5)

    # We need at least 3 peaks to calculate variability (2 RR intervals)
    if len(peaks) < 3:
        return 0, 0  # Not enough data to be considered a valid signal

    # Calculate RR intervals (the time between consecutive R-peaks)
    # Convert from samples to milliseconds
    rr_intervals = np.diff(peaks) * (1000 / fs)

    # Feature 1: Mean RR interval (average time between beats)
    mean_rr = np.mean(rr_intervals)

    # Feature 2: Standard deviation of RR intervals (SDNN)
    # This is a key measure of Heart Rate Variability (HRV)
    hrv_sdnn = np.std(rr_intervals)

    return mean_rr, hrv_sdnn