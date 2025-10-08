import numpy as np
import wfdb

def load_ecg_data(record_name: str, start: int, end: int, channel: int = 0):
    """
    Loads a segment of an ECG signal from the PhysioNet MIT-BIH database.

    Args:
        record_name (str): The name of the record to load (e.g., '100').
        start (int): The starting sample number.
        end (int): The ending sample number.
        channel (int): The signal channel to use. Defaults to 0.

    Returns:
        tuple: A tuple containing the signal (np.ndarray) and the
               sampling frequency (int), or (None, None) if an error occurs.
    """
    try:
        # CORRECTED: Use pn_dir to specify the PhysioNet database name.
        # This tells the library to download the data if it's not found locally.
        record = wfdb.rdrecord(
            record_name,
            pn_dir='mitdb',  # Specify the MIT-BIH Database on PhysioNet
            sampfrom=start,
            sampto=end,
            channels=[channel]
        )
        
        signal = record.p_signal[:, 0]
        fs = record.fs
        
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        return signal, fs
        
    except Exception as e:
        print(f"Error loading record '{record_name}': {e}")
        return None, None