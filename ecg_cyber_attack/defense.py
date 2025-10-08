import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

class DefenseCNNModel:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self._is_trained = self.model_path.exists()
        # Define the expected input shape for the model
        self.input_shape = (7000, 1) # 7000 samples (10s at 700Hz), 1 channel

    def _build_model(self):
        """
        Builds the 1D Convolutional Neural Network architecture using Keras.
        """
        model = tf.keras.models.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.input_shape),

            # Convolutional Block 1
            tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            # Convolutional Block 2
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            # Convolutional Block 3
            tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            # Flatten and Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid') # Output layer for binary classification
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def _generate_synthetic_segment(self, genuine_segment: np.ndarray, fs: int):
        """Generates a single synthetic ECG segment with zero HRV."""
        peaks, _ = find_peaks(genuine_segment, prominence=0.3, distance=fs * 0.4)
        if len(peaks) < 2:
            return None # Not enough peaks to create a template

        avg_rr_interval = int(np.mean(np.diff(peaks)))
        template_start = max(0, peaks[0] - int(0.2 * fs))
        template_end = min(len(genuine_segment), peaks[0] + int(0.4 * fs))
        heartbeat_template = genuine_segment[template_start:template_end]

        synthetic_segment = np.zeros_like(genuine_segment)
        num_beats = len(synthetic_segment) // avg_rr_interval

        for i in range(num_beats):
            start_idx = i * avg_rr_interval
            end_idx = start_idx + len(heartbeat_template)
            if end_idx < len(synthetic_segment):
                synthetic_segment[start_idx:end_idx] = heartbeat_template
        return synthetic_segment

    def _generate_replay_segment(self, genuine_segment: np.ndarray):
        """
        Generates a replay attack by looping half of the segment.
        This creates detectable periodic patterns.
        """
        # Loop the first half twice
        half_length = len(genuine_segment) // 2
        segment = genuine_segment[:half_length]
        looped = np.tile(segment, 2)[:len(genuine_segment)]
        
        # Add minimal noise
        noise = np.random.normal(0, 0.01, looped.shape)
        return looped + noise

    def train(self, data_path: Path, epochs=10, batch_size=32):
        """
        Loads data, generates spoofed samples (both replay and synthetic), 
        and trains the CNN model.
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")

        # 1. Load genuine data
        with np.load(data_path) as data:
            genuine_segments = data['segments']
            genuine_labels = data['labels']

        print(f"Loaded {len(genuine_segments)} genuine segments.")

        # 2. Generate REPLAY attack segments (looped patterns)
        print("Generating replay attack segments...")
        replay_segments = []
        for seg in genuine_segments:
            replay_seg = self._generate_replay_segment(seg)
            replay_segments.append(replay_seg)
        
        replay_segments = np.array(replay_segments)
        print(f"Generated {len(replay_segments)} replay segments.")

        # 3. Generate SYNTHETIC attack segments (zero HRV)
        print("Generating synthetic attack segments...")
        synthetic_segments = []
        for seg in genuine_segments:
            synth_seg = self._generate_synthetic_segment(seg, fs=700)
            if synth_seg is not None:
                synthetic_segments.append(synth_seg)
        
        synthetic_segments = np.array(synthetic_segments)
        print(f"Generated {len(synthetic_segments)} synthetic segments.")

        # 4. Combine all spoofed segments
        spoofed_segments = np.concatenate([replay_segments, synthetic_segments])
        spoofed_labels = np.ones(len(spoofed_segments), dtype=int)

        # 5. Combine genuine and spoofed data
        X = np.concatenate([genuine_segments, spoofed_segments])
        y = np.concatenate([genuine_labels, spoofed_labels])
        
        print(f"Total dataset: {len(X)} segments ({len(genuine_segments)} genuine, {len(spoofed_segments)} spoofed)")
        
        # Reshape X for CNN input: (num_samples, num_timesteps, num_channels)
        X = X.reshape((*X.shape, 1))

        # 6. Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # 7. Build and train the model
        self.model = self._build_model()
        print("\nStarting model training...")
        print("=" * 60)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=3, 
                    restore_best_weights=True,
                    verbose=1
                )
            ],
            verbose=1
        )
        
        # 8. Save the trained model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        self._is_trained = True
        
        print("=" * 60)
        print(f"✓ Model trained and saved to {self.model_path}")
        print(f"✓ Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    def load(self):
        """
        Loads the pre-trained Keras model from the file.
        """
        if not self._is_trained:
            raise FileNotFoundError("Model file not found. Please train the model first.")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict(self, segment: np.ndarray):
        """
        Makes a prediction on a new ECG segment using the CNN.
        
        Returns:
            tuple: (prediction, probability)
                - prediction: 0 for genuine, 1 for spoofed
                - probability: confidence score (0-1)
        """
        if self.model is None:
            self.load()
        
        # Ensure the segment has the correct shape for prediction: (1, timesteps, channels)
        segment = segment.reshape(1, *self.input_shape)
        
        probability = self.model.predict(segment, verbose=0)[0][0]
        prediction = 1 if probability > 0.5 else 0 # 1 for Spoofed, 0 for Genuine
        
        return prediction, probability

    def detect_replay_pattern(self, segment: np.ndarray, fs: int = 700):
        """
        Detects if a segment has repeating patterns indicative of a replay attack.
        Uses autocorrelation to identify periodic repetition.
        
        This is a complementary method to the CNN prediction, specifically
        designed to catch looped replay attacks.
        
        Args:
            segment (np.ndarray): The ECG segment to analyze
            fs (int): Sampling frequency
        
        Returns:
            tuple: (is_replay, correlation_score)
                - is_replay: Boolean, True if replay pattern detected
                - correlation_score: Float, the correlation value (0-1)
        """
        # Compute autocorrelation
        autocorr = np.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        
        # Normalize by the zero-lag autocorrelation
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return False, 0.0
        
        # Check for high correlation at half-segment interval (5 seconds)
        # This would indicate the signal is looping every 5 seconds
        half_segment = len(segment) // 2
        tolerance = int(0.1 * half_segment)  # 10% tolerance window
        
        if len(autocorr) > half_segment + tolerance:
            # Extract the correlation values around the expected peak
            peak_region = autocorr[half_segment - tolerance : half_segment + tolerance]
            max_correlation = np.max(peak_region)
            
            # Also check at 1/3 segment (for 3-way loops)
            third_segment = len(segment) // 3
            if len(autocorr) > third_segment + tolerance:
                third_peak_region = autocorr[third_segment - tolerance : third_segment + tolerance]
                third_correlation = np.max(third_peak_region)
                max_correlation = max(max_correlation, third_correlation)
            
            # Threshold: correlation > 0.95 indicates likely replay
            # Genuine ECG should have correlation < 0.90 at these lags due to natural HRV
            if max_correlation > 0.95:
                return True, float(max_correlation)
        
        return False, 0.0

    @property
    def is_trained(self):
        return self._is_trained
