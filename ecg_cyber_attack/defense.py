import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import pearsonr # Correct import for direct correlation
from sklearn.model_selection import train_test_split

class DefenseCNNModel:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self._is_trained = self.model_path.exists()
        self.input_shape = (7000, 1)

    def _build_model(self):
        """
        Builds the 1D Convolutional Neural Network architecture using Keras.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _generate_synthetic_segment(self, genuine_segment: np.ndarray, fs: int):
        """Generates a single synthetic ECG segment with zero HRV."""
        peaks, _ = find_peaks(genuine_segment, prominence=0.3, distance=fs * 0.4)
        if len(peaks) < 2: return None
        avg_rr_interval = int(np.mean(np.diff(peaks)))
        template_start = max(0, peaks[0] - int(0.2 * fs))
        template_end = min(len(genuine_segment), peaks[0] - int(0.2 * fs) + int(0.4 * fs))
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
        """Generates a replay attack by looping half of the segment."""
        half_length = len(genuine_segment) // 2
        segment = genuine_segment[:half_length]
        looped = np.tile(segment, 2)[:len(genuine_segment)]
        noise = np.random.normal(0, 0.01, looped.shape)
        return looped + noise

    def train(self, data_path: Path, epochs=10, batch_size=32):
        """Loads data, generates spoofed samples, and trains the CNN model."""
        if not data_path.exists(): raise FileNotFoundError(f"Data file not found at {data_path}")
        with np.load(data_path) as data:
            genuine_segments = data['segments']
            genuine_labels = data['labels']
        print(f"Loaded {len(genuine_segments)} genuine segments.")
        print("Generating replay attack segments...")
        replay_segments = np.array([self._generate_replay_segment(seg) for seg in genuine_segments])
        print(f"Generated {len(replay_segments)} replay segments.")
        print("Generating synthetic attack segments...")
        synthetic_segments = np.array([s for s in (self._generate_synthetic_segment(seg, fs=700) for seg in genuine_segments) if s is not None])
        print(f"Generated {len(synthetic_segments)} synthetic segments.")
        spoofed_segments = np.concatenate([replay_segments, synthetic_segments])
        spoofed_labels = np.ones(len(spoofed_segments), dtype=int)
        X = np.concatenate([genuine_segments, spoofed_segments])
        y = np.concatenate([genuine_labels, spoofed_labels])
        print(f"Total dataset: {len(X)} segments ({len(genuine_segments)} genuine, {len(spoofed_segments)} spoofed)")
        X = X.reshape((*X.shape, 1))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Training set: {len(X_train)} samples\nValidation set: {len(X_val)} samples")
        self.model = self._build_model()
        print("\nStarting model training...\n" + "=" * 60)
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1)], verbose=1)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        self._is_trained = True
        print("=" * 60 + f"\n✓ Model trained and saved to {self.model_path}")
        print(f"✓ Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    def load(self):
        """Loads the pre-trained Keras model from the file."""
        if not self._is_trained: raise FileNotFoundError("Model file not found.")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict(self, segment: np.ndarray):
        """Makes a prediction on a new ECG segment using the CNN."""
        if self.model is None: self.load()
        segment = segment.reshape(1, *self.input_shape)
        probability = self.model.predict(segment, verbose=0)[0][0]
        prediction = 1 if probability > 0.5 else 0
        return prediction, probability

    def detect_replay_pattern(self, segment: np.ndarray, fs: int = 700):
        """
        Detects if a segment has repeating patterns using direct correlation.
        This is a more robust method than the previous autocorrelation approach.
        
        Returns:
            tuple: (is_replay, correlation_score)
        """
        # Test for a 2-part loop (like the 'half' replay attack)
        half_len = len(segment) // 2
        part1_h = segment[:half_len]
        part2_h = segment[half_len:half_len*2]
        # Use pearsonr which returns (correlation, p-value)
        corr_h, _ = pearsonr(part1_h, part2_h)
        
        # Test for a 3-part loop (like the 'third' replay attack)
        third_len = len(segment) // 3
        part1_t = segment[:third_len]
        part2_t = segment[third_len:third_len*2]
        corr_t, _ = pearsonr(part1_t, part2_t)

        # The final score is the highest correlation found from the tests
        max_correlation = max(corr_h, corr_t)

        # A high correlation (>0.95) is a strong indicator of a loop
        if max_correlation > 0.95:
            return True, float(max_correlation)
        
        return False, float(max_correlation)

    @property
    def is_trained(self):
        return self._is_trained
