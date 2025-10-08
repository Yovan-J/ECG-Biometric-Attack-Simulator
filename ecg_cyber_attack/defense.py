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

    def train(self, data_path: Path, epochs=10, batch_size=32):
        """
        Loads data, generates spoofed samples, and trains the CNN model.
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")

        # 1. Load genuine data
        with np.load(data_path) as data:
            genuine_segments = data['segments']
            genuine_labels = data['labels']

        # 2. Generate spoofed data
        replay_segments = genuine_segments.copy()
        synthetic_segments = []
        for seg in genuine_segments:
            synth_seg = self._generate_synthetic_segment(seg, fs=700)
            if synth_seg is not None:
                synthetic_segments.append(synth_seg)
        
        synthetic_segments = np.array(synthetic_segments)
        spoofed_segments = np.concatenate([replay_segments, synthetic_segments])
        spoofed_labels = np.ones(len(spoofed_segments), dtype=int)

        # 3. Combine and prepare final dataset
        X = np.concatenate([genuine_segments, spoofed_segments])
        y = np.concatenate([genuine_labels, spoofed_labels])
        
        # Reshape X for CNN input: (num_samples, num_timesteps, num_channels)
        X = X.reshape((*X.shape, 1))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. Build and train the model
        self.model = self._build_model()
        print("Starting model training...")
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        
        # 5. Save the trained model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        self._is_trained = True
        print(f"Model trained and saved to {self.model_path}")

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
        Makes a prediction on a new ECG segment.
        """
        if self.model is None:
            self.load()
        
        # Ensure the segment has the correct shape for prediction: (1, timesteps, channels)
        segment = segment.reshape(1, *self.input_shape)
        
        probability = self.model.predict(segment)[0][0]
        prediction = 1 if probability > 0.5 else 0 # 1 for Spoofed, 0 for Genuine
        
        return prediction, probability

    @property
    def is_trained(self):
        return self._is_trained