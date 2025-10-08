import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample

# Import the modules we built
from ecg_cyber_attack.data_loader import load_ecg_data
from ecg_cyber_attack.feature_extractor import extract_features
from ecg_cyber_attack.attack_simulator import simulate_replay_attack, simulate_synthetic_attack
from ecg_cyber_attack.defense import DefenseCNNModel # Use the new CNN model class

# --- Page Configuration ---
st.set_page_config(
    page_title="ECG CNN Attack Simulator",
    layout="wide"
)

# --- Constants ---
MODEL_PATH = Path("models/defense_cnn_model.h5") # Updated model path
DATA_PATH = Path("ecg_segments.npz")
TARGET_FS = 700 # The sampling frequency our model expects (from WESAD)

# --- Helper Functions ---
def plot_signals(signals: dict, fs: int):
    # This function remains largely the same
    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle('ECG Signal Comparison', fontsize=16)
    
    # Use the first signal to determine the time axis
    time = np.arange(len(next(iter(signals.values())))) / fs
    
    colors = {'Genuine': 'blue', 'Replay Attack': 'orange', 'Synthetic Attack': 'red'}
    
    for i, (name, signal) in enumerate(signals.items()):
        ax = axes[i]
        ax.plot(time, signal, color=colors.get(name, 'black'), label=name)
        ax.set_title(name)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        if i == len(signals) // 2:
            ax.set_ylabel('Normalized Amplitude')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def train_model_callback(model_instance, data_path):
    # Simplified training callback for the new model
    st.info("Training the 1D-CNN model. This is a resource-intensive process and may take a significant amount of time. Please monitor the terminal for epoch progress.")
    with st.spinner("Training model..."):
        try:
            model_instance.train(data_path, epochs=15) # Train for more epochs
            st.success("Model trained successfully and saved!")
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during training: {e}")

# --- Main Application ---
st.title("1D-CNN ECG Attack Simulator")
st.markdown("An advanced demonstrator using a Convolutional Neural Network to detect spoofing attacks directly from raw ECG time-series data.")

# Initialize the defense model
model = DefenseCNNModel(MODEL_PATH)

# --- Sidebar for Controls ---
st.sidebar.header("Simulation Settings")
record_name = st.sidebar.text_input("Enter MIT-BIH Record Name", value='100')
run_simulation = st.sidebar.button("Run Simulation")

# --- Model Training Section ---
if not model.is_trained:
    st.warning("The 1D-CNN defense model has not been trained. This is a one-time process.")
    st.info(f"The training process will load the pre-processed WESAD data from '{DATA_PATH}' and train the neural network. This may take 15-30 minutes depending on your hardware.")
    if st.button("Train Defense Model"):
        if DATA_PATH.exists():
            train_model_callback(model, DATA_PATH)
        else:
            st.error(f"Data file not found! Please run 'preprocess_wesad.py' first to generate '{DATA_PATH}'.")
else:
    st.sidebar.success("1D-CNN model is trained and ready.")
    with st.expander("Model Retraining (Optional)"):
        st.write("Retrain the model if you have updated the preprocessing script or want to adjust training parameters.")
        if st.button("Retrain Model"):
            train_model_callback(model, DATA_PATH)

# --- Simulation Execution ---
if run_simulation:
    if not model.is_trained:
        st.error("Cannot run simulation: The model must be trained first.")
    else:
        with st.spinner(f"Loading and processing record '{record_name}'..."):
            # 1. Load Data from MIT-BIH (at 360Hz)
            original_signal, original_fs = load_ecg_data(record_name, start=5000, end=8600) # Load 10s of data (3600 samples)
            
            if original_signal is None:
                st.error(f"Failed to load record '{record_name}'. Please check the name.")
            else:
                # 2. CRITICAL: Resample the signal to the model's expected frequency (700Hz)
                num_samples_target = int(len(original_signal) * TARGET_FS / original_fs)
                resampled_signal = resample(original_signal, num_samples_target)
                
                # 3. Simulate Attacks on the resampled signal
                replay_signal = simulate_replay_attack(resampled_signal)
                synthetic_signal = simulate_synthetic_attack(resampled_signal, TARGET_FS)
                
                st.header("Signal Visualization (Resampled to 700Hz)")
                signals_to_plot = {
                    'Genuine': resampled_signal,
                    'Replay Attack': replay_signal,
                }
                if synthetic_signal is not None:
                    signals_to_plot['Synthetic Attack'] = synthetic_signal
                
                fig = plot_signals(signals_to_plot, TARGET_FS)
                st.pyplot(fig)
                
                st.header("Defense System Analysis")
                st.write("The 1D-CNN model analyzes the raw signal morphology to predict its authenticity.")
                
                # 4. Analyze and Predict for each signal
                for name, signal in signals_to_plot.items():
                    st.subheader(f"Verdict for: {name}")
                    
                    # Prediction is now done on the raw signal
                    prediction, probability = model.predict(signal)
                    
                    # For display purposes, we can still calculate statistical features
                    stat_features = extract_features(signal, TARGET_FS)
                    
                    verdict = "Spoofed" if prediction == 1 else "Genuine"
                    # Confidence is the model's certainty in its chosen class
                    confidence = probability if verdict == "Spoofed" else 1 - probability
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean RR Interval (ms)", f"{stat_features[0]:.2f}")
                        st.metric("HRV (SDNN)", f"{stat_features[1]:.2f}")
                    
                    with col2:
                        if verdict == "Genuine":
                            st.success(f"Verdict: {verdict} (Confidence: {confidence:.2f})")
                        else:
                            st.error(f"Verdict: {verdict} (Confidence: {confidence:.2f})")