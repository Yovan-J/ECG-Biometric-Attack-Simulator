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
st.title("1D-CNN ECG Attack Simulator with Replay Detection")
st.markdown("An advanced demonstrator using a Convolutional Neural Network to detect spoofing attacks (replay and synthetic) directly from raw ECG time-series data.")

# Initialize the defense model
model = DefenseCNNModel(MODEL_PATH)

# --- Sidebar for Controls ---
st.sidebar.header("Simulation Settings")
record_name = st.sidebar.text_input("Enter MIT-BIH Record Name", value='100')

# Add replay attack type selector
replay_type = st.sidebar.selectbox(
    "Replay Attack Type",
    ["half", "third", "full"],
    index=0,
    help="Half: Loop first 5s twice | Third: Loop first 3.3s three times | Full: No looping (hardest to detect)"
)

run_simulation = st.sidebar.button("Run Simulation")

# --- Model Training Section ---
if not model.is_trained:
    st.warning("The 1D-CNN defense model has not been trained. This is a one-time process.")
    st.info(f"The training process will load the pre-processed WESAD data from '{DATA_PATH}' and train the neural network on both replay and synthetic attacks. This may take 15-30 minutes depending on your hardware.")
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
                replay_signal = simulate_replay_attack(resampled_signal, fs=TARGET_FS, loop_type=replay_type)
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
                st.write("The system uses two complementary detection methods:")
                st.write("1. **1D-CNN Model**: Analyzes raw signal morphology")
                st.write("2. **Pattern Detection**: Identifies periodic repetition patterns")
                
                # 4. Analyze and Predict for each signal
                for name, signal in signals_to_plot.items():
                    st.subheader(f"Analysis: {name}")
                    
                    # CNN Prediction
                    prediction, probability = model.predict(signal)
                    
                    # Replay Pattern Detection
                    is_replay, replay_score = model.detect_replay_pattern(signal, TARGET_FS)
                    
                    # Statistical features for display
                    stat_features = extract_features(signal, TARGET_FS)
                    
                    # Determine verdicts
                    cnn_verdict = "Spoofed" if prediction == 1 else "Genuine"
                    cnn_confidence = probability if cnn_verdict == "Spoofed" else 1 - probability
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean RR Interval (ms)", f"{stat_features[0]:.2f}")
                        st.metric("HRV (SDNN)", f"{stat_features[1]:.2f}")
                    
                    with col2:
                        st.metric("CNN Verdict", cnn_verdict)
                        st.metric("CNN Confidence", f"{cnn_confidence:.2f}")
                    
                    with col3:
                        st.metric("Replay Pattern", "Detected" if is_replay else "Not Detected")
                        if is_replay:
                            st.metric("Correlation Score", f"{replay_score:.3f}")
                    
                    # Combined Final Verdict
                    st.markdown("---")
                    
                    if cnn_verdict == "Spoofed" and is_replay:
                        st.error("🚨 **ATTACK DETECTED** - Both CNN and Pattern Detection flagged this signal!")
                        st.write("**Attack Type:** Likely a looped replay attack")
                    elif cnn_verdict == "Spoofed":
                        st.error("⚠️ **ATTACK DETECTED** - CNN flagged this signal as spoofed")
                        st.write("**Attack Type:** Possibly synthetic or subtle replay")
                    elif is_replay:
                        st.error("⚠️ **ATTACK DETECTED** - Periodic pattern detected")
                        st.write("**Attack Type:** Looped replay attack")
                    else:
                        st.success("✓ **Signal appears GENUINE** - Passed all checks")
                    
                    st.markdown("---")
                
                # Summary table
                st.header("Detection Summary")
                summary_data = []
                for name, signal in signals_to_plot.items():
                    pred, prob = model.predict(signal)
                    is_rep, rep_score = model.detect_replay_pattern(signal, TARGET_FS)
                    
                    summary_data.append({
                        "Signal": name,
                        "CNN Verdict": "Spoofed" if pred == 1 else "Genuine",
                        "CNN Score": f"{prob:.3f}",
                        "Replay Detected": "Yes" if is_rep else "No",
                        "Replay Score": f"{rep_score:.3f}" if is_rep else "N/A"
                    })
                
                st.table(summary_data)
                
                # Interpretation guide
                with st.expander("📖 Understanding the Results"):
                    st.markdown("""
                    ### CNN Model
                    - **Score > 0.5**: Signal classified as spoofed
                    - **Score < 0.5**: Signal classified as genuine
                    - Trained to detect both replay and synthetic attacks
                    
                    ### Pattern Detection
                    - Looks for repeating segments using autocorrelation
                    - **Correlation > 0.95**: Strong evidence of looped replay
                    - Genuine signals typically have correlation < 0.90 due to natural heart rate variability
                    
                    ### Attack Types
                    - **Replay (Half)**: First 5 seconds looped twice
                    - **Replay (Third)**: First 3.3 seconds looped three times
                    - **Replay (Full)**: Complete signal replayed (hardest to detect)
                    - **Synthetic**: Artificial signal with zero HRV
                    
                    ### Expected Results
                    - **Genuine**: Both methods should pass
                    - **Replay**: Pattern detection should catch looped types, CNN may catch all types
                    - **Synthetic**: CNN should detect (zero HRV is very unnatural)
                    """)