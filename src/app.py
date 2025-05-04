# timeslot_lgbm_app.py
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb # Import LightGBM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Keep for splitting
from sklearn.metrics import mean_absolute_error # Use MAE for evaluation
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time # For adding delays in simulation

# --- Configuration ---
st.set_page_config(layout="wide")
st.title("📈 LightGBM for Predicting Returns from Overlapping Time Slots") # Changed title
st.markdown("""
This app demonstrates how a **LightGBM** model (a Gradient Boosting Machine) can predict future return counts
based on historical patterns of *hour-long selections initiated every 5 minutes*.
We'll use simulated data to illustrate the process. LightGBM is often easier to install and faster to train than Deep Learning models for this type of data.
""")

# --- Constants ---
N_DAYS = 14 # Simulate N days of data
MINS_PER_DAY = 24 * 60
SLOT_INTERVAL_MINS = 5
INITIATION_SLOTS_PER_DAY = MINS_PER_DAY // SLOT_INTERVAL_MINS # 288
SELECTION_DURATION_MINS = 60
SELECTION_DURATION_SLOTS = SELECTION_DURATION_MINS // SLOT_INTERVAL_MINS # 12

# --- Data Simulation (Keep as before) ---
# @st.cache_data # Cache the generated data
def simulate_data(n_days):
    """Generates synthetic initiation counts and return times."""
    start_time = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
    timestamps = pd.date_range(start=start_time, periods=n_days * INITIATION_SLOTS_PER_DAY, freq=f'{SLOT_INTERVAL_MINS}min')
    df = pd.DataFrame({'timestamp': timestamps})

    # Simulate initiation counts (peaks during day, low at night)
    time_of_day_factor = (np.sin(np.linspace(0, 2 * np.pi * n_days, len(df))) + 1.1) # Simple daily cycle
    random_noise = np.random.rand(len(df)) * 0.5
    weekend_factor = (df['timestamp'].dt.dayofweek >= 5).astype(int) * 0.3 # Higher on weekends
    base_rate = 3
    df['initiation_count'] = np.maximum(0, (base_rate * time_of_day_factor * (1 + weekend_factor) + random_noise)).astype(int)

    # Simulate return times for each initiation
    return_records = []
    st.write("Simulating returns (this might take a moment)...")
    progress_bar = st.progress(0)
    total_initiations = df['initiation_count'].sum()
    processed_initiations = 0

    # --- Simplified return simulation for speed ---
    initiation_times = df.loc[df.index.repeat(df['initiation_count'])]['timestamp'].reset_index(drop=True)
    if not initiation_times.empty:
        count = len(initiation_times)
        base_delay = np.random.uniform(90, 180, size=count) # Base delay in mins
        time_of_day_delay_factor = (1 + np.sin(initiation_times.dt.hour / 24 * 2 * np.pi)) * 30 # Longer delays if initiated late?
        noise_delay = np.random.normal(0, 30, size=count) # Noise
        delays_mins = base_delay + time_of_day_delay_factor + noise_delay
        delays_mins = np.maximum(5, delays_mins) # Ensure returns happen after initiation
        return_times = initiation_times + pd.to_timedelta(delays_mins, unit='m')
        returns_df = pd.DataFrame({'initiation_time': initiation_times, 'return_time': return_times})
        progress_bar.progress(1.0) # Mark as complete
    else:
         returns_df = pd.DataFrame(columns=['initiation_time', 'return_time'])
         progress_bar.progress(1.0) # Mark as complete

    progress_bar.empty() # Remove progress bar
    return df, returns_df
# --- End of simplified simulation ---


df_initiations, df_returns = simulate_data(N_DAYS)

st.header("1. Simulated Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Initiation Counts")
    st.markdown(f"Simulated **{len(df_initiations)}** ({N_DAYS} days) initiation time slots (every {SLOT_INTERVAL_MINS} mins).")
    st.dataframe(df_initiations.head())
    st.line_chart(df_initiations.set_index('timestamp')['initiation_count'])

with col2:
    st.subheader("Return Times")
    st.markdown(f"Simulated **{len(df_returns)}** individual returns.")
    st.dataframe(df_returns.head())
    # Plot distribution of return delays
    if not df_returns.empty:
        df_returns['delay_mins'] = (df_returns['return_time'] - df_returns['initiation_time']).dt.total_seconds() / 60
        fig_delay, ax_delay = plt.subplots()
        sns.histplot(df_returns['delay_mins'], bins=50, kde=True, ax=ax_delay)
        ax_delay.set_title('Distribution of Return Delays (Minutes)')
        ax_delay.set_xlabel('Delay (Minutes)')
        st.pyplot(fig_delay)
    else:
        st.write("No returns simulated.")


# --- Feature Engineering (Keep as before) ---
st.header("2. Feature Engineering")
st.markdown(f"""
We need features to feed into the LightGBM model.

**A. Raw Initiation Counts:** The number of selections started in each {SLOT_INTERVAL_MINS}-minute slot.

**B. Derived Active Counts:** Calculated from initiations in the previous {SELECTION_DURATION_MINS} mins.
`active_count(t) = sum(initiation_count(t - k*{SLOT_INTERVAL_MINS} mins))` for `k` from 0 to {SELECTION_DURATION_SLOTS-1}.

**C. Time Features:** Hour of day, day of week.
""")

# Calculate derived active counts
df_initiations['active_count'] = df_initiations['initiation_count'].rolling(window=SELECTION_DURATION_SLOTS, min_periods=1).sum()

col1a, col2a = st.columns(2)
with col1a:
    st.subheader("Raw Initiation Counts (Sample)")
    st.line_chart(df_initiations.set_index('timestamp')['initiation_count'].head(INITIATION_SLOTS_PER_DAY)) # Show first day
with col2a:
    st.subheader("Derived Active Counts (Sample)")
    st.line_chart(df_initiations.set_index('timestamp')['active_count'].head(INITIATION_SLOTS_PER_DAY))

st.markdown("""
**D. Target Variable (Future Return Counts):**
The goal is to predict the number of returns occurring within a future time window (e.g., the next hour). Calculated from `return_time` data.
""")

# --- Prepare Data for LightGBM ---
st.header("3. Preparing Data for LightGBM")

# Define lookback and prediction windows
lookback_hours = st.slider("Lookback Window (Hours)", 1, 12, 3)
prediction_hours = st.slider("Prediction Window (Hours)", 1, 6, 1)

lookback_slots = lookback_hours * 60 // SLOT_INTERVAL_MINS
prediction_slots = prediction_hours * 60 // SLOT_INTERVAL_MINS

st.markdown(f"""
We'll use a **lookback window** of {lookback_hours} hours ({lookback_slots} slots) of past `initiation_count`, `active_count`, and time features.
These features will be **flattened** into a single input vector for each prediction point.
The model predicts the total number of returns in the next **prediction window** of {prediction_hours} hour(s) ({prediction_slots} slots).
""")

# @st.cache_data # Cache data prep based on inputs
def create_feature_vectors(initiation_df, return_df, _lookback_slots, _prediction_slots):
    X_sequences, y = [], [] # Store sequences temporarily
    target_return_counts = []

    # Pre-calculate return counts per slot for efficiency
    if not return_df.empty:
        return_counts_per_slot = return_df.set_index('return_time').resample(f'{SLOT_INTERVAL_MINS}min').size()
        return_counts_per_slot = return_counts_per_slot.reindex(initiation_df['timestamp'], fill_value=0)
    else:
        return_counts_per_slot = pd.Series(0, index=initiation_df['timestamp'])

    initiation_features = initiation_df[['initiation_count', 'active_count']].values
    # Add time features
    timestamps = initiation_df['timestamp']
    hour_of_day = timestamps.dt.hour / 23.0 # Normalize
    day_of_week = timestamps.dt.dayofweek / 6.0 # Normalize

    # Combine all features for sequence creation
    all_features_sequential = np.hstack([
        initiation_features,
        hour_of_day.values.reshape(-1, 1),
        day_of_week.values.reshape(-1, 1)
    ])
    num_features_per_slot = all_features_sequential.shape[1]

    total_steps = len(initiation_df) - _lookback_slots - _prediction_slots + 1
    if total_steps <= 0:
        st.error("Not enough data for the selected lookback/prediction windows. Reduce window sizes or simulate more data.")
        return None, None, None

    st.write(f"Generating input/output sequences...")
    progress_bar_seq = st.progress(0)

    for i in range(total_steps):
        input_end_idx = i + _lookback_slots
        output_start_idx = input_end_idx
        output_end_idx = output_start_idx + _prediction_slots

        # Input sequence: features from i to input_end_idx
        input_seq = all_features_sequential[i:input_end_idx, :]
        X_sequences.append(input_seq) # Append the sequence

        # Target: sum of returns from output_start_idx to output_end_idx
        target_count = return_counts_per_slot.iloc[output_start_idx:output_end_idx].sum()
        y.append(target_count)

        target_return_counts.append({
            'prediction_point_time': timestamps.iloc[input_end_idx-1], # Time at end of lookback
            'target_window_start': timestamps.iloc[output_start_idx],
            'target_window_end': timestamps.iloc[output_end_idx-1] if output_end_idx > output_start_idx else timestamps.iloc[output_start_idx],
            'actual_returns': target_count
        })
        progress_bar_seq.progress(i / total_steps)

    progress_bar_seq.empty()

    # --- Reshape X from sequences to flattened vectors ---
    X_sequences_np = np.array(X_sequences)
    n_samples = X_sequences_np.shape[0]
    X_flattened = X_sequences_np.reshape(n_samples, -1) # Flatten lookback_slots * num_features_per_slot

    return X_flattened, np.array(y), pd.DataFrame(target_return_counts)

X, y, df_targets = create_feature_vectors(df_initiations, df_returns, lookback_slots, prediction_slots)

if X is not None:
    st.write(f"Created **{len(X)}** input/output samples.")
    st.write(f"Input shape (per sample): `{X.shape[1:]}` (Flattened Features)") # Updated shape description
    st.write(f"Output shape (per sample): `(1,)` (Predicted Return Count)")
    st.dataframe(df_targets.head())

    # --- Split and Scale Data (Keep as before, works for flattened data) ---
    st.subheader("Split and Scale Data")
    test_size = 0.2
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    df_targets_test = df_targets[split_idx:].reset_index(drop=True)

    st.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scaling - Still recommended for consistency, though less critical for LGBM
    feature_scaler = StandardScaler() # Single scaler for all flattened features
    target_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale targets
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))

    st.write("Features (and targets) scaled using StandardScaler (fitted on training data).")

    # --- LightGBM Model ---
    st.header("4. LightGBM Model Training")
    st.markdown("""
    We use LightGBM, a gradient boosting framework, for prediction.
    It's trained on the scaled, flattened feature vectors.
    """)

    # @st.cache_resource # Cache model training
    def train_lightgbm_model(_X_train_scaled, _y_train_scaled):
        """Builds and trains a LightGBM Regressor model."""
        with st.spinner("Training LightGBM model..."):
            model = lgb.LGBMRegressor(random_state=42, n_estimators=100) # Basic params
            # LGBM expects y as a 1D array
            model.fit(_X_train_scaled, _y_train_scaled.ravel())
        st.success("Model training finished.")
        return model

    if st.button("Train LightGBM Model"):
        st.session_state['model_trained'] = False # Reset flag
        model = train_lightgbm_model(X_train_scaled, y_train_scaled)
        st.session_state['model'] = model
        # st.session_state['history'] = None # No history object like Keras
        st.session_state['model_trained'] = True
        st.session_state['target_scaler'] = target_scaler # Store scaler
        st.session_state['df_targets_test'] = df_targets_test # Store test targets df
        st.session_state['X_test_scaled'] = X_test_scaled # Store scaled test features

    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        model = st.session_state['model']
        target_scaler = st.session_state['target_scaler']
        df_targets_test = st.session_state['df_targets_test']
        X_test_scaled = st.session_state['X_test_scaled']

        # No training history plot for basic LGBM fit

        # --- Prediction and Visualization ---
        st.header("5. Prediction on Test Data")
        with st.spinner("Making predictions on test data..."):
            y_pred_scaled = model.predict(X_test_scaled)
            # Inverse transform predictions and actual values to original scale
            # Reshape y_pred_scaled before inverse_transform
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
            # y_test_orig = target_scaler.inverse_transform(y_test_scaled) # Already stored unscaled y_test

        df_results = df_targets_test.copy()
        df_results['predicted_returns'] = np.maximum(0, y_pred.flatten()) # Ensure non-negative predictions
        df_results['actual_returns'] = y_test # Use the original unscaled y_test

        st.dataframe(df_results[['prediction_point_time', 'target_window_start', 'actual_returns', 'predicted_returns']].head())

        st.subheader("Actual vs. Predicted Returns (Test Set)")
        fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
        ax_pred.plot(df_results['target_window_start'], df_results['actual_returns'], label='Actual Returns', marker='.', linestyle='-')
        ax_pred.plot(df_results['target_window_start'], df_results['predicted_returns'], label='Predicted Returns', marker='.', linestyle='--')
        ax_pred.set_xlabel('Prediction Window Start Time')
        ax_pred.set_ylabel('Number of Returns')
        ax_pred.set_title('Actual vs. Predicted Return Counts (LightGBM)') # Updated title
        ax_pred.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig_pred)

        mae = mean_absolute_error(df_results['actual_returns'], df_results['predicted_returns'])
        st.metric("Mean Absolute Error (Test Set)", f"{mae:.2f} returns")
        st.markdown("*(Lower is better. Performance depends on simulation quality and model hyperparameters.)*")

    else:
        st.info("Click the 'Train LightGBM Model' button to train the model and see predictions.")


    st.header("Conclusion")
    st.markdown("""
    This app demonstrated the workflow for using **LightGBM** for time-series forecasting:
    1.  **Simulate/Load Data:** Obtain initiation counts and return times.
    2.  **Feature Engineering:** Create relevant input features (raw counts, derived counts, time features).
    3.  **Data Structuring:** Create lookback sequences and **flatten** them into feature vectors; calculate corresponding future return counts (target).
    4.  **Split & Scale:** Divide data chronologically and scale features/targets.
    5.  **Build & Train:** Train a LightGBM model on the historical data.
    6.  **Predict & Evaluate:** Use the trained model to forecast future returns and compare against actuals.

    LightGBM often provides a robust and efficient alternative to Deep Learning models for structured time-series tasks like this.
    """)
else:
     st.warning("Data sequence creation failed. Adjust parameters or check simulation.")