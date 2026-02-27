# lstm_anomaly_detector.py - LSTM Autoencoder for anomaly detection (Stage 1)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # suppresses round-off warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # suppresses ALL TensorFlow info/warning logs (including CPU guard)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
import joblib
import os
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

# Create models dir if missing
os.makedirs(MODELS_DIR, exist_ok=True)

def load_benign_train():
    """Load only BENIGN from train split"""
    X_train = np.load(os.path.join(PROCESSED_DIR, 'train_X_scaled.npy'))
    yb_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_yb.csv'))['Binary_Label'].values
    
    # Only BENIGN (0)
    benign_mask = (yb_train == 0)
    X_benign = X_train[benign_mask]
    
    print(f"Benign training samples: {X_benign.shape[0]} ({X_benign.shape[0]/len(yb_train)*100:.2f}% of train)")
    return X_benign

def build_lstm_autoencoder(input_dim, timesteps=1):
    """Build LSTM Autoencoder (simple 1D sequence)"""
    model = Sequential([
        # Encoder
        LSTM(64, activation='relu', input_shape=(timesteps, input_dim), return_sequences=False),
        # Latent space
        Dense(32, activation='relu'),
        # Decoder
        RepeatVector(timesteps),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_dim))
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.summary()
    return model

def main():
    # Load benign data
    X_benign = load_benign_train()
    if X_benign.shape[0] == 0:
        print("No benign samples found! Check train_yb.csv")
        return
    
    # Reshape for LSTM: (samples, timesteps=1, features)
    X_benign = X_benign.reshape((X_benign.shape[0], 1, X_benign.shape[1]))
    
    # Build & train
    input_dim = X_benign.shape[2]
    model = build_lstm_autoencoder(input_dim)
    
    print("\nTraining LSTM Autoencoder on benign traffic...")
    history = model.fit(
        X_benign, X_benign,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
        shuffle=True
    )
    
    # Save model & history
    model.save(os.path.join(MODELS_DIR, 'lstm_autoencoder.h5'))
    pd.DataFrame(history.history).to_csv(os.path.join(RESULTS_DIR, 'lstm_training_history.csv'), index=False)
    print("Model saved: models/lstm_autoencoder.h5")
    
    # Compute reconstruction error on benign train (for threshold later)
    recon = model.predict(X_benign)
    mse = np.mean(np.power(X_benign - recon, 2), axis=(1,2))
    
    threshold = np.percentile(mse, 95)  # 95th percentile as initial threshold
    print(f"Benign reconstruction MSE - mean: {mse.mean():.6f}, std: {mse.std():.6f}")
    print(f"Suggested anomaly threshold (95%): {threshold:.6f}")
    
    # Save threshold
    with open(os.path.join(RESULTS_DIR, 'anomaly_threshold.txt'), 'w') as f:
        f.write(str(threshold))
    print("Threshold saved to results/anomaly_threshold.txt")
    
    print("\nLSTM Autoencoder training completed. Next: Evaluate on full train/val/test.")

if __name__ == "__main__":
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    main()