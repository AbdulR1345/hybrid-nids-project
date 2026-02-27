# evaluate_lstm.py - Evaluate LSTM Autoencoder reconstruction errors
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # suppresses round-off warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # suppresses ALL TensorFlow info/warning logs (including CPU guard)
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

def load_data(split):
    """Load scaled X and binary label for a split"""
    X = np.load(os.path.join(PROCESSED_DIR, f'{split}_X_scaled.npy'))
    yb = pd.read_csv(os.path.join(PROCESSED_DIR, f'{split}_yb.csv'))['Binary_Label'].values
    return X, yb

def main():
    # Load model
    model_path = os.path.join(MODELS_DIR, 'lstm_autoencoder.h5')
    if not os.path.exists(model_path):
        print("Model not found! Train first.")
        return
    
    model = tf.keras.models.load_model(
    model_path,
    compile=False,                     # skip compiling
    custom_objects={'mse': 'mse'}      # force legacy MSE handling
)
    print("Loaded LSTM Autoencoder")

    # Load threshold
    threshold_path = os.path.join(RESULTS_DIR, 'anomaly_threshold.txt')
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())
    print(f"Using anomaly threshold: {threshold:.6f}")

    # Load validation and test data
    splits = ['train', 'val', 'test']
    results = {}

    plt.figure(figsize=(14, 6))

    for split in splits:
        print(f"\nEvaluating {split} set...")
        X, yb = load_data(split)
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Predict (reconstruct)
        recon = model.predict(X_reshaped, batch_size=256, verbose=0)
        
        # MSE per sample
        mse = np.mean(np.power(X_reshaped - recon, 2), axis=(1,2))
        
        # Split benign vs attack
        benign_mse = mse[yb == 0]
        attack_mse = mse[yb == 1]
        
        results[split] = {
            'benign_mean': benign_mse.mean(),
            'benign_std': benign_mse.std(),
            'attack_mean': attack_mse.mean(),
            'attack_std': attack_mse.std(),
            'benign_count': len(benign_mse),
            'attack_count': len(attack_mse)
        }
        
        print(f"{split.capitalize()} - Benign: {len(benign_mse)} samples, Mean MSE: {benign_mse.mean():.6f}")
        print(f"{split.capitalize()} - Attack: {len(attack_mse)} samples, Mean MSE: {attack_mse.mean():.6f}")
        
        # Plot histogram
        sns.histplot(benign_mse, bins=100, color='green', label='Benign', alpha=0.6, kde=True)
        sns.histplot(attack_mse, bins=100, color='red', label='Attack', alpha=0.6, kde=True)
        plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.4f})')
        plt.title(f'Reconstruction Error Distribution - {split.capitalize()} Set')
        plt.xlabel('Mean Squared Error (Reconstruction Error)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        
        hist_path = os.path.join(RESULTS_DIR, f'mse_distribution_{split}.png')
        plt.savefig(hist_path)
        plt.clf()  # clear figure for next split
        print(f"Saved histogram: {hist_path}")

    # Summary table
    summary = pd.DataFrame(results).T
    print("\nReconstruction Error Summary:")
    print(summary)

    summary.to_csv(os.path.join(RESULTS_DIR, 'lstm_recon_error_summary.csv'))
    print("Summary saved to results/lstm_recon_error_summary.csv")

    print("\nEvaluation completed. Check results/ folder for histograms and summary.")

if __name__ == "__main__":
    main()