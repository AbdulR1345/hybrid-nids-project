# preprocess.py - Train/val/test split, scaling, save splits

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_STATE, TEST_SIZE

def main():
    data_path = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    if not os.path.exists(data_path):
        print("Merged dataset not found!")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    
    # Define features (exclude Label, Binary_Label, non-numeric)
    label_cols = ['Label', 'Binary_Label']
    feature_cols = [col for col in df.columns if col not in label_cols and df[col].dtype in [np.float64, np.int64]]
    print(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")
    
    X = df[feature_cols].values
    y_binary = df['Binary_Label'].values
    y_multi = df['Label'].values
    
    # Stratified split on binary label (to preserve imbalance ratio)
    X_temp, X_test, yb_temp, yb_test, ym_temp, ym_test = train_test_split(
        X, y_binary, y_multi, test_size=TEST_SIZE, stratify=y_binary, random_state=RANDOM_STATE
    )
    
    X_train, X_val, yb_train, yb_val, ym_train, ym_val = train_test_split(
        X_temp, yb_temp, ym_temp, test_size=TEST_SIZE / (1 - TEST_SIZE), stratify=yb_temp, random_state=RANDOM_STATE
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Scale features (fit on train only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later (inference)
    joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler.pkl'))
    print("Scaler saved to results/scaler.pkl")
    
    # Save splits as CSV (easy for report) + numpy for models
    def save_split(X, yb, ym, prefix):
        pd.DataFrame(X, columns=feature_cols).to_csv(
            os.path.join(PROCESSED_DIR, f"{prefix}_X.csv"), index=False
        )
        pd.Series(yb).to_csv(os.path.join(PROCESSED_DIR, f"{prefix}_yb.csv"), index=False, header=['Binary_Label'])
        pd.Series(ym).to_csv(os.path.join(PROCESSED_DIR, f"{prefix}_ym.csv"), index=False, header=['Label'])
        np.save(os.path.join(PROCESSED_DIR, f"{prefix}_X_scaled.npy"), X)
    
    save_split(X_train_scaled, yb_train, ym_train, 'train')
    save_split(X_val_scaled, yb_val, ym_val, 'val')
    save_split(X_test_scaled, yb_test, ym_test, 'test')
    
    print("Splits saved in processed/: train_X.csv, train_yb.csv, etc.")
    
    # Quick imbalance stats
    print("\nTrain binary distribution (%):")
    print(pd.Series(yb_train).value_counts(normalize=True) * 100)
    
    print("\nReady for modeling. Next: LSTM Autoencoder for anomaly detection.")

if __name__ == "__main__":
    main()