# rf_classifier.py - Stage 3: Multi-class classification on seen attacks
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # suppresses round-off warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # suppresses ALL TensorFlow info/warning logs (including CPU guard)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

def load_split(split):
    X = np.load(os.path.join(PROCESSED_DIR, f'{split}_X_scaled.npy'))
    y_multi = pd.read_csv(os.path.join(PROCESSED_DIR, f'{split}_ym.csv'))['Label'].values
    return X, y_multi

def main():
    print("Loading LSTM model for anomaly filtering...")
    from tensorflow import keras
    model = keras.models.load_model(
        os.path.join(MODELS_DIR, 'lstm_autoencoder.h5'),
        compile=False,
        custom_objects={'mse': 'mse'}
    )
    
    with open(os.path.join(RESULTS_DIR, 'anomaly_threshold.txt'), 'r') as f:
        threshold = float(f.read().strip())
    
    def get_anomalies(X):
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        recon = model.predict(X_reshaped, batch_size=256, verbose=0)
        mse = np.mean(np.power(X_reshaped - recon, 2), axis=(1,2))
        anomaly_mask = mse > threshold
        return X[anomaly_mask], anomaly_mask
    
    print("Training Random Forest on seen attacks...")
    X_train, y_train = load_split('train')
    X_anom_train, _ = get_anomalies(X_train)
    y_anom_train = y_train[_]  # only anomalies
    
    # Filter only seen classes
    seen_classes = ['BENIGN', 'DOS_HULK', 'BOTNET', 'PORT_SCAN', 'DOS_SLOWLORIS', 'DOS_GOLDENEYE']
    seen_mask = np.isin(y_anom_train, seen_classes)
    X_seen = X_anom_train[seen_mask]
    y_seen = y_anom_train[seen_mask]
    
    print(f"Training on {len(X_seen)} seen attack samples")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_seen, y_seen)
    
    joblib.dump(rf, os.path.join(MODELS_DIR, 'rf_classifier.pkl'))
    print("RF classifier saved: models/rf_classifier.pkl")
    
    # Evaluate on val & test
    for split in ['val', 'test']:
        X, y = load_split(split)
        X_anom, mask = get_anomalies(X)
        y_anom = y[mask]
        
        seen_mask = np.isin(y_anom, seen_classes)
        X_seen_test = X_anom[seen_mask]
        y_seen_test = y_anom[seen_mask]
        
        if len(X_seen_test) == 0:
            print(f"No seen attacks in {split} after filtering.")
            continue
        
        y_pred = rf.predict(X_seen_test)
        
        print(f"\n{split.capitalize()} Multi-class Results:")
        print(classification_report(y_seen_test, y_pred))
    
    print("Multi-class evaluation completed.")

if __name__ == "__main__":
    main()