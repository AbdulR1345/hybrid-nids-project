# demo.py - Simple CLI inference demo for full hybrid pipeline
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # suppresses round-off warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # suppresses ALL TensorFlow info/warning logs (including CPU guard)
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR

def load_models():
    lstm_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, 'lstm_autoencoder.h5'),
        compile=False,
        custom_objects={'mse': 'mse'}
    )
    knn = joblib.load(os.path.join(MODELS_DIR, 'knn_discriminator.pkl'))
    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_classifier.pkl'))
    
    with open(os.path.join(RESULTS_DIR, 'anomaly_threshold.txt'), 'r') as f:
        threshold = float(f.read().strip())
    
    return lstm_model, knn, rf, threshold

def infer_single_sample(sample, lstm_model, knn, rf, threshold):
    sample = sample.reshape((1, 1, sample.shape[0]))
    recon = lstm_model.predict(sample, verbose=0)
    mse = np.mean(np.power(sample - recon, 2))
    
    if mse <= threshold:
        return "BENIGN"
    
    knn_feat = np.array([[mse, 1]])  # mse + is_anomaly
    is_unseen = knn.predict(knn_feat)[0]
    
    if is_unseen == 1:
        return "ANOMALY - UNSEEN ATTACK"
    
    # Seen: classify type
    sample = sample.reshape(1, -1)
    attack_type = rf.predict(sample)[0]
    return f"ANOMALY - SEEN ATTACK: {attack_type}"

def main():
    lstm_model, knn, rf, threshold = load_models()
    
    # Load test samples (10 examples for demo)
    X_test = np.load(os.path.join(PROCESSED_DIR, 'test_X_scaled.npy'))[:10]
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_ym.csv'))['Label'].values[:10]
    
    print("Running demo on 10 test samples...\n")
    for i, sample in enumerate(X_test):
        prediction = infer_single_sample(sample, lstm_model, knn, rf, threshold)
        actual = y_test[i]
        print(f"Sample {i+1}: Prediction = {prediction} | Actual = {actual}")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TF logs
    main()