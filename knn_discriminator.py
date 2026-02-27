# knn_discriminator.py - Adaptive KNN for seen vs unseen classification
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # suppresses round-off warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # suppresses ALL TensorFlow info/warning logs (including CPU guard)
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE
from imblearn.over_sampling import SMOTE

def load_split(split):
    X = np.load(os.path.join(PROCESSED_DIR, f'{split}_X_scaled.npy'))
    y_multi = pd.read_csv(os.path.join(PROCESSED_DIR, f'{split}_ym.csv'))['Label'].values
    return X, y_multi

def main():
    # Load LSTM model for reconstruction error feature
    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, 'lstm_autoencoder.h5'),
        compile=False,
        custom_objects={'mse': 'mse'}
    )
    
    # Load threshold
    with open(os.path.join(RESULTS_DIR, 'anomaly_threshold.txt'), 'r') as f:
        threshold = float(f.read().strip())
    
    # Function to get reconstruction error + is_anomaly flag
    def get_features(X):
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        recon = model.predict(X_reshaped, batch_size=256, verbose=0)
        mse = np.mean(np.power(X_reshaped - recon, 2), axis=(1,2))
        is_anomaly = (mse > threshold).astype(int)
        features = np.column_stack((mse, is_anomaly))
        return features, mse
    
    # Load train data (only anomalies for discriminator training)
    X_train, y_train = load_split('train')
    X_train_feat, mse_train = get_features(X_train)
    
    anomaly_mask = (mse_train > threshold)
    X_anom_train = X_train_feat[anomaly_mask]
    y_anom_train = y_train[anomaly_mask]
    
    print(f"Anomalies in train: {len(X_anom_train)}")
    
    # Map multi-class labels to seen/unseen (example: hold out some classes as unseen)
    seen_classes = ['BENIGN', 'DOS_HULK', 'BOTNET', 'PORT_SCAN', 'DOS_SLOWLORIS']
    unseen_classes = ['DOS_GOLDENEYE', 'BRUTE_FORCE', 'WEB_XSS', 'INFILTRATION', 'WEB_SQL_INJECTION']
    
    # For training: treat seen as 0, unseen as 1 (binary discrimination)
    y_disc_train = np.where(np.isin(y_anom_train, unseen_classes), 1, 0)
    
    # SMOTE for imbalance (if needed)
    if len(np.unique(y_disc_train)) > 1:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_disc_train, y_disc_train = smote.fit_resample(X_anom_train, y_disc_train)
    
    # Grid search for best k
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
    grid.fit(X_disc_train, y_disc_train)
    
    print("Best k:", grid.best_params_)
    print("Best CV F1:", grid.best_score_)
    
    # Save model
    joblib.dump(grid.best_estimator_, os.path.join(MODELS_DIR, 'knn_discriminator.pkl'))
    print("Discriminator saved: models/knn_discriminator.pkl")
    
    # Evaluate on val/test (similar logic)
    for split in ['val', 'test']:
        X, y = load_split(split)
        X_feat, mse = get_features(X)
        anomaly_mask = (mse > threshold)
        X_anom = X_feat[anomaly_mask]
        y_anom = y[anomaly_mask]
        
        y_disc_true = np.where(np.isin(y_anom, unseen_classes), 1, 0)
        y_disc_pred = grid.predict(X_anom)
        
        print(f"\n{split.capitalize()} Discriminator Results:")
        print(classification_report(y_disc_true, y_disc_pred, target_names=['Seen', 'Unseen']))
    
    print("Discriminator evaluation completed.")

if __name__ == "__main__":
    main()