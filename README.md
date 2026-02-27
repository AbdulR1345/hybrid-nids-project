# Intelligent Hybrid Network Intrusion Detection System

**Final Year Project – Abdul & kabil raj**  
**Guide: Dr.S.Usha**  
**Department of Computer Science and Engineering**

## Project Overview
Enhanced hybrid NIDS using LSTM for anomaly detection, adaptive KNN for seen/unseen discrimination, and Random Forest for multi-class classification. Built for IoT & home network security in Tamil Nadu. Includes real-time Streamlit GUI.

## Features (Improvements over base AUF paper)
- LSTM autoencoder (instead of XGBoost) – better temporal pattern capture
- Adaptive KNN discriminator with GAN augmentation
- Random Forest classifier
- Streamlit GUI with CSV analysis, live packet capture (Scapy), alerts & PDF export

## How to Run

1. Clone or download the repo
2. Open terminal in project folder
3. Create & activate conda env (optional but recommended)
4. Install dependencies
5. Run the GUI dashboard
6. Upload `data/sample_data.csv` to test

## Folder Structure

├── data/
│   └── sample_data.csv       # small test file
├── src/
│   ├── app.py                # main GUI
│   ├── config.py
│   ├── data_prep.py
│   ├── merge_data.py
│   ├── preprocess.py
│   ├── lstm_anomaly_detector.py
│   ├── knn_discriminator.py
│   ├── rf_classifier.py
│   └── demo.py               # CLI test
├── requirements.txt
└── README.md


**Note**: Full raw datasets and trained models are not included (too large). Use sample CSV for demo. Contact for full data.
Email:rahamanrahi13@gmail.com
Phone:9629690594

Made in Trichy , Tamil Nadu – 2026

