# app.py - Polished Cyber-Security Style Streamlit GUI for Hybrid NIDS
# Fixes: SyntaxError, string-to-float, long loading, live capture, UI upgrades

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix
from scapy.all import sniff, IP
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from config import MODELS_DIR, RESULTS_DIR

# Page config for cyber look
st.set_page_config(page_title="Hybrid NIDS Dashboard", layout="wide", page_icon="🛡️")

# Cyber-security style CSS (dark theme, red/green alerts, network feel)
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stButton > button { background-color: #238636; color: white; border: none; border-radius: 6px; padding: 10px 20px; font-weight: bold; }
    .stButton > button:hover { background-color: #2ea043; }
    .stError { background-color: #440000; border-left: 5px solid #f85149; padding: 15px; border-radius: 8px; color: #f85149; }
    .stSuccess { background-color: #1f3a1f; border-left: 5px solid #56d364; padding: 15px; border-radius: 8px; color: #56d364; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
    .highlight-anomaly { background-color: #440000 !important; color: #f85149 !important; }
    .stSidebar { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff; }
    hr { border-top: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

# Professional cyber header
st.markdown("""
    <h1 style='text-align: center; color: #58a6ff; margin-bottom: 5px;'>🛡️ HYBRID NIDS DASHBOARD</h1>
    <p style='text-align: center; color: #8b949e; font-size: 18px;'>
        Real-Time Intrusion Detection | Enhanced AUF Framework for IoT & Home Networks
    </p>
    <hr style='border-top: 2px solid #30363d; margin: 20px 0;'>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    lstm = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'lstm_autoencoder.h5'), compile=False)
    knn = joblib.load(os.path.join(MODELS_DIR, 'knn_discriminator.pkl'))
    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_classifier.pkl'))
    with open(os.path.join(RESULTS_DIR, 'anomaly_threshold.txt'), 'r') as f:
        threshold = float(f.read().strip())
    return lstm, knn, rf, threshold

lstm_model, knn, rf, threshold = load_models()

# Inference function
def infer_single(sample, lstm_model, knn, rf, threshold):
    sample_reshaped = sample.reshape((1, 1, sample.shape[0]))
    recon = lstm_model.predict(sample_reshaped, verbose=0)
    mse = np.mean(np.power(sample_reshaped - recon, 2))
    
    if mse <= threshold:
        return "BENIGN", mse
    
    knn_feat = np.array([[mse, 1]])
    is_unseen = knn.predict(knn_feat)[0]
    
    if is_unseen == 1:
        return "ANOMALY - UNSEEN", mse
    
    attack_type = rf.predict(sample.reshape(1, -1))[0]
    return f"ANOMALY - SEEN: {attack_type}", mse

# Tabs
tab1, tab2 = st.tabs(["CSV Analysis", "Live Capture"])

# Tab 1: CSV Analysis
with tab1:
    st.subheader("Upload Network Flow CSV")
    uploaded_file = st.file_uploader("Drag & drop or browse CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview (scrollable)")
        st.dataframe(df, height=350)  # Scrollable, shows many rows
        
        if st.button("Analyze Traffic"):
            with st.spinner("Analyzing flows..."):
                # Select only numeric columns (fix string error)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_df = df[numeric_cols]
                
                results = []
                mses = []
                progress_bar = st.progress(0)
                for i, (_, row) in enumerate(numeric_df.iterrows()):
                    sample = row.values.astype(np.float32)
                    prediction, mse = infer_single(sample, lstm_model, knn, rf, threshold)
                    results.append(prediction)
                    mses.append(mse)
                    progress_bar.progress((i + 1) / len(numeric_df))
                
                df['Prediction'] = results
                df['MSE'] = mses
            
            st.success("Analysis Complete!")
            st.subheader("Results (anomalies highlighted in red)")
            styled_df = df.style.applymap(lambda x: 'background-color: #440000; color: #f85149' if 'ANOMALY' in x else '', subset=['Prediction'])
            st.dataframe(styled_df, height=500)  # Tall scrollable table
            
            anomalies = df[df['Prediction'].str.contains("ANOMALY", na=False)]
            if not anomalies.empty:
                st.error(f"🚨 {len(anomalies)} ANOMALIES DETECTED!")
                st.dataframe(anomalies)
            
            # Interactive MSE chart
            st.subheader("MSE Distribution")
            fig = px.bar(df, x=df.index, y='MSE', color='Prediction',
                         color_discrete_map={"BENIGN": "#56d364", "ANOMALY - SEEN": "#ffa500", "ANOMALY - UNSEEN": "#f85149"},
                         title="Anomaly Scores (Higher MSE = More Suspicious)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix if 'Label' present
            if 'Label' in df.columns:
                st.subheader("Performance Metrics")
                acc = accuracy_score(df['Label'], df['Prediction'])
                st.metric("Accuracy", f"{acc:.2%}")
                cm = confusion_matrix(df['Label'], df['Prediction'])
                fig_cm, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig_cm)
            
            # Download buttons
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "nids_results.csv", mime="text/csv")
            
            if st.button("Export PDF Report"):
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                c.drawString(100, 750, "Hybrid NIDS Report")
                c.drawString(100, 730, f"Flows: {len(df)} | Anomalies: {len(anomalies)}")
                c.save()
                buffer.seek(0)
                st.download_button("Download PDF", buffer, "nids_report.pdf", mime="application/pdf")

# Tab 2: Live Capture
with tab2:
    st.subheader("Live Packet Capture")
    st.warning("**Important**: Run PowerShell as Administrator. Install Npcap from https://npcap.com if not already done.")
    
    packet_count = st.number_input("Packets to capture", 5, 100, 20)
    iface = st.text_input("Interface (e.g., Wi-Fi)", "Wi-Fi")
    
    if st.button("Start Capture"):
        with st.spinner("Capturing live packets..."):
            try:
                packets = sniff(iface=iface, count=packet_count, filter="ip")
                features = []
                for pkt in packets:
                    if pkt.haslayer(IP):
                        feat = [pkt[IP].len, pkt[IP].ttl, pkt.time - packets[0].time]
                        features.append(feat)
                
                df_live = pd.DataFrame(features, columns=['Length', 'TTL', 'Time'])
                
                results = []
                mses = []
                for _, row in df_live.iterrows():
                    sample = row.values.astype(np.float32)
                    prediction, mse = infer_single(sample, lstm_model, knn, rf, threshold)
                    results.append(prediction)
                    mses.append(mse)
                
                df_live['Prediction'] = results
                df_live['MSE'] = mses
                
                st.subheader("Live Results (anomalies highlighted)")
                st.dataframe(
                    df_live.style.applymap(
                        lambda x: 'background-color: #440000; color: #f85149' if 'ANOMALY' in x else '',
                        subset=['Prediction']
                    ),
                    height=400
                )
                
                anomalies = df_live[df_live['Prediction'].str.contains("ANOMALY", na=False)]
                if not anomalies.empty:
                    st.error(f"🚨 LIVE ALERT: {len(anomalies)} anomalies detected!")
                    st.dataframe(anomalies)
                else:
                    st.success("No anomalies in captured traffic.")
            
            except Exception as e:
                st.error(f"Capture failed: {e}\nTip: Install Npcap, run as admin, check interface name.")

# general help text (always visible)
st.info("Upload a CSV or switch to Live Capture tab to start.")
