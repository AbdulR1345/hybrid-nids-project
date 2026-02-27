# eda.py - Exploratory Data Analysis on merged CIC-IDS dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_STATE

def main():
    # Load merged data
    merged_path = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    if not os.path.exists(merged_path):
        print("Merged dataset not found! Run merge_data.py first.")
        return
    
    print("Loading merged dataset...")
    df = pd.read_csv(merged_path, low_memory=False)
    print(f"Shape: {df.shape}")
    
    # Basic info
    print("\nDataFrame Info:")
    df.info()
    
    print("\nMissing values (should be 0 after cleaning):")
    print(df.isnull().sum().sum())
    
    print("\nDescriptive statistics (numeric features):")
    print(df.describe().T[['mean', 'std', 'min', 'max']])
    
    # Label distribution
    label_counts = df['Label'].value_counts()
    binary_counts = df['Binary_Label'].value_counts(normalize=True) * 100
    
    print("\nMulti-class Label distribution:")
    print(label_counts)
    
    print("\nBinary distribution (%):")
    print(binary_counts)
    
    # === Plots ===
    plt.style.use('seaborn-v0_8')
    
    # 1. Bar plot - Attack types
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df['Label'], order=label_counts.index, palette='viridis')
    plt.title('Multi-class Label Distribution (Merged Dataset)')
    plt.xlabel('Count')
    plt.ylabel('Attack Type')
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, 'label_distribution_bar.png')
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved bar plot: {bar_path}")
    
    # 2. Pie chart - Binary (benign vs attack)
    plt.figure(figsize=(8, 8))
    plt.pie(binary_counts, labels=['BENIGN', 'ATTACK'], autopct='%1.1f%%',
            colors=['#66c2a5', '#fc8d62'], shadow=True, startangle=90)
    plt.title('Binary Classification: Benign vs Malicious Traffic')
    pie_path = os.path.join(RESULTS_DIR, 'binary_pie_chart.png')
    plt.savefig(pie_path)
    plt.close()
    print(f"Saved pie chart: {pie_path}")
    
    # 3. Correlation heatmap (top 15 features by variance or importance)
    # Select numeric columns only (exclude Label, Binary_Label)
    num_cols = df.select_dtypes(include=np.number).columns.drop(['Binary_Label'], errors='ignore')
    
    if len(num_cols) > 0:
        corr = df[num_cols].corr().abs()
        # Get top correlated features
        top_features = corr.unstack().sort_values(ascending=False).drop_duplicates()
        top_features = top_features[top_features < 1].head(30).index.get_level_values(0).unique()[:15]
        
        if len(top_features) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Heatmap - Top Features')
            plt.tight_layout()
            heatmap_path = os.path.join(RESULTS_DIR, 'correlation_heatmap.png')
            plt.savefig(heatmap_path)
            plt.close()
            print(f"Saved correlation heatmap: {heatmap_path}")
        else:
            print("Not enough numeric features for correlation heatmap.")
    else:
        print("No numeric features found for correlation.")
    
    print("\nEDA completed. Check results/ folder for plots.")

if __name__ == "__main__":
    main()