# data_prep.py - Clean and prepare CIC-IDS2017/2018 CSVs (no notebook!)

import pandas as pd
import numpy as np
import os
from config import RAW_DIR, PROCESSED_DIR, RANDOM_STATE, SAMPLE_FRAC

def get_label_column(df):
    """Find the label column (handles spaces/case differences)"""
    for col in df.columns:
        if 'label' in col.lower().strip():
            return col
    raise ValueError("No label column found! Columns: " + str(df.columns[:10]))

def clean_single_file(input_path, output_path, sample_frac=SAMPLE_FRAC):
    """Clean one CSV: drop junk, fix inf/NaN, binary label, sample if needed"""
    print(f"\nProcessing: {os.path.basename(input_path)}")
    
    # Load with low memory + latin1 for weird chars
    df = pd.read_csv(input_path, low_memory=False, encoding='latin1')
    print(f"Original shape: {df.shape}")
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Drop useless identifier columns
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port', 'Dst Port']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Find and rename label column to 'Label'
    label_col = get_label_column(df)
    if label_col != 'Label':
        df = df.rename(columns={label_col: 'Label'})
    
    # Replace inf/-inf → NaN → 0 (common safe fill for flow features)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Create binary label (BENIGN=0, anything else=1)
    df['Binary_Label'] = df['Label'].apply(
        lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1
    )
    
    # Optional sampling to speed up (remove later for full run)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE)
        print(f"Sampled to {len(df)} rows ({sample_frac*100}%)")
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned: {output_path}")
    print(f"Final shape: {df.shape}")
    print("Label distribution:\n", df['Label'].value_counts())
    print("Binary distribution:\n", df['Binary_Label'].value_counts(normalize=True) * 100)
    
    return df

def main():
    """Process all raw CSVs"""
    csv_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith('.csv')]
    if not csv_files:
        print("No CSV files found in raw folder!")
        return
    
    for csv_name in csv_files:
        input_path = os.path.join(RAW_DIR, csv_name)
        base_name = os.path.splitext(csv_name)[0]
        output_name = f"{base_name}_cleaned.csv"
        output_path = os.path.join(PROCESSED_DIR, output_name)
        
        clean_single_file(input_path, output_path)

if __name__ == "__main__":
    main()