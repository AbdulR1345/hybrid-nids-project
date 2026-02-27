# merge_data.py - Merge all cleaned CSVs, normalize labels, select common columns

import pandas as pd
import numpy as np
import os
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_STATE, SAMPLE_FRAC

def normalize_label(label):
    """Clean label names: strip, upper, fix common typos/variations"""
    if pd.isna(label):
        return "UNKNOWN"
    lbl = str(label).strip().upper()
    # Fix variations
    if "BENIGN" in lbl or "NORMAL" in lbl:
        return "BENIGN"
    if "BRUTE FORCE" in lbl or "BRUTE" in lbl:
        return "BRUTE_FORCE"
    if "XSS" in lbl:
        return "WEB_XSS"
    if "SQL" in lbl or "INJECTION" in lbl:
        return "WEB_SQL_INJECTION"
    if "GOLDENEYE" in lbl:
        return "DOS_GOLDENEYE"
    if "SLOWLORIS" in lbl or "SLOW" in lbl:
        return "DOS_SLOWLORIS"
    if "HULK" in lbl:
        return "DOS_HULK"
    if "SLOWHTTPTEST" in lbl:
        return "DOS_SLOWHTTPTEST"
    if "BOT" in lbl:
        return "BOTNET"
    if "PORTSCAN" in lbl:
        return "PORT_SCAN"
    if "INFILTRATION" in lbl:
        return "INFILTRATION"
    # Keep others as-is but cleaned
    return lbl.replace("  ", " ").replace("-", "_").replace(" ", "_")

def main():
    cleaned_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_cleaned.csv')]
    if not cleaned_files:
        print("No cleaned files found!")
        return
    
    print(f"Found {len(cleaned_files)} cleaned files. Merging...")
    
    dfs = []
    all_columns = None
    
    for file_name in cleaned_files:
        path = os.path.join(PROCESSED_DIR, file_name)
        print(f"Loading {file_name}")
        df = pd.read_csv(path, low_memory=False)
        
        # Normalize Label
        df['Label'] = df['Label'].apply(normalize_label)
        
        # Get columns
        cols = set(df.columns)
        if all_columns is None:
            all_columns = cols
        else:
            all_columns &= cols  # intersection
        
        dfs.append(df)
    
    # Merge
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Merged raw shape: {merged.shape}")
    
    # Keep only common columns + Label + Binary_Label
    common_cols = list(all_columns)
    if 'Label' not in common_cols:
        common_cols.append('Label')
    if 'Binary_Label' not in common_cols:
        common_cols.append('Binary_Label')
    
    merged = merged[common_cols]
    print(f"After common columns: {merged.shape}")
    
    # Final label stats
    print("\nFinal Label distribution:")
    print(merged['Label'].value_counts())
    print("\nBinary distribution (%):")
    print(merged['Binary_Label'].value_counts(normalize=True) * 100)
    
    # Optional: further sample if too big for your laptop
    if SAMPLE_FRAC < 1.0:
        merged = merged.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
        print(f"Further sampled to {len(merged)} rows")
    
    # Save
    output_path = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    merged.to_csv(output_path, index=False)
    print(f"\nSaved merged dataset: {output_path}")
    print("Shape:", merged.shape)
    
    # Save label mapping for later use
    label_map = merged['Label'].unique()
    pd.Series(label_map).to_csv(os.path.join(RESULTS_DIR, "label_classes.csv"), index=False, header=['Label'])
    print("Label classes saved to results/label_classes.csv")

if __name__ == "__main__":
    main()