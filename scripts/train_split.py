"""
Step 3: Train/Validation Split
==============================
Splits cleaned data into train and validation CSV files.

Input:  data/cleaned_data.csv
Output: data/train.csv, data/valid.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def train_split(
    input_file='data/cleaned_data.csv',
    train_output='data/train.csv',
    valid_output='data/valid.csv',
    test_size=0.2,
    random_state=42
):
    """Split data into train and validation sets."""
    try:
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Total samples: {len(df)}")
        print(f"Unique targets: {df['target'].nunique()}")
        
        # Split with stratification
        train_df, valid_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['target']
        )
        
        # Save splits
        train_df.to_csv(train_output, index=False)
        valid_df.to_csv(valid_output, index=False)
        
        print(f"\nSplit complete:")
        print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%) -> {train_output}")
        print(f"  Valid: {len(valid_df)} samples ({len(valid_df)/len(df)*100:.1f}%) -> {valid_output}")
        
        return train_df, valid_df

    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        sys.exit(1)


if __name__ == "__main__":
    train_split()
