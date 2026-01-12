"""
Step 1: Clean Data
==================
Removes spaces from all string columns in the training data.

Input:  data/train_all.csv
Output: data/cleaned_data.csv
"""

import pandas as pd
import sys

def clean_data(input_file='data/train_all.csv', output_file='data/cleaned_data.csv'):
    """Remove spaces from all string columns."""
    try:
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Remove spaces from all string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(' ', '')

        # Save the cleaned data
        df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")
        
        return df

    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        sys.exit(1)


if __name__ == "__main__":
    clean_data()
