"""
Step 1: Clean Data
==================
Normalizes point names in the training data while preserving word boundaries.

Text column: BOM/whitespace stripped, then split on separators (_ - . / : #)
and camelCase boundaries into lowercase space-separated words.
  "Zone Temperature"  -> "zone temperature"
  "ZN-T"              -> "zn t"
  "CDK_DMPR_STATUS"   -> "cdk dmpr status"
  "zoneCO2Sp"         -> "zone co2 sp"

Target column: only stripped of BOM/whitespace (labels stay camelCase).

Input:  data/train_all.csv
Output: data/cleaned_data.csv
"""

import pandas as pd
import re
import sys


def normalize_text(s: str) -> str:
    """Normalize a raw point name into lowercase space-separated words."""
    s = s.replace('\ufeff', '').strip()
    # Separators become spaces
    s = re.sub(r'[_\-./:#]+', ' ', s)
    # camelCase boundary: lowercase/digit followed by uppercase
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', s)
    # Acronym boundary: "CDKDamper" -> "CDK Damper" (keeps "CO2" intact)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()


def clean_data(input_file='data/train_all.csv', output_file='data/cleaned_data.csv'):
    """Normalize text column, strip target column."""
    try:
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        df['text'] = df['text'].astype(str).map(normalize_text)
        df['target'] = df['target'].astype(str).str.replace('\ufeff', '').str.strip()

        # Drop rows that normalized to nothing
        before = len(df)
        df = df[(df['text'].str.len() > 0) & (df['target'].str.len() > 0)]
        if before - len(df) > 0:
            print(f"Removed {before - len(df)} rows with empty text/target")

        df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

        return df

    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        sys.exit(1)


if __name__ == "__main__":
    clean_data()
