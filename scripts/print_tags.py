"""
Step 2: Print Tags
==================
Displays unique target values and their counts.

Input: data/cleaned_data.csv
Output: Console output (label distribution)
"""

import pandas as pd
import sys

def print_tags(input_file='../data/cleaned_data.csv'):
    """Print unique target values and their counts."""
    try:
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)
        
        # Get unique target values and their counts
        target_counts = df['target'].value_counts()
        
        print(f"\n{'='*60}")
        print(f"Label Distribution ({len(target_counts)} unique labels)")
        print(f"{'='*60}")
        
        # Sort alphabetically and print
        for target, count in target_counts.sort_index().items():
            print(f"  {target}: {count}")
        
        print(f"{'='*60}")
        print(f"Total samples: {len(df)}")
        print(f"Unique labels: {len(target_counts)}")
        print(f"{'='*60}")
        
        return target_counts

    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        sys.exit(1)


if __name__ == "__main__":
    print_tags()
