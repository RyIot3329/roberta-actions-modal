"""
Step 4: Convert to JSONL
========================
Converts cleaned CSV to JSONL format with train/val/test splits.

Input:  data/cleaned_data.csv
Output: data/train.jsonl, data/validation.jsonl, data/test.jsonl,
        data/label_mapping.json, data/dataset_summary.json
"""

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import sys

def convert_to_jsonl(
    input_csv='data/cleaned_data.csv',
    output_dir='data',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    """
    Convert CSV to JSONL format with data cleaning and train/val/test split.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Loading {input_csv}...")
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: '{input_csv}' not found.")
        sys.exit(1)
    
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Strip whitespace from text and target columns
    df['text'] = df['text'].str.strip()
    df['target'] = df['target'].str.strip()
    
    # Remove any rows with empty text or target
    original_len = len(df)
    df = df.dropna(subset=['text', 'target'])
    df = df[df['text'].str.len() > 0]
    df = df[df['target'].str.len() > 0]
    if original_len - len(df) > 0:
        print(f"Removed {original_len - len(df)} rows with empty values")
    
    # Class distribution
    class_counts = Counter(df['target'])
    print(f"Unique classes: {len(class_counts)}")
    
    # Create label to ID mapping
    unique_labels = sorted(df['target'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Split data
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=df['target']
    )
    
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=train_val_df['target']
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Write JSONL files
    def write_jsonl(data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in data.iterrows():
                record = {
                    'text': row['text'],
                    'label': row['target'],
                    'label_id': label2id[row['target']]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  Saved: {filepath}")
    
    print(f"\nWriting JSONL files...")
    write_jsonl(train_df, os.path.join(output_dir, 'train.jsonl'))
    write_jsonl(val_df, os.path.join(output_dir, 'validation.jsonl'))
    write_jsonl(test_df, os.path.join(output_dir, 'test.jsonl'))
    
    # Save label mappings
    label_mapping_path = os.path.join(output_dir, 'label_mapping.json')
    with open(label_mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'label2id': label2id,
            'id2label': {str(k): v for k, v in id2label.items()},
            'num_labels': len(unique_labels)
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {label_mapping_path}")
    
    # Create summary file
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(df),
            'num_classes': len(unique_labels),
            'train_samples': len(train_df),
            'validation_samples': len(val_df),
            'test_samples': len(test_df),
            'class_distribution': dict(class_counts.most_common()),
            'text_length_stats': {
                'min': int(df['text'].str.len().min()),
                'max': int(df['text'].str.len().max()),
                'mean': float(df['text'].str.len().mean()),
                'median': float(df['text'].str.len().median())
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")
    
    print(f"\nConversion complete!")
    
    return train_df, val_df, test_df, label2id


if __name__ == "__main__":
    convert_to_jsonl()
