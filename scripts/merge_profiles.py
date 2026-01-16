#!/usr/bin/env python3
"""
Merge Profile Data into Training Dataset
=========================================

This script merges extracted device profile data with existing training data,
handling deduplication and creating proper train/val/test splits.

Usage:
    python merge_profile_data.py \
        --existing-train data/train.jsonl \
        --existing-val data/validation.jsonl \
        --profile-data extracted_points.jsonl \
        --output-dir data/
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], filepath: Path):
    """Save data to a JSONL file."""
    with open(filepath, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def normalize_text(text: str) -> str:
    """Normalize text for deduplication."""
    return text.lower().strip()


def main():
    parser = argparse.ArgumentParser(
        description='Merge profile data with existing training data'
    )
    parser.add_argument(
        '--existing-train', '-t',
        type=Path,
        required=True,
        help='Path to existing train.jsonl'
    )
    parser.add_argument(
        '--existing-val', '-v',
        type=Path,
        required=True,
        help='Path to existing validation.jsonl'
    )
    parser.add_argument(
        '--profile-data', '-p',
        type=Path,
        required=True,
        help='Path to extracted profile data (JSONL format)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data/'),
        help='Output directory for merged datasets'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Ratio of new profile data to add to validation set (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load existing data
    print("Loading existing data...")
    train_data = load_jsonl(args.existing_train)
    val_data = load_jsonl(args.existing_val)
    profile_data = load_jsonl(args.profile_data)
    
    print(f"  Existing train: {len(train_data)} samples")
    print(f"  Existing val:   {len(val_data)} samples")
    print(f"  Profile data:   {len(profile_data)} samples")
    
    # Build set of existing texts (normalized) for deduplication
    existing_texts = set()
    for entry in train_data + val_data:
        existing_texts.add(normalize_text(entry['text']))
    
    # Filter profile data to remove duplicates
    new_entries = []
    duplicates = 0
    for entry in profile_data:
        normalized = normalize_text(entry['text'])
        if normalized not in existing_texts:
            existing_texts.add(normalized)
            new_entries.append(entry)
        else:
            duplicates += 1
    
    print(f"\nDeduplication:")
    print(f"  Duplicates removed: {duplicates}")
    print(f"  New unique entries: {len(new_entries)}")
    
    if not new_entries:
        print("No new unique entries to add. Exiting.")
        return
    
    # Shuffle and split new entries
    random.shuffle(new_entries)
    val_count = int(len(new_entries) * args.val_ratio)
    
    new_val = new_entries[:val_count]
    new_train = new_entries[val_count:]
    
    # Merge with existing data
    merged_train = train_data + new_train
    merged_val = val_data + new_val
    
    # Shuffle merged data
    random.shuffle(merged_train)
    random.shuffle(merged_val)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged datasets
    train_out = args.output_dir / 'train.jsonl'
    val_out = args.output_dir / 'validation.jsonl'
    
    save_jsonl(merged_train, train_out)
    save_jsonl(merged_val, val_out)
    
    print(f"\nMerged datasets saved:")
    print(f"  Train: {len(merged_train)} samples -> {train_out}")
    print(f"  Val:   {len(merged_val)} samples -> {val_out}")
    
    # Print label distribution of new data
    label_counts = defaultdict(int)
    for entry in new_entries:
        label_counts[entry['label']] += 1
    
    print(f"\nLabel distribution of new profile data:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {label}: {count}")
    if len(label_counts) > 15:
        print(f"  ... and {len(label_counts) - 15} more labels")


if __name__ == '__main__':
    main()