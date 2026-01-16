#!/usr/bin/env python3
"""
Extract Device Resource Names from YAML Profiles for Fine-tuning
================================================================

This script scans device profile YAML files and extracts the device resource names,
preparing them for inclusion in training data. It handles the :present-value suffix
and can optionally attempt to auto-map to existing labels.

Usage:
    python extract_profiles.py --profiles-dir ./profiles --output extracted_points.csv
    python extract_profiles.py --profiles-dir ./profiles --label-mapping data/label_mapping.json --output matched_points.csv
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import csv

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Install with: pip install pyyaml")
    exit(1)


def load_yaml_file(filepath: Path) -> dict:
    """Load a YAML file and return its contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_device_resources(profile: dict) -> list[dict]:
    """
    Extract device resource names from a profile.
    
    Returns a list of dicts with:
    - raw_name: The original name from the profile
    - clean_name: Name without :present-value suffix
    - profile_name: The profile this came from
    """
    resources = []
    profile_name = profile.get('name', 'unknown')
    
    for resource in profile.get('deviceResources', []):
        raw_name = resource.get('name', '')
        if not raw_name:
            continue
            
        # Remove :present-value suffix (case-insensitive)
        clean_name = re.sub(r':present-value$', '', raw_name, flags=re.IGNORECASE)
        
        resources.append({
            'raw_name': raw_name,
            'clean_name': clean_name,
            'profile_name': profile_name,
            'value_type': resource.get('properties', {}).get('valueType', 'unknown'),
        })
    
    return resources


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Convert to lowercase, replace separators with spaces, collapse whitespace
    normalized = text.lower()
    normalized = re.sub(r'[-_:\s]+', ' ', normalized)
    normalized = normalized.strip()
    return normalized


def create_label_matcher(label2id: dict) -> dict:
    """
    Create a dictionary for fuzzy matching labels.
    Maps normalized label names to their original label names.
    """
    matcher = {}
    for label in label2id.keys():
        # Create normalized version
        normalized = normalize_text(label)
        matcher[normalized] = label
        
        # Also add with camelCase split
        # e.g., "dischargeTemp" -> "discharge temp"
        split_camel = re.sub(r'([a-z])([A-Z])', r'\1 \2', label).lower()
        matcher[split_camel] = label
        
    return matcher


def attempt_auto_match(clean_name: str, label_matcher: dict, label2id: dict) -> tuple[str, float]:
    """
    Attempt to automatically match a point name to a label.
    
    Returns (matched_label, confidence_score) or (None, 0.0) if no match.
    Confidence: 1.0 = exact match, 0.5-0.9 = partial match, 0 = no match
    """
    normalized = normalize_text(clean_name)
    
    # Try exact match first
    if normalized in label_matcher:
        return label_matcher[normalized], 1.0
    
    # Try partial matching with known keywords
    keyword_mappings = {
        # Temperature
        'temp': ['Temp', 'Temperature'],
        'temperature': ['Temp', 'Temperature'],
        
        # Energy/Power
        'energy': ['energy'],
        'power': ['power'],
        'reactive': ['Reactive'],
        'apparent': ['Apparent'],
        
        # Current
        'current': ['current', 'Current'],
        'phase a': ['PhaseA'],
        'phase b': ['PhaseB'],
        'phase c': ['PhaseC'],
        
        # Flow
        'flow': ['Flow'],
        
        # Pressure
        'pressure': ['Pressure'],
        'press': ['Pressure'],
        
        # Humidity
        'humidity': ['Humidity'],
        'rh': ['Humidity'],
        
        # Damper
        'damper': ['Damper'],
        
        # Fan
        'fan': ['Fan'],
        
        # Valve
        'valve': ['Valve'],
        
        # Status
        'status': ['Status'],
        'sts': ['Status'],
        
        # Setpoint
        'setpoint': ['Sp'],
        'sp': ['Sp'],
    }
    
    # Find candidate labels that contain matching keywords
    words = normalized.split()
    candidates = []
    
    for label in label2id.keys():
        label_lower = label.lower()
        score = 0
        
        for word in words:
            if word in label_lower:
                score += 1
            for keyword, label_parts in keyword_mappings.items():
                if keyword in word:
                    for part in label_parts:
                        if part.lower() in label_lower:
                            score += 0.5
        
        if score > 0:
            candidates.append((label, score / len(words)))
    
    if candidates:
        # Return best match if confidence is reasonable
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = candidates[0]
        if best_score >= 0.5:
            return best_label, min(best_score, 0.9)  # Cap at 0.9 for non-exact matches
    
    return None, 0.0


def scan_profiles_directory(profiles_dir: Path) -> list[dict]:
    """Scan a directory for YAML profile files and extract all device resources."""
    all_resources = []
    
    for yaml_file in profiles_dir.glob('**/*.yml'):
        try:
            profile = load_yaml_file(yaml_file)
            resources = extract_device_resources(profile)
            for r in resources:
                r['source_file'] = str(yaml_file.relative_to(profiles_dir))
            all_resources.extend(resources)
        except Exception as e:
            print(f"Warning: Failed to process {yaml_file}: {e}")
    
    for yaml_file in profiles_dir.glob('**/*.yaml'):
        try:
            profile = load_yaml_file(yaml_file)
            resources = extract_device_resources(profile)
            for r in resources:
                r['source_file'] = str(yaml_file.relative_to(profiles_dir))
            all_resources.extend(resources)
        except Exception as e:
            print(f"Warning: Failed to process {yaml_file}: {e}")
    
    return all_resources


def main():
    parser = argparse.ArgumentParser(
        description='Extract device resource names from YAML profiles for training data'
    )
    parser.add_argument(
        '--profiles-dir', '-p',
        type=Path,
        required=True,
        help='Directory containing YAML profile files'
    )
    parser.add_argument(
        '--label-mapping', '-l',
        type=Path,
        help='Path to label_mapping.json for auto-matching (optional)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('extracted_points.csv'),
        help='Output CSV file'
    )
    parser.add_argument(
        '--min-confidence', '-c',
        type=float,
        default=0.0,
        help='Minimum confidence threshold for auto-matched labels (0.0-1.0)'
    )
    parser.add_argument(
        '--jsonl',
        action='store_true',
        help='Also output JSONL format suitable for training'
    )
    
    args = parser.parse_args()
    
    # Load label mapping if provided
    label2id = None
    label_matcher = None
    if args.label_mapping and args.label_mapping.exists():
        with open(args.label_mapping) as f:
            mapping = json.load(f)
            label2id = mapping.get('label2id', {})
            label_matcher = create_label_matcher(label2id)
            print(f"Loaded {len(label2id)} labels from {args.label_mapping}")
    
    # Scan profiles
    print(f"Scanning profiles in {args.profiles_dir}...")
    resources = scan_profiles_directory(args.profiles_dir)
    print(f"Found {len(resources)} device resources")
    
    # Process and optionally match labels
    results = []
    matched_count = 0
    unmatched = []
    
    for resource in resources:
        result = {
            'text': resource['clean_name'],
            'raw_name': resource['raw_name'],
            'profile': resource['profile_name'],
            'source_file': resource['source_file'],
            'value_type': resource['value_type'],
            'target': '',
            'confidence': 0.0,
        }
        
        if label_matcher:
            matched_label, confidence = attempt_auto_match(
                resource['clean_name'], 
                label_matcher, 
                label2id
            )
            if matched_label and confidence >= args.min_confidence:
                result['target'] = matched_label
                result['confidence'] = confidence
                matched_count += 1
            else:
                unmatched.append(resource['clean_name'])
        
        results.append(result)
    
    # Write CSV output
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'text', 'target', 'confidence', 'raw_name', 'profile', 'source_file', 'value_type'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Wrote {len(results)} entries to {args.output}")
    
    # Write JSONL if requested (only matched entries with confidence >= threshold)
    if args.jsonl:
        jsonl_path = args.output.with_suffix('.jsonl')
        matched_results = [r for r in results if r['target'] and r['confidence'] >= args.min_confidence]
        
        if label2id:
            with open(jsonl_path, 'w') as f:
                for r in matched_results:
                    entry = {
                        'text': r['text'],
                        'label': r['target'],
                        'label_id': label2id[r['target']]
                    }
                    f.write(json.dumps(entry) + '\n')
            print(f"Wrote {len(matched_results)} matched entries to {jsonl_path}")
    
    # Print summary
    if label_matcher:
        print(f"\nMatching Summary:")
        print(f"  Total resources: {len(resources)}")
        print(f"  Auto-matched:    {matched_count} ({100*matched_count/len(resources):.1f}%)")
        print(f"  Unmatched:       {len(unmatched)} ({100*len(unmatched)/len(resources):.1f}%)")
        
        if unmatched:
            print(f"\nUnmatched point names (need manual labeling):")
            for name in sorted(set(unmatched))[:20]:
                print(f"  - {name}")
            if len(unmatched) > 20:
                print(f"  ... and {len(unmatched) - 20} more")


if __name__ == '__main__':
    main()