#!/usr/bin/env python3
"""
Device Profile Ingestion Pipeline for HVAC Point Classification
================================================================

A comprehensive pipeline for:
1. Scanning device profiles for point names
2. Matching against existing labels with configurable confidence
3. Tracking unmatched points for taxonomy expansion decisions
4. Generating training-ready JSONL files

This is designed to be run periodically as new profiles are added.

Usage:
    # First time setup - analyze what you have
    python profile_pipeline.py analyze --profiles-dir ./profiles --label-mapping data/label_mapping.json
    
    # Generate training data (matched points only)
    python profile_pipeline.py generate --profiles-dir ./profiles --label-mapping data/label_mapping.json --output data/from_profiles.jsonl
    
    # Generate report of unmatched points for taxonomy review
    python profile_pipeline.py report-unmatched --profiles-dir ./profiles --label-mapping data/label_mapping.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import csv

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)


@dataclass
class PointMatch:
    """Represents a matched point name."""
    raw_name: str
    clean_name: str
    label: str
    label_id: int
    confidence: float
    profile_name: str
    source_file: str


@dataclass  
class UnmatchedPoint:
    """Represents an unmatched point name."""
    raw_name: str
    clean_name: str
    profile_name: str
    source_file: str
    suggested_label: Optional[str] = None


class LabelMatcher:
    """
    Intelligent label matcher with configurable rules.
    
    Matching strategy:
    1. Exact match (normalized)
    2. Keyword-based matching
    3. Pattern matching
    """
    
    def __init__(self, label2id: dict):
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        
        # Build normalized lookup
        self.normalized_labels = {}
        for label in label2id.keys():
            normalized = self._normalize(label)
            self.normalized_labels[normalized] = label
            
            # Also add camelCase split version
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', label).lower()
            self.normalized_labels[split] = label
        
        # Keyword mappings for HVAC domain
        self.keyword_mappings = self._build_keyword_mappings()
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[-_:\s]+', ' ', text)
        return text.strip()
    
    def _build_keyword_mappings(self) -> dict:
        """Build mappings from common keywords to label patterns."""
        return {
            # Temperature
            ('temp', 'temperature'): {
                'zone': 'zoneTemp',
                'discharge': 'dischargeTemp', 
                'supply': 'dischargeTemp',
                'return': 'returnTemp',
                'outside': 'outsideTemp',
                'outdoor': 'outsideTemp',
                'mixed': 'mixedTemp',
            },
            # Humidity
            ('humidity', 'rh'): {
                'zone': 'zoneHumidity',
                'discharge': 'dischargeHumidity',
                'return': 'returnHumidity',
                'outside': 'outsideHumidity',
            },
            # Fan
            ('fan',): {
                'discharge': 'dischargeFanStatus',
                'supply': 'dischargeFanStatus',
                'return': 'returnFanStatus',
                'exhaust': 'exhaustFanStatus',
            },
            # Damper
            ('damper',): {
                'discharge': 'dischargeDamper',
                'return': 'returnDamper',
                'outside': 'outsideDamper',
                'exhaust': 'exhaustDamper',
            },
            # Current/Power (electrical)
            ('current',): {
                'phase a': 'currentPhaseA',
                'phase b': 'currentPhaseB', 
                'phase c': 'currentPhaseC',
                'average': 'current',
                'default': 'current',
            },
            # Energy
            ('energy',): {
                'reactive': 'energyReactive',
                'apparent': 'energyApparent',
                'default': 'energy',
            },
        }
    
    def match(self, point_name: str) -> tuple[Optional[str], float]:
        """
        Attempt to match a point name to a label.
        
        Returns: (label, confidence) or (None, 0.0)
        """
        normalized = self._normalize(point_name)
        
        # 1. Try exact match
        if normalized in self.normalized_labels:
            return self.normalized_labels[normalized], 1.0
        
        # 2. Try keyword-based matching
        for keywords, location_map in self.keyword_mappings.items():
            if any(kw in normalized for kw in keywords):
                # Found a keyword match, now find location
                for location, label in location_map.items():
                    if location == 'default':
                        continue
                    if location in normalized:
                        if label in self.label2id:
                            return label, 0.85
                
                # No location match, try default
                if 'default' in location_map:
                    label = location_map['default']
                    if label in self.label2id:
                        return label, 0.6
        
        # 3. Try fuzzy matching - find labels that share words
        words = set(normalized.split())
        best_match = None
        best_score = 0.0
        
        for label in self.label2id.keys():
            label_normalized = self._normalize(label)
            label_words = set(label_normalized.split())
            
            # Calculate Jaccard-like similarity
            intersection = words & label_words
            if intersection:
                union = words | label_words
                score = len(intersection) / len(union)
                if score > best_score and score >= 0.3:
                    best_score = score
                    best_match = label
        
        if best_match:
            return best_match, min(best_score, 0.7)
        
        return None, 0.0


def load_profile(filepath: Path) -> dict:
    """Load a YAML profile file."""
    with open(filepath, encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_points_from_profile(profile: dict, source_file: str) -> list[tuple[str, str]]:
    """Extract (raw_name, clean_name) tuples from a profile."""
    points = []
    profile_name = profile.get('name', 'unknown')
    
    for resource in profile.get('deviceResources', []):
        raw_name = resource.get('name', '')
        if not raw_name:
            continue
        
        # Remove :present-value suffix
        clean_name = re.sub(r':present-value$', '', raw_name, flags=re.IGNORECASE)
        points.append((raw_name, clean_name, profile_name, source_file))
    
    return points


def scan_profiles(profiles_dir: Path) -> list[tuple]:
    """Scan directory for all YAML profiles and extract points."""
    all_points = []
    
    for pattern in ['**/*.yml', '**/*.yaml']:
        for filepath in profiles_dir.glob(pattern):
            try:
                profile = load_profile(filepath)
                rel_path = str(filepath.relative_to(profiles_dir))
                points = extract_points_from_profile(profile, rel_path)
                all_points.extend(points)
            except Exception as e:
                print(f"Warning: Failed to process {filepath}: {e}", file=sys.stderr)
    
    return all_points


def cmd_analyze(args):
    """Analyze command - show overview of profile data."""
    print(f"Scanning profiles in: {args.profiles_dir}")
    
    points = scan_profiles(args.profiles_dir)
    print(f"Found {len(points)} device resources\n")
    
    if not points:
        print("No points found. Check that profiles directory contains .yml/.yaml files.")
        return
    
    # Load label mapping
    with open(args.label_mapping) as f:
        mapping = json.load(f)
        label2id = mapping['label2id']
    
    matcher = LabelMatcher(label2id)
    
    # Analyze matches
    matched = []
    unmatched = []
    
    for raw_name, clean_name, profile_name, source_file in points:
        label, confidence = matcher.match(clean_name)
        if label and confidence >= args.min_confidence:
            matched.append(PointMatch(
                raw_name=raw_name,
                clean_name=clean_name,
                label=label,
                label_id=label2id[label],
                confidence=confidence,
                profile_name=profile_name,
                source_file=source_file
            ))
        else:
            unmatched.append(UnmatchedPoint(
                raw_name=raw_name,
                clean_name=clean_name,
                profile_name=profile_name,
                source_file=source_file
            ))
    
    # Print summary
    print("=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total points:     {len(points)}")
    print(f"Matched:          {len(matched)} ({100*len(matched)/len(points):.1f}%)")
    print(f"Unmatched:        {len(unmatched)} ({100*len(unmatched)/len(points):.1f}%)")
    print(f"Min confidence:   {args.min_confidence}")
    print()
    
    # Label distribution
    label_counts = defaultdict(int)
    for m in matched:
        label_counts[m.label] += 1
    
    print("Matched label distribution (top 15):")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {label}: {count}")
    print()
    
    # Unique unmatched point names
    unique_unmatched = sorted(set(u.clean_name for u in unmatched))
    print(f"Unique unmatched point names: {len(unique_unmatched)}")
    for name in unique_unmatched[:20]:
        print(f"  • {name}")
    if len(unique_unmatched) > 20:
        print(f"  ... and {len(unique_unmatched) - 20} more")


def cmd_generate(args):
    """Generate command - create training JSONL."""
    print(f"Scanning profiles in: {args.profiles_dir}")
    
    points = scan_profiles(args.profiles_dir)
    print(f"Found {len(points)} device resources")
    
    # Load label mapping
    with open(args.label_mapping) as f:
        mapping = json.load(f)
        label2id = mapping['label2id']
    
    matcher = LabelMatcher(label2id)
    
    # Match and generate
    entries = []
    seen_texts = set()
    
    for raw_name, clean_name, profile_name, source_file in points:
        # Deduplicate
        norm = clean_name.lower().strip()
        if norm in seen_texts:
            continue
        seen_texts.add(norm)
        
        label, confidence = matcher.match(clean_name)
        if label and confidence >= args.min_confidence:
            entries.append({
                'text': clean_name,
                'label': label,
                'label_id': label2id[label],
                'confidence': round(confidence, 3),
                'source': source_file
            })
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        for entry in entries:
            # Write only required fields for training
            train_entry = {
                'text': entry['text'],
                'label': entry['label'],
                'label_id': entry['label_id']
            }
            f.write(json.dumps(train_entry) + '\n')
    
    print(f"\nGenerated {len(entries)} training entries -> {args.output}")
    print(f"(from {len(points)} total points, {len(points) - len(entries)} filtered/deduplicated)")


def cmd_report_unmatched(args):
    """Report unmatched points for taxonomy review."""
    points = scan_profiles(args.profiles_dir)
    
    with open(args.label_mapping) as f:
        mapping = json.load(f)
        label2id = mapping['label2id']
    
    matcher = LabelMatcher(label2id)
    
    # Find all unmatched
    unmatched_counts = defaultdict(lambda: {'count': 0, 'profiles': set()})
    
    for raw_name, clean_name, profile_name, source_file in points:
        label, confidence = matcher.match(clean_name)
        if not label or confidence < args.min_confidence:
            unmatched_counts[clean_name]['count'] += 1
            unmatched_counts[clean_name]['profiles'].add(profile_name)
    
    # Write report
    output = args.output or Path('unmatched_points_report.csv')
    
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['point_name', 'occurrence_count', 'num_profiles', 'suggested_label'])
        
        for name, data in sorted(unmatched_counts.items(), key=lambda x: -x[1]['count']):
            writer.writerow([
                name,
                data['count'],
                len(data['profiles']),
                ''  # Empty for manual review
            ])
    
    print(f"Unmatched points report: {output}")
    print(f"Total unique unmatched: {len(unmatched_counts)}")
    print(f"\nMost common unmatched points:")
    for name, data in sorted(unmatched_counts.items(), key=lambda x: -x[1]['count'])[:15]:
        print(f"  {name}: {data['count']}x across {len(data['profiles'])} profile(s)")


def main():
    parser = argparse.ArgumentParser(
        description='Device Profile Ingestion Pipeline for HVAC Point Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Analyze command
    analyze = subparsers.add_parser('analyze', help='Analyze profiles and show match statistics')
    analyze.add_argument('--profiles-dir', '-p', type=Path, required=True)
    analyze.add_argument('--label-mapping', '-l', type=Path, required=True)
    analyze.add_argument('--min-confidence', '-c', type=float, default=0.6)
    analyze.set_defaults(func=cmd_analyze)
    
    # Generate command
    generate = subparsers.add_parser('generate', help='Generate training JSONL from profiles')
    generate.add_argument('--profiles-dir', '-p', type=Path, required=True)
    generate.add_argument('--label-mapping', '-l', type=Path, required=True)
    generate.add_argument('--output', '-o', type=Path, default=Path('data/from_profiles.jsonl'))
    generate.add_argument('--min-confidence', '-c', type=float, default=0.6)
    generate.set_defaults(func=cmd_generate)
    
    # Report unmatched command
    report = subparsers.add_parser('report-unmatched', help='Generate CSV report of unmatched points')
    report.add_argument('--profiles-dir', '-p', type=Path, required=True)
    report.add_argument('--label-mapping', '-l', type=Path, required=True)
    report.add_argument('--output', '-o', type=Path)
    report.add_argument('--min-confidence', '-c', type=float, default=0.6)
    report.set_defaults(func=cmd_report_unmatched)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()