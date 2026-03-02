#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilingual Shuffled Dataset Creation Script
Creates a single shuffled JSONL file containing all languages and domains mixed together
"""

import os
import json
import argparse
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def jsonl_to_df(jsonl_data, lang=None):
    """Convert JSONL to DataFrame with language tag"""
    rows = []
    for record in jsonl_data:
        text = record['Text']
        record_id = record['ID']
        # Handle both Aspect_VA and Quadruplet formats
        aspect_va_list = record.get('Aspect_VA', [])
        if not aspect_va_list:
            # Try Quadruplet format
            aspect_va_list = record.get('Quadruplet', [])
        
        for aspect_va in aspect_va_list:
            aspect = aspect_va.get('Aspect', aspect_va.get('Category', 'UNKNOWN'))
            va_str = aspect_va.get('VA', '')
            if va_str:
                try:
                    valence, arousal = map(float, va_str.split('#'))
                    rows.append({
                        'ID': record_id,
                        'Text': text,
                        'Aspect': aspect,
                        'Valence': valence,
                        'Arousal': arousal,
                        'Language': lang,  # Track original language
                        'OriginalRecord': record  # Keep full record for reconstruction
                    })
                except (ValueError, AttributeError):
                    # Skip invalid VA format
                    continue
    return pd.DataFrame(rows)

def reconstruct_jsonl(grouped_df):
    """Reconstruct JSONL format from grouped DataFrame"""
    reconstructed_data = []
    for group_id, group_df in grouped_df:
        # Get the original record structure from first row
        original_record = group_df['OriginalRecord'].iloc[0]
        record = {
            "ID": group_id,
            "Text": group_df['Text'].iloc[0],
            "Aspect_VA": []
        }
        for _, row in group_df.iterrows():
            record["Aspect_VA"].append({
                "Aspect": row["Aspect"],
                "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
            })
        reconstructed_data.append(record)
    return reconstructed_data

def main():
    parser = argparse.ArgumentParser(description='Create multilingual shuffled dataset with 80/20 split')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory with original data (task-dataset/track_a/subtask_1)')
    parser.add_argument('--output_dir', type=str, default='./data', 
                       help='Output directory for split data')
    parser.add_argument('--languages', nargs='+', 
                       default=['eng', 'jpn', 'rus', 'tat', 'ukr', 'zho'],
                       help='Languages to process')
    parser.add_argument('--test_size', type=float, default=0.2, 
                       help='Test set size (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for shuffling and splitting')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file paths (single files for all languages)
    output_train_file = os.path.join(args.output_dir, 'train_80_multilingual_shuffled.jsonl')
    output_test_file = os.path.join(args.output_dir, 'test_20_multilingual_shuffled.jsonl')
    
    # Step 1: Load all data from all languages and domains
    print("="*60)
    print("Step 1: Loading all training data from all languages...")
    print("="*60)
    
    all_data = []
    language_stats = {}
    
    for lang in args.languages:
        lang_dir = os.path.join(args.input_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"Warning: {lang_dir} not found, skipping...")
            continue
        
        lang_records = 0
        # Process each JSONL file in the language directory
        for filename in os.listdir(lang_dir):
            if not filename.endswith('.jsonl') or 'train' not in filename.lower():
                continue
            
            input_file = os.path.join(lang_dir, filename)
            print(f"Loading {input_file}...")
            
            try:
                jsonl_data = load_jsonl(input_file)
                # Add language tag to each record
                for record in jsonl_data:
                    record['_source_lang'] = lang
                    record['_source_file'] = filename
                    all_data.append(record)
                    lang_records += 1
            except Exception as e:
                print(f"  Error loading {input_file}: {e}")
                continue
        
        language_stats[lang] = lang_records
        print(f"  {lang}: {lang_records} records")
    
    print(f"\nTotal records loaded: {len(all_data)}")
    print(f"Languages: {list(language_stats.keys())}")
    print(f"Language distribution: {language_stats}")
    
    if not all_data:
        print("Error: No training data found!")
        return
    
    # Step 2: Shuffle all data
    print("\n" + "="*60)
    print("Step 2: Shuffling all data...")
    print("="*60)
    random.shuffle(all_data)
    print(f"Shuffled {len(all_data)} records")
    
    # Step 3: Split into train/test (80/20) at record level
    print("\n" + "="*60)
    print("Step 3: Splitting into 80/20 train/test...")
    print("="*60)
    
    # Get unique IDs for splitting
    unique_ids = list(set(record['ID'] for record in all_data))
    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)
    
    train_data = [r for r in all_data if r['ID'] in train_ids_set]
    test_data = [r for r in all_data if r['ID'] in test_ids_set]
    
    print(f"Train: {len(train_data)} records ({len(train_ids)} unique IDs)")
    print(f"Test: {len(test_data)} records ({len(test_ids)} unique IDs)")
    
    # Step 4: Clean metadata and prepare for saving
    print("\n" + "="*60)
    print("Step 4: Preparing data for saving...")
    print("="*60)
    
    # Remove metadata fields and count language distribution
    train_clean = []
    test_clean = []
    train_lang_dist = {}
    test_lang_dist = {}
    
    for record in train_data:
        lang = record.get('_source_lang', 'unknown')
        train_lang_dist[lang] = train_lang_dist.get(lang, 0) + 1
        # Remove metadata fields before saving
        clean_record = {k: v for k, v in record.items() if not k.startswith('_')}
        train_clean.append(clean_record)
    
    for record in test_data:
        lang = record.get('_source_lang', 'unknown')
        test_lang_dist[lang] = test_lang_dist.get(lang, 0) + 1
        # Remove metadata fields before saving
        clean_record = {k: v for k, v in record.items() if not k.startswith('_')}
        test_clean.append(clean_record)
    
    # Step 5: Save to single JSONL files
    print("\n" + "="*60)
    print("Step 5: Saving to single shuffled JSONL files...")
    print("="*60)
    
    save_jsonl(train_clean, output_train_file)
    print(f"  Train: {len(train_clean)} records -> {output_train_file}")
    
    save_jsonl(test_clean, output_test_file)
    print(f"  Test: {len(test_clean)} records -> {output_test_file}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Train file: {output_train_file}")
    print(f"Test file: {output_test_file}")
    print(f"\nTotal train records: {len(train_clean)}")
    print(f"Total test records: {len(test_clean)}")
    print(f"\nLanguage distribution in train set:")
    for lang in sorted(train_lang_dist.keys()):
        print(f"  {lang}: {train_lang_dist[lang]} records")
    print(f"\nLanguage distribution in test set:")
    for lang in sorted(test_lang_dist.keys()):
        print(f"  {lang}: {test_lang_dist[lang]} records")
    print("\n✅ Multilingual shuffled dataset creation completed!")

if __name__ == "__main__":
    main()
