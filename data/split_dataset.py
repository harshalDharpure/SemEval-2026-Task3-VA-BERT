#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Splitting Script
Splits official training data into 80/20 train/test split
"""

import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def jsonl_to_df(jsonl_data):
    """Convert JSONL to DataFrame"""
    rows = []
    for record in jsonl_data:
        text = record['Text']
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
                        'ID': record['ID'],
                        'Text': text,
                        'Aspect': aspect,
                        'Valence': valence,
                        'Arousal': arousal
                    })
                except (ValueError, AttributeError) as e:
                    # Skip invalid VA format
                    continue
    return pd.DataFrame(rows)

def reconstruct_jsonl(grouped_df):
    """Reconstruct JSONL format from grouped DataFrame"""
    reconstructed_data = []
    for group_id, group_df in grouped_df:
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
        # Only add records that have at least one aspect
        if record["Aspect_VA"]:
            reconstructed_data.append(record)
    return reconstructed_data

def main():
    parser = argparse.ArgumentParser(description='Split dataset into 80/20 train/test')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with original data')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory for split data')
    parser.add_argument('--languages', nargs='+', default=['eng', 'jpn', 'rus', 'tat', 'ukr', 'zho'], help='Languages to process (English, Japanese, Russian, Tatar, Ukrainian, Chinese)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directories
    output_train_dir = os.path.join(args.output_dir, 'train_80')
    output_test_dir = os.path.join(args.output_dir, 'test_20')
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)
    
    # Process each language
    for lang in args.languages:
        lang_dir = os.path.join(args.input_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"Warning: {lang_dir} not found, skipping...")
            continue
        
        # Create language subdirectories
        output_lang_train_dir = os.path.join(output_train_dir, lang)
        output_lang_test_dir = os.path.join(output_test_dir, lang)
        os.makedirs(output_lang_train_dir, exist_ok=True)
        os.makedirs(output_lang_test_dir, exist_ok=True)
        
        # Process each JSONL file in the language directory
        for filename in os.listdir(lang_dir):
            if not filename.endswith('.jsonl'):
                continue
            
            # Only process training files, skip dev/test files
            if 'dev' in filename.lower() or 'test' in filename.lower():
                continue
            if 'train' not in filename.lower():
                continue
            
            input_file = os.path.join(lang_dir, filename)
            print(f"Processing {input_file}...")
            
            # Load and convert to DataFrame
            jsonl_data = load_jsonl(input_file)
            if not jsonl_data:
                print(f"  Warning: {input_file} is empty, skipping...")
                continue
            
            df = jsonl_to_df(jsonl_data)
            
            if df.empty:
                print(f"  Warning: {input_file} has no valid data, skipping...")
                continue
            
            # Get unique IDs and split
            unique_ids = df['ID'].unique()
            train_ids, test_ids = train_test_split(
                unique_ids,
                test_size=args.test_size,
                random_state=args.seed
            )
            
            # Split data
            train_df = df[df['ID'].isin(train_ids)]
            test_df = df[df['ID'].isin(test_ids)]
            
            # Reconstruct JSONL format
            train_grouped = train_df.groupby('ID')
            test_grouped = test_df.groupby('ID')
            
            train_jsonl = reconstruct_jsonl(train_grouped)
            test_jsonl = reconstruct_jsonl(test_grouped)
            
            # Save files
            input_basename = os.path.basename(input_file)
            train_file_path = os.path.join(
                output_lang_train_dir,
                input_basename.replace('.jsonl', '_train_80.jsonl')
            )
            test_file_path = os.path.join(
                output_lang_test_dir,
                input_basename.replace('.jsonl', '_test_20.jsonl')
            )
            
            with open(train_file_path, 'w', encoding='utf-8') as f:
                for record in train_jsonl:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                for record in test_jsonl:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"   Train: {len(train_jsonl)} records -> {train_file_path}")
            print(f"   Test: {len(test_jsonl)} records -> {test_file_path}")
    
    print(f"\n Dataset splitting completed!")
    print(f"Train data: {output_train_dir}")
    print(f"Test data: {output_test_dir}")

if __name__ == "__main__":
    main()
