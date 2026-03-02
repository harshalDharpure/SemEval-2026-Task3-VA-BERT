#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert prediction files from Aspect_VA format (VA as string) to eval format (separate valence/arousal fields).
"""

import json
import os
import sys
from pathlib import Path


def convert_file(input_path: str, output_path: str):
    """Convert a single prediction file from Aspect_VA format to eval format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    converted = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            
            # Extract ID
            text_id = record.get('ID') or record.get('id')
            
            # Extract Aspect_VA list
            aspect_list = record.get('Aspect_VA', [])
            if not aspect_list:
                continue
            
            # Convert each aspect
            for item in aspect_list:
                aspect = item.get('Aspect') or item.get('aspect')
                va_str = item.get('VA', '')
                
                try:
                    valence, arousal = map(float, va_str.split('#'))
                except Exception:
                    continue
                
                converted.append({
                    'id': text_id,
                    'aspect': aspect,
                    'valence': valence,
                    'arousal': arousal
                })
    
    # Write converted file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted)} predictions: {input_path} -> {output_path}")


def convert_directory(input_dir: str):
    """Convert all prediction files in a directory."""
    input_path = Path(input_dir)
    output_dir = input_path / '_converted'
    output_dir.mkdir(exist_ok=True)
    
    # Find all pred_*.jsonl files
    pred_files = list(input_path.glob('pred_*.jsonl'))
    
    if not pred_files:
        print(f"No prediction files found in {input_dir}")
        return
    
    for pred_file in pred_files:
        output_file = output_dir / pred_file.name
        convert_file(str(pred_file), str(output_file))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_predictions_to_eval_format.py <input_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    convert_directory(input_dir)
