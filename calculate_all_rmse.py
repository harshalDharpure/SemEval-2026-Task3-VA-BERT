#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate RMSE for all completed experiments.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

EXPERIMENTS_DIR = "experiments"
TEST_DATA_DIR = "data/test_20"
EVAL_SCRIPT = "evaluation/eval_task1.py"
CONVERT_SCRIPT = "evaluation/convert_predictions_to_eval_format.py"


def find_completed_experiments():
    """Find all experiments with prediction files."""
    completed = []
    exp_dir = Path(EXPERIMENTS_DIR)
    
    if not exp_dir.exists():
        return completed
    
    for exp_folder in exp_dir.iterdir():
        if not exp_folder.is_dir():
            continue
        
        # Look for subtask_1 directory with pred_*.jsonl files
        subtask_dir = exp_folder / "subtask_1"
        if not subtask_dir.exists():
            continue
        
        pred_files = list(subtask_dir.glob("pred_*.jsonl"))
        if pred_files:
            # Check if converted directory exists, if not, we need to convert
            converted_dir = subtask_dir / "_converted"
            if not converted_dir.exists():
                # Convert predictions
                print(f"\n[Converting] {exp_folder.name}")
                subprocess.run([sys.executable, CONVERT_SCRIPT, str(subtask_dir)], 
                             check=False)
            
            completed.append({
                'exp_name': exp_folder.name,
                'exp_path': str(exp_folder),
                'pred_dir': str(subtask_dir / "_converted"),
                'pred_files': [f.name for f in pred_files]
            })
    
    return completed


def extract_experiment_info(exp_name):
    """Extract experiment type, model, data structure, and language from exp name."""
    parts = exp_name.split('_')
    
    info = {
        'exp_type': None,
        'model': None,
        'data_structure': None,
        'language': None
    }
    
    # Experiment type
    if 'exp1' in exp_name or 'direct_finetune' in exp_name:
        info['exp_type'] = 'Exp1'
    elif 'exp2' in exp_name or 'pretrained_only' in exp_name:
        info['exp_type'] = 'Exp2'
    elif 'exp3' in exp_name or 'pretrain_finetune' in exp_name:
        info['exp_type'] = 'Exp3'
    
    # Model
    if 'base' in exp_name:
        info['model'] = 'base'
    elif 'large' in exp_name:
        info['model'] = 'large'
    elif 'xl' in exp_name:
        info['model'] = 'xl'
    elif 'mbert' in exp_name:
        info['model'] = 'mbert'
    elif 'mdeberta' in exp_name:
        info['model'] = 'mdeberta'
    
    # Data structure
    if 'language_specific' in exp_name:
        info['data_structure'] = 'language_specific'
    elif 'multilingual_shuffled' in exp_name or 'shuffled' in exp_name:
        info['data_structure'] = 'multilingual_shuffled'
    
    # Language
    lang_codes = ['eng', 'jpn', 'rus', 'tat', 'ukr', 'zho']
    for lang in lang_codes:
        if lang in exp_name:
            info['language'] = lang
            break
    
    return info


def calculate_rmse_for_experiment(exp_info):
    """Calculate RMSE for a single experiment."""
    exp_name = exp_info['exp_name']
    pred_dir = exp_info['pred_dir']
    
    # Check if converted predictions exist
    if not os.path.exists(pred_dir):
        print(f"[SKIP] {exp_name}: No converted predictions found")
        return None
    
    # Run evaluation
    output_json = os.path.join(exp_info['exp_path'], 'rmse_results.json')
    
    try:
        result = subprocess.run(
            [sys.executable, EVAL_SCRIPT,
             '--test_dir', TEST_DATA_DIR,
             '--pred_dir', pred_dir,
             '--output_json', output_json],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load results
        with open(output_json, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {exp_name}: Evaluation failed")
        print(f"  stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"[ERROR] {exp_name}: {str(e)}")
        return None


def main():
    print("=" * 80)
    print("Calculating RMSE for all completed experiments")
    print("=" * 80)
    
    # Find completed experiments
    completed = find_completed_experiments()
    print(f"\nFound {len(completed)} completed experiments")
    
    if not completed:
        print("No completed experiments found!")
        return
    
    # Calculate RMSE for each
    all_results = {}
    
    for exp_info in completed:
        exp_name = exp_info['exp_name']
        print(f"\n[Processing] {exp_name}")
        
        exp_info_dict = extract_experiment_info(exp_name)
        results = calculate_rmse_for_experiment(exp_info)
        
        if results:
            all_results[exp_name] = {
                'info': exp_info_dict,
                'results': results
            }
            print(f"  ✓ RMSE calculated")
        else:
            print(f"  ✗ Failed to calculate RMSE")
    
    # Save summary
    summary_file = os.path.join(EXPERIMENTS_DIR, 'all_rmse_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'=' * 80}")
    
    # Print summary table
    print("\nRMSE Summary by Experiment:")
    print("-" * 80)
    print(f"{'Experiment':<50} | {'Lang':<5} | {'Avg RMSE':<10} | {'V RMSE':<10} | {'A RMSE':<10}")
    print("-" * 80)
    
    for exp_name, data in sorted(all_results.items()):
        info = data['info']
        results = data['results']
        
        # Get per-language summary
        by_lang = results.get('by_language', {})
        if by_lang:
            for lang, lang_data in sorted(by_lang.items()):
                avg_rmse = lang_data.get('rmse_avg', float('nan'))
                v_rmse = lang_data.get('rmse_valence', float('nan'))
                a_rmse = lang_data.get('rmse_arousal', float('nan'))
                
                exp_display = f"{info['exp_type']} {info['model']} {info['data_structure']}"
                print(f"{exp_display:<50} | {lang:<5} | {avg_rmse:>10.4f} | {v_rmse:>10.4f} | {a_rmse:>10.4f}")
        else:
            # Fallback to overall results
            exp_display = f"{info['exp_type']} {info['model']} {info['data_structure']}"
            if info['language']:
                print(f"{exp_display:<50} | {info['language']:<5} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
            else:
                print(f"{exp_display:<50} | {'N/A':<5} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")


if __name__ == '__main__':
    main()
