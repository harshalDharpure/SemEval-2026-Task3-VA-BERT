#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalculate missing valence/arousal RMSE values from saved predictions
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_rmse_from_predictions(pred_file, test_file):
    """Calculate RMSE from prediction and test files"""
    try:
        predictions = load_jsonl(pred_file)
        test_data = load_jsonl(test_file)
        
        if len(predictions) != len(test_data):
            print(f"  ⚠️  Mismatch: {len(predictions)} predictions vs {len(test_data)} test samples")
            return None, None, None
        
        valence_pred = []
        valence_true = []
        arousal_pred = []
        arousal_true = []
        
        for pred, test in zip(predictions, test_data):
            # Handle prediction format: Aspect_VA with 'VA': 'valence#arousal'
            pred_v = None
            pred_a = None
            
            if isinstance(pred, dict):
                if 'valence' in pred and 'arousal' in pred:
                    pred_v = float(pred['valence'])
                    pred_a = float(pred['arousal'])
                elif 'pred_valence' in pred and 'pred_arousal' in pred:
                    pred_v = float(pred['pred_valence'])
                    pred_a = float(pred['pred_arousal'])
                elif 'Aspect_VA' in pred and isinstance(pred['Aspect_VA'], list):
                    # Format: Aspect_VA: [{'Aspect': '...', 'VA': 'valence#arousal'}]
                    for aspect_va in pred['Aspect_VA']:
                        if 'VA' in aspect_va:
                            va_str = aspect_va['VA']
                            if '#' in va_str:
                                parts = va_str.split('#')
                                if len(parts) == 2:
                                    pred_v = float(parts[0])
                                    pred_a = float(parts[1])
                                    break
            
            # Handle test format: Aspect_VA with 'VA': 'valence#arousal'
            test_v = None
            test_a = None
            
            if isinstance(test, dict):
                if 'valence' in test and 'arousal' in test:
                    test_v = float(test['valence'])
                    test_a = float(test['arousal'])
                elif 'label_valence' in test and 'label_arousal' in test:
                    test_v = float(test['label_valence'])
                    test_a = float(test['label_arousal'])
                elif 'Aspect_VA' in test and isinstance(test['Aspect_VA'], list):
                    # Format: Aspect_VA: [{'Aspect': '...', 'VA': 'valence#arousal'}]
                    # For test, we need to average all aspects
                    all_v = []
                    all_a = []
                    for aspect_va in test['Aspect_VA']:
                        if 'VA' in aspect_va:
                            va_str = aspect_va['VA']
                            if '#' in va_str:
                                parts = va_str.split('#')
                                if len(parts) == 2:
                                    all_v.append(float(parts[0]))
                                    all_a.append(float(parts[1]))
                    if all_v and all_a:
                        test_v = sum(all_v) / len(all_v)
                        test_a = sum(all_a) / len(all_a)
            
            if pred_v is not None and pred_a is not None and test_v is not None and test_a is not None:
                valence_pred.append(pred_v)
                valence_true.append(test_v)
                arousal_pred.append(pred_a)
                arousal_true.append(test_a)
        
        if len(valence_pred) == 0:
            return None, None, None
        
        rmse_v = np.sqrt(mean_squared_error(valence_true, valence_pred))
        rmse_a = np.sqrt(mean_squared_error(arousal_true, arousal_pred))
        rmse = (rmse_v + rmse_a) / 2
        
        return rmse, rmse_v, rmse_a
    except Exception as e:
        print(f"  ❌ Error calculating RMSE: {e}")
        return None, None, None

def update_exp1_results():
    """Update Experiment 1 results with missing valence/arousal RMSE"""
    print("\n" + "="*90)
    print("UPDATING EXPERIMENT 1 - Missing Valence/Arousal RMSE")
    print("="*90)
    
    updated_count = 0
    for model in ['base', 'large']:
        for lang in ['eng', 'rus', 'ukr', 'jpn', 'tat', 'zho']:
            exp_dir = Path(f"experiments/exp1_direct_finetune_{model}_language_specific_{lang}")
            if not exp_dir.exists():
                continue
            
            results_dir = exp_dir / "results"
            if not results_dir.exists():
                continue
            
            json_files = list(results_dir.glob("finetune_*_results_*.json"))
            pred_files = list(results_dir.glob("predictions_*.jsonl"))
            
            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    domain = data.get('config', {}).get('domain', 'unknown')
                    if domain == 'unknown':
                        continue
                    
                    # Check if valence/arousal RMSE is missing
                    if data.get('best_valence_rmse') is None or data.get('best_arousal_rmse') is None:
                        # Try to find prediction file
                        pred_file = None
                        for pf in pred_files:
                            if domain in pf.name and lang in pf.name:
                                pred_file = pf
                                break
                        
                        if pred_file:
                            # Find test file
                            test_file_patterns = [
                                f"data/test_20/{lang}/{lang}_{domain}_train_alltasks_test_20.jsonl",
                                f"data/test_20/{lang}/{lang}_{domain}_train_task1_test_20.jsonl"
                            ]
                            test_file = None
                            for pattern in test_file_patterns:
                                if Path(pattern).exists():
                                    test_file = pattern
                                    break
                            
                            if test_file:
                                print(f"  📊 {model}/{lang}/{domain}: Recalculating from predictions...")
                                rmse, rmse_v, rmse_a = calculate_rmse_from_predictions(pred_file, test_file)
                                if rmse_v is not None and rmse_a is not None:
                                    data['best_valence_rmse'] = float(rmse_v)
                                    data['best_arousal_rmse'] = float(rmse_a)
                                    # Update best_rmse if needed
                                    if data.get('best_rmse') is None:
                                        data['best_rmse'] = float(rmse)
                                    
                                    with open(json_file, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    updated_count += 1
                                    print(f"     ✅ Updated: rmse_v={rmse_v:.4f}, rmse_a={rmse_a:.4f}")
                                else:
                                    print(f"     ❌ Could not calculate RMSE")
                            else:
                                print(f"  ⚠️  {model}/{lang}/{domain}: Test file not found")
                        else:
                            print(f"  ⚠️  {model}/{lang}/{domain}: Prediction file not found")
                except Exception as e:
                    print(f"  ❌ Error processing {json_file}: {e}")
    
    print(f"\n✅ Updated {updated_count} Experiment 1 results")

def update_exp3_results():
    """Update Experiment 3 results with missing valence/arousal RMSE"""
    print("\n" + "="*90)
    print("UPDATING EXPERIMENT 3 - Missing Valence/Arousal RMSE")
    print("="*90)
    
    updated_count = 0
    for model in ['base', 'large']:
        for lang in ['eng', 'rus', 'ukr', 'jpn', 'tat', 'zho']:
            exp_dir = Path(f"experiments/exp3_pretrain_finetune_{model}_language_specific_{lang}_finetune")
            if not exp_dir.exists():
                continue
            
            results_dir = exp_dir / "results"
            if not results_dir.exists():
                continue
            
            json_files = list(results_dir.glob("finetune_*_results_*.json"))
            pred_files = list(results_dir.glob("predictions_*.jsonl"))
            
            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    domain = data.get('config', {}).get('domain', 'unknown')
                    if domain == 'unknown':
                        continue
                    
                    # Check if valence/arousal RMSE is missing
                    if data.get('best_valence_rmse') is None or data.get('best_arousal_rmse') is None:
                        # Try to find prediction file
                        pred_file = None
                        for pf in pred_files:
                            if domain in pf.name and lang in pf.name:
                                pred_file = pf
                                break
                        
                        if pred_file:
                            # Find test file
                            test_file_patterns = [
                                f"data/test_20/{lang}/{lang}_{domain}_train_alltasks_test_20.jsonl",
                                f"data/test_20/{lang}/{lang}_{domain}_train_task1_test_20.jsonl"
                            ]
                            test_file = None
                            for pattern in test_file_patterns:
                                if Path(pattern).exists():
                                    test_file = pattern
                                    break
                            
                            if test_file:
                                print(f"  📊 {model}/{lang}/{domain}: Recalculating from predictions...")
                                rmse, rmse_v, rmse_a = calculate_rmse_from_predictions(pred_file, test_file)
                                if rmse_v is not None and rmse_a is not None:
                                    data['best_valence_rmse'] = float(rmse_v)
                                    data['best_arousal_rmse'] = float(rmse_a)
                                    # Update best_rmse if needed
                                    if data.get('best_rmse') is None:
                                        data['best_rmse'] = float(rmse)
                                    
                                    with open(json_file, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    updated_count += 1
                                    print(f"     ✅ Updated: rmse_v={rmse_v:.4f}, rmse_a={rmse_a:.4f}")
                                else:
                                    print(f"     ❌ Could not calculate RMSE")
                            else:
                                print(f"  ⚠️  {model}/{lang}/{domain}: Test file not found")
                        else:
                            print(f"  ⚠️  {model}/{lang}/{domain}: Prediction file not found")
                except Exception as e:
                    print(f"  ❌ Error processing {json_file}: {e}")
    
    print(f"\n✅ Updated {updated_count} Experiment 3 results")

if __name__ == "__main__":
    print("="*90)
    print("RECALCULATING MISSING RMSE VALUES")
    print("="*90)
    
    update_exp1_results()
    update_exp3_results()
    
    print("\n" + "="*90)
    print("✅ DONE! Now run create_separate_results.py to update the markdown file")
    print("="*90)
