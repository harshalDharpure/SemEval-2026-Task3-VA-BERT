#!/usr/bin/env python3
"""
Create comprehensive submission by copying existing predictions from Experiment 3
Uses pretraining + fine-tuning base model predictions that already exist
"""
import os
import shutil
import zipfile
from pathlib import Path

# Define all language-domain combinations and their experiment paths
LANGUAGE_DOMAIN_MAPPING = {
    'eng_restaurant': 'experiments/exp3_pretrain_finetune_base_language_specific_eng_finetune/subtask_1/pred_eng_restaurant.jsonl',
    'eng_laptop': 'experiments/exp3_pretrain_finetune_base_language_specific_eng_finetune/subtask_1/pred_eng_laptop.jsonl',
    'rus_restaurant': 'experiments/exp3_pretrain_finetune_base_language_specific_rus_finetune/subtask_1/pred_rus_restaurant.jsonl',
    'ukr_restaurant': 'experiments/exp3_pretrain_finetune_base_language_specific_ukr_finetune/subtask_1/pred_ukr_restaurant.jsonl',
    'tat_restaurant': 'experiments/exp3_pretrain_finetune_base_language_specific_tat_finetune/subtask_1/pred_tat_restaurant.jsonl',
    'zho_restaurant': 'experiments/exp3_pretrain_finetune_base_language_specific_zho_finetune/subtask_1/pred_zho_restaurant.jsonl',
    'zho_laptop': 'experiments/exp3_pretrain_finetune_base_language_specific_zho_finetune/subtask_1/pred_zho_laptop.jsonl',
    'zho_finance': 'experiments/exp3_pretrain_finetune_base_language_specific_zho_finetune/subtask_1/pred_zho_finance.jsonl',
    'jpn_hotel': 'experiments/exp3_pretrain_finetune_base_language_specific_jpn_finetune/subtask_1/pred_jpn_hotel.jsonl',
    'jpn_finance': 'experiments/exp3_pretrain_finetune_base_language_specific_jpn_finetune/subtask_1/pred_jpn_finance.jsonl',
}

def main():
    output_dir = 'submission_comprehensive'
    subtask_dir = os.path.join(output_dir, 'subtask_1')
    
    # Create output directory
    os.makedirs(subtask_dir, exist_ok=True)
    
    print("="*60)
    print("Creating Comprehensive Submission from Experiment 3 Results")
    print("="*60)
    
    copied_files = []
    missing_files = []
    
    # Copy all prediction files
    for lang_domain, source_path in LANGUAGE_DOMAIN_MAPPING.items():
        lang, domain = lang_domain.split('_', 1)
        target_file = os.path.join(subtask_dir, f"pred_{lang}_{domain}.jsonl")
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_file)
            file_size = os.path.getsize(target_file)
            print(f"✓ Copied: {lang}_{domain} ({file_size:,} bytes)")
            copied_files.append(f"pred_{lang}_{domain}.jsonl")
        else:
            print(f"✗ Missing: {lang}_{domain} (source: {source_path})")
            missing_files.append(f"{lang}_{domain}")
    
    # Create zip file
    zip_path = os.path.join(output_dir, 'subtask_1.zip')
    print(f"\n{'='*60}")
    print(f"Creating submission zip: {zip_path}")
    print(f"{'='*60}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in copied_files:
            file_path = os.path.join(subtask_dir, file)
            if os.path.exists(file_path):
                zf.write(file_path, f"subtask_1/{file}")
                print(f"  Added: subtask_1/{file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully copied: {len(copied_files)} files")
    for f in copied_files:
        print(f"  ✓ {f}")
    
    if missing_files:
        print(f"\nMissing files: {len(missing_files)}")
        for f in missing_files:
            print(f"  ✗ {f}")
    
    print(f"\n{'='*60}")
    print(f"Submission zip created: {zip_path}")
    print(f"Location: {os.path.abspath(zip_path)}")
    print(f"Ready for leaderboard submission!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
