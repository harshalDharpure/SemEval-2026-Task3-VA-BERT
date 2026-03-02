#!/usr/bin/env python3
"""
Generate predictions for official test files using Experiment 3 (Pretraining + Fine-tuning) base models
Uses the official test files from DimABSA2026/task-dataset/track_a/subtask_1
"""
import os
import json
import torch
import zipfile
from transformers import AutoTokenizer
import sys
sys.path.append('pretraining')
from pretrain import VARoBERTaModel, VADataset, load_jsonl
from torch.utils.data import DataLoader

# Official test file directory
OFFICIAL_TEST_DIR = "/DATA/vaneet_2221cs15/demo/DimABSA2026/task-dataset/track_a/subtask_1"

# Language-domain combinations and their model paths
LANGUAGE_DOMAIN_CONFIG = {
    'eng_restaurant': {
        'test_file': f"{OFFICIAL_TEST_DIR}/eng/eng_restaurant_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_eng_finetune/va_roberta_eng_restaurant_best.pth',
        'output_file': 'pred_eng_restaurant.jsonl'
    },
    'eng_laptop': {
        'test_file': f"{OFFICIAL_TEST_DIR}/eng/eng_laptop_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_eng_finetune/va_roberta_eng_laptop_best.pth',
        'output_file': 'pred_eng_laptop.jsonl'
    },
    'rus_restaurant': {
        'test_file': f"{OFFICIAL_TEST_DIR}/rus/rus_restaurant_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_rus_finetune/va_roberta_rus_restaurant_best.pth',
        'output_file': 'pred_rus_restaurant.jsonl'
    },
    'ukr_restaurant': {
        'test_file': f"{OFFICIAL_TEST_DIR}/ukr/ukr_restaurant_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_ukr_finetune/va_roberta_ukr_restaurant_best.pth',
        'output_file': 'pred_ukr_restaurant.jsonl'
    },
    'tat_restaurant': {
        'test_file': f"{OFFICIAL_TEST_DIR}/tat/tat_restaurant_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_tat_finetune/va_roberta_tat_restaurant_best.pth',
        'output_file': 'pred_tat_restaurant.jsonl'
    },
    'zho_restaurant': {
        'test_file': f"{OFFICIAL_TEST_DIR}/zho/zho_restaurant_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_zho_finetune/va_roberta_zho_restaurant_best.pth',
        'output_file': 'pred_zho_restaurant.jsonl'
    },
    'zho_laptop': {
        'test_file': f"{OFFICIAL_TEST_DIR}/zho/zho_laptop_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_zho_finetune/va_roberta_zho_laptop_best.pth',
        'output_file': 'pred_zho_laptop.jsonl'
    },
    'zho_finance': {
        'test_file': f"{OFFICIAL_TEST_DIR}/zho/zho_finance_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_zho_finetune/va_roberta_zho_finance_best.pth',
        'output_file': 'pred_zho_finance.jsonl'
    },
    'jpn_hotel': {
        'test_file': f"{OFFICIAL_TEST_DIR}/jpn/jpn_hotel_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_jpn_finetune/va_roberta_jpn_hotel_best.pth',
        'output_file': 'pred_jpn_hotel.jsonl'
    },
    'jpn_finance': {
        'test_file': f"{OFFICIAL_TEST_DIR}/jpn/jpn_finance_test_task1.jsonl",
        'model_path': 'experiments/exp3_pretrain_finetune_base_language_specific_jpn_finetune/va_roberta_jpn_finance_best.pth',
        'output_file': 'pred_jpn_finance.jsonl'
    },
}

def generate_predictions_for_file(model, tokenizer, test_file, output_file, device, max_len=192, batch_size=4):
    """Generate predictions for a test file"""
    print(f"\n{'='*60}")
    print(f"Processing: {test_file}")
    print(f"{'='*60}")
    
    # Load test data
    if not os.path.exists(test_file):
        print(f"  ERROR: Test file not found: {test_file}")
        return False
    
    test_data = load_jsonl(test_file)
    print(f"  Loaded {len(test_data)} test records")
    
    # Process data - extract aspects
    processed_data = []
    for record in test_data:
        text = record['Text']
        # Official test format: 'Aspect' is a direct list
        aspects = record.get('Aspect', [])
        if not isinstance(aspects, list):
            aspects = [aspects] if aspects else []
        
        # Create one entry per aspect
        for aspect in aspects:
            processed_data.append({
                'Text': text,
                'Aspect': aspect,
                'ID': record['ID'],
                'Valence': 5.0,  # Dummy value for inference (not used)
                'Arousal': 5.0   # Dummy value for inference (not used)
            })
    
    if not processed_data:
        print(f"  Warning: No aspects found in {test_file}")
        # Create empty predictions file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in test_data:
                f.write(json.dumps({
                    'ID': record['ID'],
                    'Text': record['Text'],
                    'Aspect_VA': []
                }, ensure_ascii=False) + '\n')
        print(f"  Created empty predictions file: {output_file}")
        return True
    
    print(f"  Found {len(processed_data)} aspect-text pairs")
    
    # Create dataset
    dataset = VADataset(processed_data, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Generate predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            pred_valence, pred_arousal = model(input_ids, attention_mask)
            
            for v, a in zip(pred_valence.cpu().numpy(), pred_arousal.cpu().numpy()):
                # Clamp to [1.0, 9.0] range
                v = max(1.0, min(9.0, float(v)))
                a = max(1.0, min(9.0, float(a)))
                predictions.append({
                    'valence': v,
                    'arousal': a
                })
    
    print(f"  Generated {len(predictions)} predictions")
    
    # Group predictions by ID
    pred_idx = 0
    output_data = {}
    for record in test_data:
        record_id = record['ID']
        text = record['Text']
        
        # Extract aspects
        aspects = record.get('Aspect', [])
        if not isinstance(aspects, list):
            aspects = [aspects] if aspects else []
        
        aspect_va_list = []
        for aspect in aspects:
            if pred_idx < len(predictions):
                pred = predictions[pred_idx]
                aspect_va_list.append({
                    'Aspect': aspect,
                    'VA': f"{pred['valence']:.2f}#{pred['arousal']:.2f}"
                })
                pred_idx += 1
        
        output_data[record_id] = {
            'ID': record_id,
            'Text': text,
            'Aspect_VA': aspect_va_list
        }
    
    # Write output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in test_data:
            record_id = record['ID']
            if record_id in output_data:
                f.write(json.dumps(output_data[record_id], ensure_ascii=False) + '\n')
            else:
                # Empty aspect list
                f.write(json.dumps({
                    'ID': record_id,
                    'Text': record['Text'],
                    'Aspect_VA': []
                }, ensure_ascii=False) + '\n')
    
    file_size = os.path.getsize(output_file)
    print(f"  ✓ Generated: {output_file} ({file_size:,} bytes, {len(output_data)} records)")
    return True

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'xlm-roberta-base'
    max_len = 192
    batch_size = 4
    
    print("="*60)
    print("Generating Predictions from Official Test Files")
    print("Experiment 3: Pretraining + Fine-tuning (Base)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print("="*60)
    
    output_dir = 'submission_official_test'
    subtask_dir = os.path.join(output_dir, 'subtask_1')
    os.makedirs(subtask_dir, exist_ok=True)
    
    successful = []
    failed = []
    
    # Process each language-domain combination
    for lang_domain, config in LANGUAGE_DOMAIN_CONFIG.items():
        test_file = config['test_file']
        model_path = config['model_path']
        output_file = os.path.join(subtask_dir, config['output_file'])
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"\n✗ SKIPPED: {lang_domain} - Model not found: {model_path}")
            failed.append(lang_domain)
            continue
        
        # Check if test file exists
        if not os.path.exists(test_file):
            print(f"\n✗ SKIPPED: {lang_domain} - Test file not found: {test_file}")
            failed.append(lang_domain)
            continue
        
        try:
            # Load model
            print(f"\nLoading model: {model_path}")
            model = VARoBERTaModel(model_name)
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model = model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Generate predictions
            success = generate_predictions_for_file(
                model, tokenizer, test_file, output_file, 
                device, max_len=max_len, batch_size=batch_size
            )
            
            if success:
                successful.append(lang_domain)
            else:
                failed.append(lang_domain)
            
            # Clear memory
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"\n✗ ERROR processing {lang_domain}: {str(e)}")
            failed.append(lang_domain)
            import traceback
            traceback.print_exc()
    
    # Create zip file
    print(f"\n{'='*60}")
    print("Creating submission zip file")
    print(f"{'='*60}")
    
    zip_path = os.path.join(output_dir, 'subtask_1.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for lang_domain in successful:
            config = LANGUAGE_DOMAIN_CONFIG[lang_domain]
            output_file = os.path.join(subtask_dir, config['output_file'])
            if os.path.exists(output_file):
                zf.write(output_file, f"subtask_1/{config['output_file']}")
                print(f"  ✓ Added: subtask_1/{config['output_file']}")
    
    zip_size = os.path.getsize(zip_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful)}/{len(LANGUAGE_DOMAIN_CONFIG)}")
    for lang_domain in successful:
        print(f"  ✓ {lang_domain}")
    
    if failed:
        print(f"\nFailed/Skipped: {len(failed)}")
        for lang_domain in failed:
            print(f"  ✗ {lang_domain}")
    
    print(f"\n{'='*60}")
    print(f"Submission zip created: {zip_path}")
    print(f"Size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
    print(f"Location: {os.path.abspath(zip_path)}")
    print(f"Ready for leaderboard submission!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
