#!/usr/bin/env python3
"""
Generate comprehensive submission file with predictions for ALL languages and domains
Uses Experiment 3 (Pretraining + Fine-tuning) base models
"""
import os
import json
import torch
import argparse
import zipfile
from transformers import AutoTokenizer
import sys
sys.path.append('pretraining')
from pretrain import VARoBERTaModel, VADataset, load_jsonl
from torch.utils.data import DataLoader

# Define all language-domain combinations based on available data
LANGUAGE_DOMAIN_COMBINATIONS = {
    'eng': ['restaurant', 'laptop'],
    'rus': ['restaurant'],
    'ukr': ['restaurant'],
    'tat': ['restaurant'],
    'zho': ['restaurant', 'laptop', 'finance'],
    'jpn': ['hotel', 'finance']
}

def find_best_model(experiment_dir, lang, domain):
    """Find the best model checkpoint in experiment directory"""
    # Try specific model name first (most common pattern)
    specific_path = os.path.join(experiment_dir, f'va_roberta_{lang}_{domain}_best.pth')
    if os.path.exists(specific_path):
        return specific_path
    
    # Try other possible paths
    possible_paths = [
        os.path.join(experiment_dir, 'best_model.pth'),
        os.path.join(experiment_dir, 'va_roberta_multilingual_best.pth'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try to find any .pth file with 'best' in name
    if os.path.exists(experiment_dir):
        for file in os.listdir(experiment_dir):
            if file.endswith('.pth') and 'best' in file.lower():
                return os.path.join(experiment_dir, file)
    
    return None

def generate_predictions(model, tokenizer, test_file, device, max_len=192, batch_size=4):
    """Generate predictions for a test file"""
    print(f"  Processing: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"  Warning: Test file not found: {test_file}")
        return None
    
    # Load test data
    test_data = load_jsonl(test_file)
    print(f"  Loaded {len(test_data)} records")
    
    # Process data - extract aspects
    processed_data = []
    for record in test_data:
        text = record.get('Text', '')
        record_id = record.get('ID', '')
        
        # Extract aspects from various formats
        aspects = []
        if 'Aspect' in record and isinstance(record['Aspect'], list):
            aspects = record['Aspect']
        elif 'Aspect_VA' in record:
            aspects = [item['Aspect'] for item in record.get('Aspect_VA', [])]
        elif 'Quadruplet' in record:
            aspects = [item['Aspect'] for item in record.get('Quadruplet', []) if item.get('Aspect') != 'NULL']
        elif 'Triplet' in record:
            aspects = [item['Aspect'] for item in record.get('Triplet', []) if item.get('Aspect') != 'NULL']
        
        # Create one entry per aspect
        for aspect in aspects:
            processed_data.append({
                'Text': text,
                'Aspect': aspect,
                'ID': record_id,
                'Valence': 5.0,  # Dummy value for inference
                'Arousal': 5.0
            })
    
    if not processed_data:
        print(f"  Warning: No aspects found in {test_file}")
        # Return empty predictions
        output_data = {}
        for record in test_data:
            output_data[record.get('ID', '')] = {
                'ID': record.get('ID', ''),
                'Text': record.get('Text', ''),
                'Aspect_VA': []
            }
        return output_data
    
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
    
    # Group predictions by ID
    pred_idx = 0
    output_data = {}
    for record in test_data:
        record_id = record.get('ID', '')
        text = record.get('Text', '')
        
        # Extract aspects
        aspects = []
        if 'Aspect' in record and isinstance(record['Aspect'], list):
            aspects = record['Aspect']
        elif 'Aspect_VA' in record:
            aspects = [item['Aspect'] for item in record.get('Aspect_VA', [])]
        elif 'Quadruplet' in record:
            aspects = [item['Aspect'] for item in record.get('Quadruplet', []) if item.get('Aspect') != 'NULL']
        elif 'Triplet' in record:
            aspects = [item['Aspect'] for item in record.get('Triplet', []) if item.get('Aspect') != 'NULL']
        
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
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive submission for all languages and domains')
    parser.add_argument('--test_data_dir', required=True, help='Directory containing official test files')
    parser.add_argument('--output_dir', default='submission_comprehensive', help='Output directory for submission files')
    parser.add_argument('--model_name', default='xlm-roberta-base', help='Base model name')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    subtask_dir = os.path.join(args.output_dir, 'subtask_1')
    os.makedirs(subtask_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Track statistics
    generated_files = []
    failed_combinations = []
    
    # Process each language-domain combination
    for lang, domains in LANGUAGE_DOMAIN_COMBINATIONS.items():
        for domain in domains:
            print(f"\n{'='*60}")
            print(f"Processing: {lang} - {domain}")
            print(f"{'='*60}")
            
            # Find model path
            exp_dir = f"experiments/exp3_pretrain_finetune_base_language_specific_{lang}_finetune"
            model_path = find_best_model(exp_dir, lang, domain)
            
            if not model_path or not os.path.exists(model_path):
                print(f"  Error: Model not found for {lang}-{domain}")
                print(f"  Searched in: {exp_dir}")
                failed_combinations.append(f"{lang}_{domain}")
                continue
            
            print(f"  Found model: {model_path}")
            
            # Load model
            model = VARoBERTaModel(args.model_name)
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model = model.to(device)
                model.eval()
                print(f"  Model loaded successfully")
            except Exception as e:
                print(f"  Error loading model: {e}")
                failed_combinations.append(f"{lang}_{domain}")
                continue
            
            # Find test file
            test_file = None
            possible_paths = [
                os.path.join(args.test_data_dir, lang, f"{lang}_{domain}_test_alltasks.jsonl"),
                os.path.join(args.test_data_dir, lang, f"{lang}_{domain}_test.jsonl"),
                os.path.join(args.test_data_dir, f"{lang}_{domain}_test_alltasks.jsonl"),
                os.path.join(args.test_data_dir, f"{lang}_{domain}_test.jsonl"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    test_file = path
                    break
            
            if not test_file:
                print(f"  Warning: Test file not found for {lang}-{domain}")
                failed_combinations.append(f"{lang}_{domain}")
                continue
            
            # Generate predictions
            predictions = generate_predictions(model, tokenizer, test_file, device, batch_size=args.batch_size)
            
            if predictions is None:
                failed_combinations.append(f"{lang}_{domain}")
                continue
            
            # Write output file
            output_file = os.path.join(subtask_dir, f"pred_{lang}_{domain}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for record_id in sorted(predictions.keys()):
                    f.write(json.dumps(predictions[record_id], ensure_ascii=False) + '\n')
            
            print(f"  Generated: {output_file} ({len(predictions)} records)")
            generated_files.append(f"pred_{lang}_{domain}.jsonl")
            
            # Clear model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create zip file
    zip_path = os.path.join(args.output_dir, 'subtask_1.zip')
    print(f"\n{'='*60}")
    print(f"Creating submission zip: {zip_path}")
    print(f"{'='*60}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in generated_files:
            file_path = os.path.join(subtask_dir, file)
            if os.path.exists(file_path):
                zf.write(file_path, f"subtask_1/{file}")
                print(f"  Added: subtask_1/{file}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Generated files: {len(generated_files)}")
    for f in generated_files:
        print(f"  ✓ {f}")
    
    if failed_combinations:
        print(f"\nFailed combinations: {len(failed_combinations)}")
        for f in failed_combinations:
            print(f"  ✗ {f}")
    
    print(f"\nSubmission zip created: {zip_path}")
    print(f"Ready for leaderboard submission!")

if __name__ == "__main__":
    main()
