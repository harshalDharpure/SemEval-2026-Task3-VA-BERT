#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning Script for VA-RoBERTa LARGE Model
Fine-tunes pretrained LARGE model on language-specific data and generates predictions
Separate file for LARGE model only
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
import numpy as np
import csv
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Import model from pretraining - LARGE model specific
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import model classes from LARGE pretraining script
from pretraining.pretrain_large import VARoBERTaModel, VADataset, process_data, load_jsonl, set_seed

# Import LARGE model config
from config_large import CFG_LARGE as CFG
CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(CFG.seed)

def train_epoch(model, dataloader, criterion, optimizer, device, gradient_accumulation_steps=1, scaler=None, use_bfloat16=False):
    """Train for one epoch with gradient accumulation and mixed precision"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    use_amp = scaler is not None
    
    for step, batch in enumerate(tqdm(dataloader, desc="Fine-tuning")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        valence = batch['valence'].to(device)
        arousal = batch['arousal'].to(device)
        
        if use_amp:
            # Use bfloat16 if configured, otherwise float16
            autocast_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
        pred_valence, pred_arousal = model(input_ids, attention_mask)
                loss_v = criterion(pred_valence, valence)
                loss_a = criterion(pred_arousal, arousal)
                loss = (loss_v + loss_a) / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            pred_valence, pred_arousal = model(input_ids, attention_mask)
        loss_v = criterion(pred_valence, valence)
        loss_a = criterion(pred_arousal, arousal)
        loss = (loss_v + loss_a) / gradient_accumulation_steps
        
        loss.backward()
        total_loss += loss.item() * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Handle remaining gradients if any
    if len(dataloader) % gradient_accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_valence_pred = []
    all_valence_true = []
    all_arousal_pred = []
    all_arousal_true = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            valence = batch['valence'].to(device)
            arousal = batch['arousal'].to(device)
            
            pred_valence, pred_arousal = model(input_ids, attention_mask)
            
            loss_v = criterion(pred_valence, valence)
            loss_a = criterion(pred_arousal, arousal)
            loss = loss_v + loss_a
            
            total_loss += loss.item()
            
            all_valence_pred.extend(pred_valence.cpu().numpy())
            all_valence_true.extend(valence.cpu().numpy())
            all_arousal_pred.extend(pred_arousal.cpu().numpy())
            all_arousal_true.extend(arousal.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    rmse_v = np.sqrt(mean_squared_error(all_valence_true, all_valence_pred))
    rmse_a = np.sqrt(mean_squared_error(all_arousal_true, all_arousal_pred))
    rmse = (rmse_v + rmse_a) / 2
    
    return avg_loss, rmse, rmse_v, rmse_a

def predict(model, dataloader, device):
    """Generate predictions"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            pred_valence, pred_arousal = model(input_ids, attention_mask)
            
            for v, a in zip(pred_valence.cpu().numpy(), pred_arousal.cpu().numpy()):
                predictions.append({
                    'valence': float(v),
                    'arousal': float(a)
                })
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Fine-tune VA-RoBERTa on English data')
    parser.add_argument('--pretrained_model_dir', type=str, default=None, help='Directory with pretrained model (use "NONE" to skip pretraining)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with training data')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Directory with test data')
    parser.add_argument('--output_dir', type=str, default='./finetuning/models', help='Output directory')
    parser.add_argument('--lang', type=str, default='eng', help='Language to fine-tune on')
    parser.add_argument('--domains', nargs='+', default=['restaurant', 'laptop'], help='Domains to fine-tune on')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', help='Base model name')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only run inference (for experiment 2)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, 'subtask_1')
    os.makedirs(pred_dir, exist_ok=True)
    
    # Load model (with or without pretraining)
    print("Loading model...")
    base_model_name = args.model_name
    
    # Check if pretrained model should be used
    use_pretrained = False
    model_path = None
    
    if args.pretrained_model_dir and args.pretrained_model_dir.upper() != 'NONE':
        # Check for .pth file directly (our pretrained model format)
        model_path = os.path.join(args.pretrained_model_dir, "va_roberta_multilingual_best.pth")
        if os.path.exists(model_path):
            use_pretrained = True
            print(f" Found pretrained model: {model_path}")
        else:
            # Try HuggingFace format as fallback
            try:
                config = AutoConfig.from_pretrained(args.pretrained_model_dir)
                base_model_name = config.name_or_path if hasattr(config, 'name_or_path') else args.model_name
                use_pretrained = True
                print(f" Found HuggingFace model: {args.pretrained_model_dir}")
            except Exception:
                print(f"  Pretrained model not found in: {args.pretrained_model_dir}")
                print("   Using base model instead")
                use_pretrained = False
    
    # Using LARGE model config (already imported)
    print(f"  Using LARGE model config: {CFG.model_name}")
    print(f"  Batch size: {CFG.batch_size}, Bfloat16: {CFG.use_bfloat16}, Patience: {CFG.early_stopping_patience}")
    
    model = VARoBERTaModel(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if use_pretrained and model_path and os.path.exists(model_path):
        # Load .pth checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=CFG.device)
            # Handle both state_dict and full checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f" Loaded pretrained model weights from: {model_path}")
        except Exception as e:
            print(f"  Error loading pretrained model: {e}")
            print("   Using base model instead")
    elif use_pretrained:
        print("  Pretrained model path not found, using base model")
    else:
        print(" Using base model (no pretraining)")
    
    model = model.to(CFG.device)
    # Use mixed precision training with bfloat16 or float16
    if CFG.use_bfloat16 and CFG.device.type == 'cuda':
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("  ✅ Using mixed precision training (bfloat16)")
    elif CFG.use_float16 and CFG.device.type == 'cuda':
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("  ✅ Using mixed precision training (float16)")
    else:
        scaler = None
        print("  Using full precision (float32)")
    
    # Fine-tune on each domain
    for domain in args.domains:
        print(f"\n{'='*60}")
        print(f"Fine-tuning on {args.lang}_{domain}")
        print(f"{'='*60}")
        
        # Load training data
        train_file_patterns = [
            os.path.join(args.data_dir, args.lang, f"{args.lang}_{domain}_train_alltasks_train_80.jsonl"),
            os.path.join(args.data_dir, args.lang, f"{args.lang}_{domain}_train_task1_train_80.jsonl")
        ]
        train_file = None
        for pattern in train_file_patterns:
            if os.path.exists(pattern):
                train_file = pattern
                break
        
        if train_file is None:
            print(f"  Training file not found for {args.lang}_{domain}")
            continue
        
        # Load test data
        test_file_patterns = [
            os.path.join(args.test_data_dir, args.lang, f"{args.lang}_{domain}_train_alltasks_test_20.jsonl"),
            os.path.join(args.test_data_dir, args.lang, f"{args.lang}_{domain}_train_task1_test_20.jsonl")
        ]
        test_file = None
        for pattern in test_file_patterns:
            if os.path.exists(pattern):
                test_file = pattern
                break
        
        if test_file is None:
            print(f"  Test file not found for {args.lang}_{domain}")
            continue
        
        # Process data
        train_jsonl = load_jsonl(train_file)
        train_data = process_data(train_jsonl)
        
        test_jsonl = load_jsonl(test_file)
        test_data = process_data(test_jsonl)
        
        # Split train/val
        split_idx = int(0.9 * len(train_data))
        train_split = train_data[:split_idx]
        val_split = train_data[split_idx:]
        
        # Create datasets
        train_dataset = VADataset(train_split, tokenizer, CFG.max_len)
        val_dataset = VADataset(val_split, tokenizer, CFG.max_len)
        test_dataset = VADataset(test_data, tokenizer, CFG.max_len)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
        
        # Initialize results storage (needed for both training and inference-only modes)
        results_dir = os.path.join(args.output_dir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        predictions_file = os.path.join(results_dir, f'predictions_{args.lang}_{domain}_{timestamp}.jsonl')
        
        # Skip training if --skip_training flag is set (for experiment 2)
        if args.skip_training:
            print("  Skipping training (inference only mode)")
            print("   Loading pretrained model...")
            # For inference-only, load pretrained model
            model_path = os.path.join(args.pretrained_model_dir, "va_roberta_multilingual_best.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=CFG.device))
                print(f" Loaded pretrained model: {model_path}")
            else:
                print(f" Warning: Pretrained model not found: {model_path}")
                print("   Using base model without pretrained weights")
        else:
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
            
            # Results files (results_dir and predictions_file already defined above)
            results_json = os.path.join(results_dir, f'finetune_{args.lang}_{domain}_results_{timestamp}.json')
            results_csv = os.path.join(results_dir, f'finetune_{args.lang}_{domain}_metrics_{timestamp}.csv')
            results_log = os.path.join(results_dir, f'finetune_{args.lang}_{domain}_log_{timestamp}.txt')
            config_file = os.path.join(results_dir, f'finetune_{args.lang}_{domain}_config_{timestamp}.json')
            predictions_file = os.path.join(results_dir, f'predictions_{args.lang}_{domain}_{timestamp}.jsonl')
            
            # Save training configuration
            config = {
                'model_name': f"va_roberta_{args.lang}_{domain}",
                'pretrained_model': args.pretrained_model_dir,
                'language': args.lang,
                'domain': domain,
                'epochs': CFG.epochs,
                'batch_size': CFG.batch_size,
                'learning_rate': CFG.lr,
                'max_len': CFG.max_len,
                'dropout': CFG.dropout,
                'fc_dropout': CFG.fc_dropout,
                'early_stopping_patience': CFG.early_stopping_patience,
                'train_samples': len(train_split),
                'val_samples': len(val_split),
                'test_samples': len(test_data),
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Initialize CSV file
            with open(results_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_RMSE', 'Is_Best', 'Early_Stopped', 'Timestamp'])
            
            # Initialize results dictionary
            training_results = {
                'config': config,
                'epochs_data': []
            }
            
            # Training loop with early stopping (skip if inference only)
            if args.skip_training:
                # Create minimal results for inference-only mode
                training_results = {
                    'config': config,
                    'epochs_data': [],
                    'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'final_best_rmse': 'N/A (inference only)',
                    'best_epoch': 0,
                    'total_epochs_trained': 0,
                    'early_stopped': False,
                    'mode': 'inference_only'
                }
                with open(results_json, 'w', encoding='utf-8') as f:
                    json.dump(training_results, f, indent=2, ensure_ascii=False)
                print(" Inference-only mode: Skipping training")
            else:
                # Training loop with early stopping
                best_rmse = float('inf')
                patience_counter = 0
                early_stopped = False
                
                print(f"Results will be saved to: {results_dir}")
                
                # Open log file
                with open(results_log, 'w', encoding='utf-8') as log_file:
                    log_file.write(f"Fine-tuning Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"{'='*60}\n")
                    log_file.write(f"Language: {args.lang}, Domain: {domain}\n")
                    log_file.write(f"Pretrained model: {args.pretrained_model_dir}\n")
                    log_file.write(f"Train samples: {len(train_split)}, Val samples: {len(val_split)}\n")
                    log_file.write(f"{'='*60}\n\n")
                    
                    for epoch in range(CFG.epochs):
                        epoch_start = datetime.now()
                        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
                        log_file.write(f"\nEpoch {epoch+1}/{CFG.epochs} - {epoch_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        
                        train_loss = train_epoch(model, train_loader, criterion, optimizer, CFG.device, CFG.gradient_accumulation_steps, scaler, CFG.use_bfloat16)
                        val_loss, val_rmse, val_rmse_v, val_rmse_a = validate(model, val_loader, criterion, CFG.device)
                        
                        epoch_time = (datetime.now() - epoch_start).total_seconds()
                        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f} (V: {val_rmse_v:.4f}, A: {val_rmse_a:.4f}) (Time: {epoch_time:.2f}s)")
                        log_file.write(f"  Train Loss: {train_loss:.4f}\n")
                        log_file.write(f"  Val Loss: {val_loss:.4f}\n")
                        log_file.write(f"  Val RMSE: {val_rmse:.4f} (Valence: {val_rmse_v:.4f}, Arousal: {val_rmse_a:.4f})\n")
                        log_file.write(f"  Time: {epoch_time:.2f}s\n")
                        
                        # Prepare epoch results
                        epoch_result = {
                            'epoch': epoch + 1,
                            'train_loss': float(train_loss),
                            'val_loss': float(val_loss),
                            'val_rmse': float(val_rmse),
                            'val_rmse_v': float(val_rmse_v),
                            'val_rmse_a': float(val_rmse_a),
                            'is_best': False,
                            'timestamp': epoch_start.strftime('%Y-%m-%d %H:%M:%S'),
                            'epoch_time_seconds': epoch_time
                        }
                        
                        # Save best model
                        if val_rmse < best_rmse:
                            best_rmse = val_rmse
                            patience_counter = 0
                            epoch_result['is_best'] = True
                            best_model_path = os.path.join(args.output_dir, f"va_roberta_{args.lang}_{domain}_best.pth")
                            torch.save(model.state_dict(), best_model_path)
                            print(f" Saved best model (RMSE: {best_rmse:.4f})")
                            log_file.write(f"   Best model saved (RMSE: {best_rmse:.4f})\n")
                        else:
                            patience_counter += 1
                            log_file.write(f"  Patience counter: {patience_counter}/{CFG.early_stopping_patience}\n")
                            if patience_counter >= CFG.early_stopping_patience:
                                early_stopped = True
                                print(f"Early stopping at epoch {epoch+1}")
                                log_file.write(f"    Early stopping triggered\n")
                                break
                        
                        # Add epoch result to training results
                        training_results['epochs_data'].append(epoch_result)
                        
                        # Append to CSV
                        with open(results_csv, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                epoch + 1, train_loss, val_loss, val_rmse,
                                'Yes' if epoch_result['is_best'] else 'No',
                                'Yes' if early_stopped else 'No',
                                epoch_start.strftime('%Y-%m-%d %H:%M:%S')
                            ])
                        
                        # Save JSON after each epoch
                        training_results['best_rmse'] = float(best_rmse)
                        training_results['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(results_json, 'w', encoding='utf-8') as f:
                            json.dump(training_results, f, indent=2, ensure_ascii=False)
                        print(f'Results saved to separate files')
                    
                    # Final training summary - get valence and arousal RMSE from best epoch
                    best_epoch_data = next((e for e in training_results['epochs_data'] if e.get('is_best', False)), None)
                    if best_epoch_data:
                        training_results['best_valence_rmse'] = float(best_epoch_data.get('val_rmse_v', 0))
                        training_results['best_arousal_rmse'] = float(best_epoch_data.get('val_rmse_a', 0))
                    training_results['training_completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    training_results['final_best_rmse'] = float(best_rmse)
                    training_results['best_epoch'] = max([ep['epoch'] for ep in training_results['epochs_data'] if ep['is_best']], default=0)
                    training_results['total_epochs_trained'] = len(training_results['epochs_data'])
                    training_results['early_stopped'] = early_stopped
                    
                    with open(results_json, 'w', encoding='utf-8') as f:
                        json.dump(training_results, f, indent=2, ensure_ascii=False)
                    
                    log_file.write(f"\n{'='*60}\n")
                    log_file.write(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"Best RMSE: {best_rmse:.4f} at epoch {training_results['best_epoch']}\n")
                    log_file.write(f"Total epochs: {len(training_results['epochs_data'])}\n")
                    log_file.write(f"Early stopped: {early_stopped}\n")
                    log_file.write(f"{'='*60}\n")
                
                print(f'\nTraining completed. Results saved to separate files:')
                print(f'  - JSON: {results_json}')
                print(f'  - CSV: {results_csv}')
                print(f'  - Log: {results_log}')
                print(f'  - Config: {config_file}')
                if not args.skip_training:
                    print(f'Best RMSE: {best_rmse:.4f} at epoch {training_results["best_epoch"]}')
                
                # Load best model for prediction (if training was done)
                if not args.skip_training:
                    best_model_path = os.path.join(args.output_dir, f"va_roberta_{args.lang}_{domain}_best.pth")
                    if os.path.exists(best_model_path):
                        model.load_state_dict(torch.load(best_model_path, map_location=CFG.device))
                        print(f" Loaded best model for prediction")
        
        # Generate predictions
        print(f"\nGenerating predictions for: {test_file}")
        predictions = predict(model, test_loader, CFG.device)
        
        # Extract aspects from test data (handle both Aspect_VA and Quadruplet formats)
        test_aspects = []
        for record in test_jsonl:
            aspects_in_record = []
            # Handle Aspect_VA format
            if 'Aspect_VA' in record:
                for aspect_va in record.get('Aspect_VA', []):
                    aspects_in_record.append(aspect_va['Aspect'])
            # Handle Quadruplet format
            elif 'Quadruplet' in record:
                for quad in record.get('Quadruplet', []):
                    aspect = quad.get('Aspect', 'NULL')
                    if aspect != 'NULL':
                        aspects_in_record.append(aspect)
            # Handle Triplet format
            elif 'Triplet' in record:
                for triplet in record.get('Triplet', []):
                    aspect = triplet.get('Aspect', 'NULL')
                    if aspect != 'NULL':
                        aspects_in_record.append(aspect)
            test_aspects.append(aspects_in_record)
        
        # Save predictions in JSONL format (both in predictions dir and results dir)
        output_file = os.path.join(pred_dir, f"pred_{args.lang}_{domain}.jsonl")
        pred_idx = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, record in enumerate(test_jsonl):
                aspect_va_list = []
                for aspect in test_aspects[i]:
                    if pred_idx < len(predictions):
                        pred = predictions[pred_idx]
                        aspect_va_list.append({
                            'Aspect': aspect,
                            'VA': f"{pred['valence']:.2f}#{pred['arousal']:.2f}"
                        })
                        pred_idx += 1
                
                output_record = {
                    'ID': record['ID'],
                    'Text': record['Text'],
                    'Aspect_VA': aspect_va_list
                }
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        
        # Also save predictions to results directory with timestamp
        pred_idx = 0
        with open(predictions_file, 'w', encoding='utf-8') as f:
            for i, record in enumerate(test_jsonl):
                aspect_va_list = []
                for aspect in test_aspects[i]:
                    if pred_idx < len(predictions):
                        pred = predictions[pred_idx]
                        aspect_va_list.append({
                            'Aspect': aspect,
                            'VA': f"{pred['valence']:.2f}#{pred['arousal']:.2f}"
                        })
                        pred_idx += 1
                output_record = {
                    'ID': record['ID'],
                    'Text': record['Text'],
                    'Aspect_VA': aspect_va_list
                }
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        
        print(f" Predictions saved:")
        print(f"  - {output_file} ({len(predictions)} records)")
        print(f"  - {predictions_file} (timestamped copy)")
    
    print(f"\n{'='*60}")
    print(f" Fine-tuning completed!")
    print(f"Predictions saved to: {pred_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
