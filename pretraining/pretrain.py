#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretraining Script for Multilingual VA-RoBERTa Model
Trains a multilingual Valence-Arousal prediction model on multiple languages
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

# Configuration - Will be imported from model-specific config files
# This is a base class, actual config imported from config_large.py or config_xl.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default config (will be overridden by model-specific configs)
class CFG:
    model_name = None  # Must be set from model-specific config
    max_len = 192
    batch_size = 4
    epochs = 5
    lr = 1.5e-5
    dropout = 0.1
    fc_dropout = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    use_float16 = True

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(CFG.seed)

# Dataset Class
class VADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=192):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['Text']
        aspect = item['Aspect']
        valence = item['Valence']
        arousal = item['Arousal']
        
        # Combine text and aspect
        input_text = f"{aspect} [SEP] {text}"
        
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'valence': torch.tensor(valence, dtype=torch.float),
            'arousal': torch.tensor(arousal, dtype=torch.float)
        }

# Model Class
class VARoBERTaModel(nn.Module):
    """VA-RoBERTa Model for Valence-Arousal prediction"""
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(CFG.dropout)
        self.fc_dropout = nn.Dropout(CFG.fc_dropout)
        
        # Mean pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Regression heads for Valence and Arousal
        self.fc = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        self.valence_head = nn.Linear(self.config.hidden_size, 1)
        self.arousal_head = nn.Linear(self.config.hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.valence_head.weight)
        nn.init.zeros_(self.valence_head.bias)
        nn.init.xavier_uniform_(self.arousal_head.weight)
        nn.init.zeros_(self.arousal_head.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask
        
        # Regression heads
        x = self.dropout(pooled)
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.fc_dropout(x)
        
        valence = self.valence_head(x).squeeze(-1)
        arousal = self.arousal_head(x).squeeze(-1)
        
        return valence, arousal

# Data Loading
def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def process_data(jsonl_data):
    """Process JSONL data into training format - handles both Aspect_VA and Quadruplet formats"""
    processed = []
    for record in jsonl_data:
        text = record['Text']
        
        # Handle Aspect_VA format (Task 1)
        if 'Aspect_VA' in record:
            for aspect_va in record.get('Aspect_VA', []):
                aspect = aspect_va['Aspect']
                va_str = aspect_va['VA']
                valence, arousal = map(float, va_str.split('#'))
                processed.append({
                    'Text': text,
                    'Aspect': aspect,
                    'Valence': valence,
                    'Arousal': arousal
                })
        
        # Handle Quadruplet format (Task 3) - extract Aspect and VA
        elif 'Quadruplet' in record:
            for quad in record.get('Quadruplet', []):
                aspect = quad.get('Aspect', 'NULL')
                if aspect != 'NULL':  # Skip NULL aspects
                    va_str = quad['VA']
                    valence, arousal = map(float, va_str.split('#'))
                    processed.append({
                        'Text': text,
                        'Aspect': aspect,
                        'Valence': valence,
                        'Arousal': arousal
                    })
        
        # Handle Triplet format (Task 2) - extract Aspect and VA
        elif 'Triplet' in record:
            for triplet in record.get('Triplet', []):
                aspect = triplet.get('Aspect', 'NULL')
                if aspect != 'NULL':  # Skip NULL aspects
                    va_str = triplet['VA']
                    valence, arousal = map(float, va_str.split('#'))
                    processed.append({
                        'Text': text,
                        'Aspect': aspect,
                        'Valence': valence,
                        'Arousal': arousal
                    })
    
    return processed

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    use_amp = scaler is not None
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        valence = batch['valence'].to(device)
        arousal = batch['arousal'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                pred_valence, pred_arousal = model(input_ids, attention_mask)
                loss_v = criterion(pred_valence, valence)
                loss_a = criterion(pred_arousal, arousal)
                loss = loss_v + loss_a
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_valence, pred_arousal = model(input_ids, attention_mask)
            loss_v = criterion(pred_valence, valence)
            loss_a = criterion(pred_arousal, arousal)
            loss = loss_v + loss_a
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
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
    
    return avg_loss, rmse

def main():
    parser = argparse.ArgumentParser(description='Pretrain Multilingual VA-RoBERTa')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data OR path to single JSONL file')
    parser.add_argument('--output_dir', type=str, default='./pretraining/models', help='Output directory')
    parser.add_argument('--languages', nargs='+', default=['eng', 'jpn', 'rus', 'ukr'], help='Languages to train on (English, Japanese, Russian, Ukrainian)')
    parser.add_argument('--domains', nargs='+', default=['restaurant'], help='Domains to train on')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', help='Base model name (xlm-roberta-base, xlm-roberta-large, xlm-roberta-xl)')
    parser.add_argument('--data_format', type=str, default='folder', choices=['folder', 'single_jsonl'], 
                       help='Data format: folder (language folders) or single_jsonl (single shuffled file)')
    
    args = parser.parse_args()
    
    # Update CFG
    CFG.model_name = args.model_name
    
    # Batch size is set to 4 for all models (no reduction needed)
    # Using float16 for memory efficiency
    print(f"Using batch_size: {CFG.batch_size} with float16 mixed precision")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading training data...")
    all_data = []
    
    # Check if data_dir is a single JSONL file
    if args.data_format == 'single_jsonl' or (os.path.isfile(args.data_dir) and args.data_dir.endswith('.jsonl')):
        # Load from single JSONL file
        data_file = args.data_dir if os.path.isfile(args.data_dir) else args.data_dir
        print(f"Loading from single JSONL file: {data_file}...")
        jsonl_data = load_jsonl(data_file)
        processed = process_data(jsonl_data)
        all_data.extend(processed)
        print(f"Loaded {len(processed)} samples from single file")
    else:
        # Load from language folders (original format)
        for lang in args.languages:
            for domain in args.domains:
                # Try multiple file patterns
                data_file_patterns = [
                    os.path.join(args.data_dir, lang, f"{lang}_{domain}_train_alltasks_train_80.jsonl"),
                    os.path.join(args.data_dir, lang, f"{lang}_{domain}_train_task1_train_80.jsonl")
                ]
                data_file = None
                for pattern in data_file_patterns:
                    if os.path.exists(pattern):
                        data_file = pattern
                        break
                
                if data_file and os.path.exists(data_file):
                    print(f"Loading {data_file}...")
                    jsonl_data = load_jsonl(data_file)
                    processed = process_data(jsonl_data)
                    all_data.extend(processed)
                else:
                    print(f"Warning: No data file found for {lang}_{domain}")
    
    if not all_data:
        print("Error: No training data found!")
        return
    
    print(f"Total training samples: {len(all_data)}")
    
    # Split train/val (90/10)
    split_idx = int(0.9 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = VARoBERTaModel(CFG.model_name).to(CFG.device)
    
    # Use mixed precision training with float16
    if CFG.use_float16 and CFG.device.type == 'cuda':
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("Using mixed precision training (float16)")
    else:
        scaler = None
    
    # Create datasets and dataloaders
    train_dataset = VADataset(train_data, tokenizer, CFG.max_len)
    val_dataset = VADataset(val_data, tokenizer, CFG.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)
    
    # Create results directory
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results storage files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_json = os.path.join(results_dir, f'pretraining_results_{timestamp}.json')
    results_csv = os.path.join(results_dir, f'pretraining_metrics_{timestamp}.csv')
    results_log = os.path.join(results_dir, f'pretraining_log_{timestamp}.txt')
    config_file = os.path.join(results_dir, f'training_config_{timestamp}.json')
    
    # Save training configuration
    config = {
        'model_name': CFG.model_name,
        'max_len': CFG.max_len,
        'batch_size': CFG.batch_size,
        'epochs': CFG.epochs,
        'learning_rate': CFG.lr,
        'dropout': CFG.dropout,
        'fc_dropout': CFG.fc_dropout,
        'languages': args.languages,
        'domains': args.domains,
        'total_samples': len(all_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Initialize CSV file
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_RMSE', 'Is_Best', 'Timestamp'])
    
    # Initialize results dictionary
    training_results = {
        'config': config,
        'epochs_data': []
    }
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    
    # Training loop
    best_rmse = float('inf')
    print("\nStarting pretraining...")
    print(f"Results will be saved to: {results_dir}")
    
    # Open log file
    with open(results_log, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Pretraining Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Model: {CFG.model_name}\n")
        log_file.write(f"Languages: {', '.join(args.languages)}\n")
        log_file.write(f"Total samples: {len(all_data)}\n")
        log_file.write(f"{'='*60}\n\n")
        
        for epoch in range(CFG.epochs):
            epoch_start = datetime.now()
            print(f"\nEpoch {epoch+1}/{CFG.epochs}")
            log_file.write(f"\nEpoch {epoch+1}/{CFG.epochs} - {epoch_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, CFG.device, scaler)
            val_loss, val_rmse = validate(model, val_loader, criterion, CFG.device)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f} (Time: {epoch_time:.2f}s)")
            log_file.write(f"  Train Loss: {train_loss:.4f}\n")
            log_file.write(f"  Val Loss: {val_loss:.4f}\n")
            log_file.write(f"  Val RMSE: {val_rmse:.4f}\n")
            log_file.write(f"  Time: {epoch_time:.2f}s\n")
            
            # Prepare epoch result
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'val_rmse': float(val_rmse),
                'is_best': False,
                'timestamp': epoch_start.strftime('%Y-%m-%d %H:%M:%S'),
                'epoch_time_seconds': epoch_time
            }
            
            # Save best model
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                epoch_result['is_best'] = True
                best_model_path = os.path.join(args.output_dir, "va_roberta_multilingual_best.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f" Saved best model (RMSE: {best_rmse:.4f})")
                log_file.write(f"   Best model saved (RMSE: {best_rmse:.4f})\n")
            
            # Add to results
            training_results['epochs_data'].append(epoch_result)
            
            # Append to CSV
            with open(results_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, train_loss, val_loss, val_rmse,
                    'Yes' if epoch_result['is_best'] else 'No',
                    epoch_start.strftime('%Y-%m-%d %H:%M:%S')
                ])
            
            # Save JSON after each epoch
            training_results['best_rmse'] = float(best_rmse)
            training_results['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(results_json, 'w', encoding='utf-8') as f:
                json.dump(training_results, f, indent=2, ensure_ascii=False)
        
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Best RMSE: {best_rmse:.4f}\n")
        log_file.write(f"{'='*60}\n")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "va_roberta_multilingual_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    # Save tokenizer and config
    tokenizer.save_pretrained(args.output_dir)
    model.config.save_pretrained(args.output_dir)
    
    # Final summary
    training_results['training_completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_results['final_best_rmse'] = float(best_rmse)
    training_results['best_epoch'] = max([ep['epoch'] for ep in training_results['epochs_data'] if ep['is_best']], default=0)
    training_results['total_epochs_trained'] = len(training_results['epochs_data'])
    
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, indent=2, ensure_ascii=False)
    
    with open(results_log, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Best RMSE: {best_rmse:.4f} at epoch {training_results['best_epoch']}\n")
        log_file.write(f"Total epochs: {len(training_results['epochs_data'])}\n")
        log_file.write(f"{'='*60}\n")
    
    print(f"\n Pretraining completed!")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print(f"\nResults saved to separate files:")
    print(f"  - JSON: {results_json}")
    print(f"  - CSV: {results_csv}")
    print(f"  - Log: {results_log}")
    print(f"  - Config: {config_file}")

if __name__ == "__main__":
    main()
