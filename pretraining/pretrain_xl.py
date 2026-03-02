#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretraining Script for XLM-RoBERTa-XL Model
Separate file for XL model only
"""

import sys
import os
# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model-specific config
from config_xl import CFG_XL as CFG

# Import all other necessary modules from original pretrain.py
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

# Set device - use first available GPU (respects CUDA_VISIBLE_DEVICES)
if torch.cuda.is_available():
    # Get the first visible GPU (CUDA_VISIBLE_DEVICES sets this to 0)
    CFG.device = torch.device(f"cuda:0")
    print(f"Using device: {CFG.device} (GPU {torch.cuda.current_device()})")
else:
    CFG.device = torch.device("cpu")
    print(f"Using device: {CFG.device}")

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(CFG.seed)

# Define model class and helper functions (self-contained)
def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def process_data(jsonl_data):
    """Process JSONL data into training format"""
    processed = []
    for item in jsonl_data:
        text = item.get('Text', '')
        aspect_va = item.get('Aspect_VA', [])
        for aspect_item in aspect_va:
            aspect = aspect_item.get('Aspect', '')
            va_str = aspect_item.get('VA', '')
            if va_str:
                try:
                    v, a = map(float, va_str.split('#'))
                    processed.append({
                        'Text': text,
                        'Aspect': aspect,
                        'Valence': v,
                        'Arousal': a
                    })
                except:
                    continue
    return processed

class VADataset(Dataset):
    """Dataset for Valence-Arousal prediction"""
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

class VARoBERTaModel(nn.Module):
    """VA-RoBERTa Model for Valence-Arousal prediction"""
    def __init__(self, model_name, config=None):
        super().__init__()
        if config is None:
        self.config = AutoConfig.from_pretrained(model_name)
        else:
            self.config = config
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
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
        """Forward pass"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask  # [batch, hidden_size]
        
        # Apply dropout and FC layer
        x = self.dropout(pooled)
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.fc_dropout(x)
        
        # Predict Valence and Arousal
        valence = self.valence_head(x)
        arousal = self.arousal_head(x)
        
        return valence.squeeze(-1), arousal.squeeze(-1)

# Override train_epoch to use mixed precision
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
            # Use bfloat16 if available, otherwise float16
            dtype = torch.bfloat16 if hasattr(CFG, 'use_bfloat16') and CFG.use_bfloat16 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
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
    parser = argparse.ArgumentParser(description='Pretrain XLM-RoBERTa-XL Model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=CFG.pretrain_output_dir)
    parser.add_argument('--languages', nargs='+', default=CFG.languages)
    parser.add_argument('--domains', nargs='+', default=['restaurant'])
    parser.add_argument('--data_format', type=str, choices=['folder', 'single_jsonl'], default='folder')
    parser.add_argument('--model_name', type=str, default=CFG.model_name, help='Model name (overrides config)')
    
    args = parser.parse_args()
    
    # Use model name from config
    CFG.model_name = args.model_name if args.model_name != CFG.model_name else CFG.model_name
    CFG.epochs = CFG.pretrain_epochs
    
    print(f"Using model: {CFG.model_name}")
    print(f"Batch size: {CFG.batch_size}")
    print(f"Float16: {CFG.use_float16}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading training data...")
    all_data = []
    
    if args.data_format == 'single_jsonl' or (os.path.isfile(args.data_dir) and args.data_dir.endswith('.jsonl')):
        data_file = args.data_dir if os.path.isfile(args.data_dir) else args.data_dir
        print(f"Loading from single JSONL file: {data_file}...")
        jsonl_data = load_jsonl(data_file)
        processed = process_data(jsonl_data)
        all_data.extend(processed)
        print(f"Loaded {len(processed)} samples from single file")
    else:
        for lang in args.languages:
            for domain in args.domains:
                lang_dir = os.path.join(args.data_dir, lang)
                if not os.path.exists(lang_dir):
                    continue
                
                patterns = [
                    f"{lang}_{domain}_train_alltasks_train_80.jsonl",
                    f"{lang}_{domain}_train_task1_train_80.jsonl"
                ]
                
                for pattern in patterns:
                    file_path = os.path.join(lang_dir, pattern)
                    if os.path.exists(file_path):
                        print(f"Loading {file_path}...")
                        jsonl_data = load_jsonl(file_path)
                        processed = process_data(jsonl_data)
                        all_data.extend(processed)
                        print(f"  Loaded {len(processed)} samples")
                        break
    
    if not all_data:
        print("Error: No training data found!")
        return
    
    print(f"\nTotal samples: {len(all_data)}")
    
    # Split train/val (90/10)
    split_idx = int(0.9 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Initialize tokenizer and model
    print(f"Loading tokenizer and model on {CFG.device}...")
    print(f"CUDA available: {torch.cuda.is_available()}, Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    
    # Enable gradient checkpointing to save memory
    config = AutoConfig.from_pretrained(CFG.model_name)
    config.gradient_checkpointing = True
    print("✅ Gradient checkpointing enabled in config")
    
    model = VARoBERTaModel(CFG.model_name, config=config).to(CFG.device)
    
    # Enable gradient checkpointing on the model
    if hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("✅ Model gradient checkpointing enabled")
    
    # Use mixed precision training with bfloat16 or float16
    if CFG.device.type == 'cuda':
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        if hasattr(CFG, 'use_bfloat16') and CFG.use_bfloat16:
            print("✅ Mixed precision training (bfloat16) enabled")
        elif CFG.use_float16:
            print("✅ Mixed precision training (float16) enabled")
        else:
            scaler = None
            print("⚠️ Mixed precision disabled")
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
        'use_float16': CFG.use_float16,
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
    
    # Try 8-bit optimizer to save memory
    try:
        import bitsandbytes as bnb
        print("✅ Using 8-bit AdamW optimizer (bitsandbytes) to save memory")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=CFG.lr,
            weight_decay=0.01,
            eps=1e-8
        )
        print(f"✅ 8-bit Optimizer created (AdamW8bit, lr={CFG.lr})")
    except ImportError:
        print("⚠️ bitsandbytes not available, using standard AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=CFG.lr,
            weight_decay=0.01,
            eps=1e-8
        )
        print(f"✅ Optimizer created (AdamW, lr={CFG.lr})")
    
    # Training loop
    best_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = getattr(CFG, 'early_stopping_patience', 5)
    print("\nStarting pretraining...")
    print(f"Results will be saved to: {results_dir}")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    
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
            
            # Save best model and check for early stopping
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_epoch = epoch + 1
                epoch_result['is_best'] = True
                patience_counter = 0  # Reset patience counter
                best_model_path = os.path.join(args.output_dir, "va_roberta_multilingual_best.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✅ Best model saved (RMSE: {best_rmse:.4f} at epoch {best_epoch})")
                log_file.write(f"   Best model saved (RMSE: {best_rmse:.4f} at epoch {best_epoch})\n")
            else:
                patience_counter += 1
                print(f"  No improvement (patience: {patience_counter}/{early_stopping_patience})")
                log_file.write(f"   No improvement (patience: {patience_counter}/{early_stopping_patience})\n")
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                print(f"Best RMSE: {best_rmse:.4f} at epoch {best_epoch}")
                log_file.write(f"\n{'='*60}\n")
                log_file.write(f"Early stopping triggered at epoch {epoch + 1}\n")
                log_file.write(f"Best RMSE: {best_rmse:.4f} at epoch {best_epoch}\n")
                log_file.write(f"{'='*60}\n")
                break
            
            # Add epoch result to training results
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
        
        # Final training summary
        training_results['training_completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        training_results['final_best_rmse'] = float(best_rmse)
        training_results['best_epoch'] = best_epoch
        training_results['total_epochs_trained'] = len(training_results['epochs_data'])
        training_results['early_stopping_patience'] = early_stopping_patience
        
        with open(results_json, 'w', encoding='utf-8') as f:
            json.dump(training_results, f, indent=2, ensure_ascii=False)
        
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Best RMSE: {best_rmse:.4f} at epoch {training_results['best_epoch']}\n")
        log_file.write(f"Total epochs: {len(training_results['epochs_data'])}\n")
        log_file.write(f"{'='*60}\n")
    
    print(f"\nPretraining completed!")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"  - JSON: {results_json}")
    print(f"  - CSV: {results_csv}")
    print(f"  - Log: {results_log}")
    print(f"  - Config: {config_file}")
    print(f"  - Best Model: {os.path.join(args.output_dir, 'va_roberta_multilingual_best.pth')}")

if __name__ == "__main__":
    main()
