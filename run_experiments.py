#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Experiment Runner
Runs all experiments systematically:
- Experiment 1: Direct fine-tuning (base, large, XL)
- Experiment 3: Pretraining + Fine-tuning (base, large, XL)
- Both data structures: language-specific folders and single shuffled JSONL
- All 6 languages
- Domain-specific evaluation
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# Model configurations
# NOTE: Only base and large models are used in experiments
# XL and mdeberta models are excluded
MODELS = {
    'base': 'FacebookAI/xlm-roberta-base',
    'large': 'FacebookAI/xlm-roberta-large',
    # 'xl': 'facebook/xlm-roberta-xl',  # EXCLUDED - Not used in experiments
    # 'mdeberta': 'microsoft/mdeberta-v3-base',  # EXCLUDED - Not used in experiments
}

# Languages
LANGUAGES = ['eng', 'jpn', 'rus', 'tat', 'ukr', 'zho']

# Data paths
DATA_PATHS = {
    'language_specific': {
        'train': 'data/train_80',
        'test': 'data/test_20'
    },
    'multilingual_shuffled': {
        'train': 'data/train_80_multilingual_shuffled.jsonl',
        'test': 'data/test_20_multilingual_shuffled.jsonl'
    }
}

def _query_gpus() -> List[Tuple[int, int, int, int]]:
    """
    Returns list of tuples: (gpu_index, mem_used_mb, mem_total_mb, util_pct)
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
        except ValueError:
            continue
    return gpus


def pick_gpu(preferred: Optional[int] = None, min_free_memory_mb: int = 10000) -> Optional[int]:
    """
    Picks a GPU index. If preferred is provided, returns it.
    Otherwise chooses a FREE GPU (utilization < 10% and free memory > min_free_memory_mb).
    Among free GPUs, chooses the one with the most free memory (tie-break: lower util, then lower index).
    """
    if preferred is not None:
        return preferred
    gpus = _query_gpus()
    if not gpus:
        return None
    
    # Filter for free GPUs only (utilization < 10% and sufficient free memory)
    free_gpus = [
        gpu for gpu in gpus
        if gpu[3] < 10 and (gpu[2] - gpu[1]) >= min_free_memory_mb
    ]
    
    if not free_gpus:
        print("Warning: No free GPUs found. Using GPU with most free memory anyway.")
        free_gpus = gpus
    
    # Among free GPUs, maximize free memory, then minimize util, then minimize index
    gpus_sorted = sorted(
        free_gpus,
        key=lambda t: (-(t[2] - t[1]), t[3], t[0]),
    )
    return gpus_sorted[0][0]


def run_command(cmd, description, log_file=None, env_overrides: Optional[Dict[str, str]] = None):
    """Run a command and log output"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{datetime.now()}: {description}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"{'='*80}\n")
            f.flush()
    
    try:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        # Always set PYTORCH_CUDA_ALLOC_CONF for memory optimization
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
        )
        output = result.stdout + result.stderr
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(output)
        print(output)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"Error: {e}\n{e.stderr}"
        print(error_msg)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg)
        return False

def experiment_1_direct_finetune(model_size, model_name, data_structure, lang=None, log_file=None, gpu: Optional[int] = None):
    """Experiment 1: Direct fine-tuning without pretraining"""
    exp_name = f"exp1_direct_finetune_{model_size}_{data_structure}"
    if lang:
        exp_name += f"_{lang}"
    
    output_dir = f"experiments/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    if data_structure == 'language_specific':
        data_dir = DATA_PATHS['language_specific']['train']
        test_dir = DATA_PATHS['language_specific']['test']
        
        # Get domains for the language
        lang_dir = os.path.join(data_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"Warning: {lang_dir} not found, skipping...")
            return False
        
        domains = []
        for f in os.listdir(lang_dir):
            if f.endswith('.jsonl') and 'train' in f:
                # Extract domain from filename
                parts = f.replace('_train_80.jsonl', '').split('_')
                if len(parts) >= 2:
                    domain = parts[1]  # e.g., restaurant, laptop, finance
                    if domain not in domains:
                        domains.append(domain)
        
        if not domains:
            print(f"No domains found for {lang}")
            return False
        
        # Use model-specific fine-tuning script
        if 'large' in model_name.lower():
            finetune_script = 'finetuning/finetune_large.py'
        elif 'base' in model_name.lower() or model_size == 'base':
            finetune_script = 'finetuning/finetune_base.py'
        else:
            finetune_script = 'finetuning/finetune.py'  # Fallback for xl, mdeberta
        
        cmd = [
            sys.executable, finetune_script,
            '--pretrained_model_dir', 'NONE',
            '--data_dir', data_dir,
            '--test_data_dir', test_dir,
            '--output_dir', output_dir,
            '--lang', lang,
            '--domains'] + domains + [
            '--model_name', model_name
        ]
    else:
        # Multilingual shuffled - fine-tune on all languages from single JSONL
        train_file = DATA_PATHS['multilingual_shuffled']['train']
        test_file = DATA_PATHS['multilingual_shuffled']['test']
        
        cmd = [
            sys.executable, 'finetuning/finetune.py',
            '--pretrained_model_dir', 'NONE',
            '--data_dir', train_file,
            '--test_data_dir', test_file,
            '--output_dir', output_dir,
            '--model_name', model_name,
            '--data_format', 'single_jsonl'
        ]
    
    description = f"Experiment 1: Direct Fine-tuning | Model: {model_size} | Data: {data_structure} | Lang: {lang}"
    env_overrides = {}
    if gpu is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return run_command(cmd, description, log_file, env_overrides=env_overrides)

def experiment_2_pretrained_only(model_size, model_name, data_structure, lang=None, log_file=None, gpu: Optional[int] = None):
    """Experiment 2: Pretrained model only (inference only, no fine-tuning)"""
    exp_name = f"exp2_pretrained_only_{model_size}_{data_structure}"
    if lang:
        exp_name += f"_{lang}"
    
    output_dir = f"experiments/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if pretrained model exists - Experiment 3 saves them in experiments/exp3_*_pretrain/
    # Try Experiment 3 location first, then fallback to old pretraining/models locations
    exp3_pretrain_dir = f"experiments/exp3_pretrain_finetune_{model_size}_language_specific_pretrain"
    pretrain_dir_base = f"pretraining/models"
    pretrain_dir_large = f"pretraining/models_large"
    pretrain_dir_xl = f"pretraining/models_xl"
    pretrain_dir_mdeberta = f"pretraining/models_mdeberta"
    
    # Determine which pretrained model to use based on model size
    # First check Experiment 3 location
    if os.path.exists(exp3_pretrain_dir):
        pretrain_dir = exp3_pretrain_dir
    elif 'large' in model_name.lower():
        pretrain_dir = pretrain_dir_large
    elif 'xl' in model_name.lower():
        pretrain_dir = pretrain_dir_xl
    elif 'mdeberta' in model_name.lower() or 'deberta' in model_name.lower():
        pretrain_dir = pretrain_dir_mdeberta
    elif 'base' in model_name.lower():
        pretrain_dir = pretrain_dir_base
    else:
        pretrain_dir = pretrain_dir_base
    
    pretrained_model_path = os.path.join(pretrain_dir, "va_roberta_multilingual_best.pth")
    
    if not os.path.exists(pretrained_model_path):
        print(f"Warning: Pretrained model not found: {pretrained_model_path}")
        print("   Run pretraining first (Experiment 3 pretraining step)!")
        return False
    
    if data_structure == 'language_specific':
        data_dir = DATA_PATHS['language_specific']['train']
        test_dir = DATA_PATHS['language_specific']['test']
        
        # Get domains for the language
        lang_dir = os.path.join(data_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"Warning: {lang_dir} not found, skipping...")
            return False
        
        domains = []
        for domain_file in os.listdir(lang_dir):
            if domain_file.endswith('_train_80.jsonl'):
                domain = domain_file.replace(f"{lang}_", "").replace("_train_alltasks_train_80.jsonl", "").replace("_train_task1_train_80.jsonl", "")
                if domain not in domains:
                    domains.append(domain)
        
        if not domains:
            print(f"Warning: No training files found for {lang}")
            return False
        
        # Use model-specific fine-tuning script
        if 'large' in model_name.lower():
            finetune_script = 'finetuning/finetune_large.py'
        elif 'base' in model_name.lower() or model_size == 'base':
            finetune_script = 'finetuning/finetune_base.py'
        else:
            finetune_script = 'finetuning/finetune.py'  # Fallback for xl, mdeberta
        
        cmd = [
            sys.executable, finetune_script,
            '--pretrained_model_dir', pretrain_dir,
            '--data_dir', data_dir,
            '--test_data_dir', test_dir,
            '--output_dir', output_dir,
            '--lang', lang,
            '--domains'] + domains + [
            '--model_name', model_name,
            '--skip_training'  # Inference only
        ]
    else:
        # Multilingual shuffled
        train_file = DATA_PATHS['multilingual_shuffled']['train']
        test_file = DATA_PATHS['multilingual_shuffled']['test']
        
        # Use model-specific fine-tuning script
        if 'large' in model_name.lower():
            finetune_script = 'finetuning/finetune_large.py'
        elif 'base' in model_name.lower() or model_size == 'base':
            finetune_script = 'finetuning/finetune_base.py'
        else:
            finetune_script = 'finetuning/finetune.py'  # Fallback for xl, mdeberta
        
        cmd = [
            sys.executable, finetune_script,
            '--pretrained_model_dir', pretrain_dir,
            '--data_dir', train_file,
            '--test_data_dir', test_file,
            '--output_dir', output_dir,
            '--model_name', model_name,
            '--data_format', 'single_jsonl',
            '--skip_training'  # Inference only
        ]
    
    description = f"Experiment 2: Pretrained Only (Inference) | Model: {model_size} | Data: {data_structure}"
    if lang:
        description += f" | Lang: {lang}"
    
    env_overrides = {}
    if gpu is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return run_command(cmd, description, log_file, env_overrides=env_overrides)

def experiment_3_pretrain_finetune(model_size, model_name, data_structure, lang=None, log_file=None, gpu: Optional[int] = None):
    """Experiment 3: Pretraining + Fine-tuning"""
    exp_name = f"exp3_pretrain_finetune_{model_size}_{data_structure}"
    if lang:
        exp_name += f"_{lang}"
    
    pretrain_dir = f"experiments/{exp_name}_pretrain"
    finetune_dir = f"experiments/{exp_name}_finetune"
    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(finetune_dir, exist_ok=True)
    
    # Step 1: Pretraining - Use model-specific script
    if 'large' in model_name.lower():
        pretrain_script = 'pretraining/pretrain_large.py'
    elif 'base' in model_name.lower() or model_size == 'base':
        pretrain_script = 'pretraining/pretrain_base.py'
    elif 'xl' in model_name.lower():
        pretrain_script = 'pretraining/pretrain_xl.py'
    elif 'mdeberta' in model_name.lower() or 'deberta' in model_name.lower():
        pretrain_script = 'pretraining/pretrain_mdeberta.py'
    else:
        pretrain_script = 'pretraining/pretrain.py'
    
    if data_structure == 'language_specific':
        data_dir = DATA_PATHS['language_specific']['train']
        languages = LANGUAGES  # Pretrain on all languages
        domains = ['restaurant', 'laptop', 'finance', 'hotel']  # All domains
        
        pretrain_cmd = [
            sys.executable, pretrain_script,
            '--data_dir', data_dir,
            '--output_dir', pretrain_dir,
            '--languages'] + languages + [
            '--domains'] + domains + [
            '--model_name', model_name,
            '--data_format', 'folder'
        ]
    else:
        # Multilingual shuffled
        data_dir = DATA_PATHS['multilingual_shuffled']['train']
        pretrain_cmd = [
            sys.executable, pretrain_script,
            '--data_dir', data_dir,
            '--output_dir', pretrain_dir,
            '--model_name', model_name,
            '--data_format', 'single_jsonl'
        ]
    
    # Check if pretraining already exists
    pretrained_model_path = os.path.join(pretrain_dir, "va_roberta_multilingual_best.pth")
    if not os.path.exists(pretrained_model_path):
    description = f"Experiment 3: Pretraining | Model: {model_size} | Data: {data_structure}"
    env_overrides = {}
    if gpu is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not run_command(pretrain_cmd, description, log_file, env_overrides=env_overrides):
        return False
    else:
        print(f"Pretraining already exists at {pretrained_model_path}, skipping pretraining step")
    
    # Step 2: Fine-tuning
    if data_structure == 'language_specific' and lang:
        test_dir = DATA_PATHS['language_specific']['test']
        lang_dir = os.path.join(data_dir, lang)
        domains = []
        for f in os.listdir(lang_dir):
            if f.endswith('.jsonl') and 'train' in f:
                parts = f.replace('_train_80.jsonl', '').split('_')
                if len(parts) >= 2:
                    domain = parts[1]
                    if domain not in domains:
                        domains.append(domain)
        
        if not domains:
            return False
        
        # Use model-specific fine-tuning script
        if 'large' in model_name.lower():
            finetune_script = 'finetuning/finetune_large.py'
        elif 'base' in model_name.lower() or model_size == 'base':
            finetune_script = 'finetuning/finetune_base.py'
        else:
            finetune_script = 'finetuning/finetune.py'  # Fallback for xl, mdeberta
        
        finetune_cmd = [
            sys.executable, finetune_script,
            '--pretrained_model_dir', pretrain_dir,
            '--data_dir', data_dir,
            '--test_data_dir', test_dir,
            '--output_dir', finetune_dir,
            '--lang', lang,
            '--domains'] + domains + [
            '--model_name', model_name
        ]
    else:
        print(f"Fine-tuning for multilingual shuffled not yet fully implemented")
        return False
    
    description = f"Experiment 3: Fine-tuning | Model: {model_size} | Data: {data_structure} | Lang: {lang}"
    return run_command(finetune_cmd, description, log_file, env_overrides=env_overrides)

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive experiments')
    parser.add_argument('--experiment', type=str, choices=['1', '2', '3', 'all'], default='all',
                       help='Which experiment to run (1=direct finetune, 2=pretrained only inference, 3=pretrain+finetune, all=all)')
    parser.add_argument('--model_size', type=str, choices=['base', 'large', 'all'], default='all',
                       help='Model size to use (base or large only - xl and mdeberta are excluded)')
    parser.add_argument('--data_structure', type=str, choices=['language_specific', 'multilingual_shuffled', 'all'], 
                       default='all', help='Data structure to use')
    parser.add_argument('--languages', nargs='+', default=LANGUAGES,
                       help='Languages to run experiments on')
    parser.add_argument('--skip_pretrain', action='store_true',
                       help='Skip pretraining phase (for Experiment 3)')
    parser.add_argument('--log_file', type=str, default='experiments/experiment_log.txt',
                       help='Log file path')
    parser.add_argument('--gpu', type=str, default='auto',
                       help='GPU index to use (e.g., 2) or \"auto\" to pick least-used GPU')
    
    args = parser.parse_args()
    
    # Create experiments directory
    os.makedirs('experiments', exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    # Determine which models to run
    models_to_run = []
    if args.model_size == 'all':
        models_to_run = list(MODELS.items())
    else:
        if args.model_size in MODELS:
            models_to_run = [(args.model_size, MODELS[args.model_size])]
        else:
            print(f"Warning: Model size {args.model_size} not available")
            return
    
    # Determine data structures
    data_structures = []
    if args.data_structure == 'all':
        data_structures = ['language_specific', 'multilingual_shuffled']
    else:
        data_structures = [args.data_structure]
    
    # Determine experiments
    experiments = []
    if args.experiment == 'all':
        experiments = ['1', '2', '3']
    else:
        experiments = [args.experiment]
    
    # Run experiments
    results = {
        'start_time': datetime.now().isoformat(),
        'experiments': []
    }
    
    print(f"\n{'='*80}")
    print(f"Starting Comprehensive Experiments")
    print(f"Experiments: {experiments}")
    print(f"Models: {[m[0] for m in models_to_run]}")
    print(f"Data Structures: {data_structures}")
    print(f"Languages: {args.languages}")
    print(f"{'='*80}\n")

    preferred_gpu = None
    if args.gpu != "auto":
        try:
            preferred_gpu = int(args.gpu)
        except ValueError:
            preferred_gpu = None
    selected_gpu = pick_gpu(preferred_gpu, min_free_memory_mb=10000)
    if selected_gpu is None:
        print("Warning: Could not query GPUs; running without CUDA_VISIBLE_DEVICES pinning.")
    else:
        gpus = _query_gpus()
        gpu_info = next((g for g in gpus if g[0] == selected_gpu), None)
        if gpu_info:
            free_mem = gpu_info[2] - gpu_info[1]
            util = gpu_info[3]
            print(f"Selected FREE GPU: {selected_gpu} (Free Memory: {free_mem}MB, Utilization: {util}%)")
    else:
        print(f"Selected GPU: {selected_gpu} (CUDA_VISIBLE_DEVICES={selected_gpu})")
    
    for exp_num in experiments:
        for model_size, model_name in models_to_run:
            for data_structure in data_structures:
                if exp_num == '1':
                    # Experiment 1: Direct fine-tuning
                    for lang in args.languages:
                        if data_structure == 'language_specific':
                            success = experiment_1_direct_finetune(
                                model_size, model_name, data_structure, lang, args.log_file, gpu=selected_gpu
                            )
                            results['experiments'].append({
                                'experiment': '1',
                                'model': model_size,
                                'data_structure': data_structure,
                                'language': lang,
                                'success': success,
                                'timestamp': datetime.now().isoformat()
                            })
                
                elif exp_num == '2':
                    # Experiment 2: Pretrained model only (inference only)
                    for lang in args.languages:
                        if data_structure == 'language_specific':
                            success = experiment_2_pretrained_only(
                                model_size, model_name, data_structure, lang, args.log_file, gpu=selected_gpu
                            )
                            results['experiments'].append({
                                'experiment': '2',
                                'model': model_size,
                                'data_structure': data_structure,
                                'language': lang,
                                'success': success,
                                'timestamp': datetime.now().isoformat()
                            })
                        elif data_structure == 'multilingual_shuffled':
                            # For multilingual shuffled, no language loop needed
                            success = experiment_2_pretrained_only(
                                model_size, model_name, data_structure, None, args.log_file, gpu=selected_gpu
                            )
                            results['experiments'].append({
                                'experiment': '2',
                                'model': model_size,
                                'data_structure': data_structure,
                                'language': None,
                                'success': success,
                                'timestamp': datetime.now().isoformat()
                            })
                
                elif exp_num == '3':
                    # Experiment 3: Pretraining + Fine-tuning
                    if not args.skip_pretrain:
                        # Pretraining (once per model/data combination)
                        success = experiment_3_pretrain_finetune(
                            model_size, model_name, data_structure, None, args.log_file, gpu=selected_gpu
                        )
                        if not success:
                            print(f"Pretraining failed, skipping fine-tuning")
                            continue
                    
                    # Fine-tuning (per language)
                    for lang in args.languages:
                        if data_structure == 'language_specific':
                            success = experiment_3_pretrain_finetune(
                                model_size, model_name, data_structure, lang, args.log_file, gpu=selected_gpu
                            )
                            results['experiments'].append({
                                'experiment': '3',
                                'model': model_size,
                                'data_structure': data_structure,
                                'language': lang,
                                'success': success,
                                'timestamp': datetime.now().isoformat()
                            })
    
    results['end_time'] = datetime.now().isoformat()
    
    # Save results
    results_file = f"experiments/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Experiments Completed!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
