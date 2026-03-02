# Configuration for XLM-RoBERTa-Base Model
# File: config_base.py

class CFG_BASE:
    model_name = "FacebookAI/xlm-roberta-base"
    max_len = 192
    batch_size = 4
    epochs = 10
    lr = 1.5e-5
    dropout = 0.1
    fc_dropout = 0.3
    device = "cuda"
    seed = 42
    early_stopping_patience = 5  # Stop when no improvement for 5 epochs
    gradient_accumulation_steps = 1
    use_float16 = False  # Will use bfloat16 instead
    use_bfloat16 = True  # Use bfloat16 for better stability and performance
    
    # Pretraining specific
    pretrain_epochs = 5
    
    # Data paths
    train_data_dir = "data/train_80"
    test_data_dir = "data/test_20"
    
    # Languages
    languages = ['eng', 'jpn', 'rus', 'tat', 'ukr', 'zho']
    
    # Output directories
    pretrain_output_dir = "pretraining/models_base"
    finetune_output_dir = "finetuning/models_base"
