# SemEval-2026 Task 3 — Code and Results (Paper Review)

Repository for **SemEval-2026 Task 3** (Track A, Subtask 1): Dimensional Aspect-Based Sentiment Analysis (DimABSA). Code and results accompanying the paper submission.

## Contents

- **Code**: Pretraining and fine-tuning (XLM-RoBERTa), evaluation, data splitting, and experiment runner.
- **Results**: Paper tables (`result.tex`, `results_Ss.tex`, `tables/`), dataset stats (`dataset_set.tex`, `tables/dataset_stats.tex`).
- **Paper**: `main.tex`, `main.bib`; figures in `pics/`.

## Requirements

- Python 3.8+
- PyTorch, transformers, and dependencies (see `evaluation_script/requirements.txt` for evaluation).

## Running Experiments

1. Prepare data: 80/20 train/test split in `data/train_80` and `data/test_20` (per language/domain JSONL; use `data/split_dataset.py` on the official task data).
2. Run experiments:
   ```bash
   python run_experiments.py
   ```
   This runs Exp1 (direct fine-tuning), Exp2 (pretrained only), and Exp3 (pretrain + fine-tune) for base/large models and all languages/domains.
3. Compute RMSE: `python calculate_all_rmse.py` (paths in script).
4. Official test predictions: `python generate_submission_from_official_test.py` (adjust paths to your official test data).

## Results

- **result.tex** / **results_Ss.tex**: Main result tables (RMSE on TrainEval 20% and official test).
- **tables/results.tex**, **tables/results_detailed.tex**: Detailed result tables for the paper.
- **dataset_set.tex**, **tables/dataset_stats.tex**: Dataset sentence counts (official training data, 80/20 split).

## Citation

If you use this code or results, please cite the SemEval-2026 Task 3 paper and the DimABSA dataset as indicated in the main paper.
