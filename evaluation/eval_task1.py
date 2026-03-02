#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 evaluation: computes RMSE for Valence/Arousal per language and per domain.
Expected prediction files: pred_{lang}_{domain}.jsonl with fields: id, aspect, valence, arousal.
Gold files: test JSONL from split (Aspect_VA or Quadruplet formats).
"""

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, Tuple, List


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_domain_from_filename(filename: str) -> str:
    # Expected patterns: lang_{domain}_train_alltasks_test_20.jsonl
    name = os.path.basename(filename)
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"


def gold_records_from_file(path: str) -> List[Tuple[str, str, float, float]]:
    """Return list of (id, aspect, valence, arousal)."""
    gold = []
    for record in load_jsonl(path):
        text_id = record.get("ID")
        # Prefer Aspect_VA; fallback Quadruplet
        aspect_list = record.get("Aspect_VA", []) or record.get("Quadruplet", [])
        for item in aspect_list:
            aspect = item.get("Aspect") or item.get("Category") or "UNKNOWN"
            va_str = item.get("VA", "")
            try:
                v, a = map(float, va_str.split("#"))
            except Exception:
                continue
            gold.append((text_id, aspect, v, a))
    return gold


def preds_from_file(path: str) -> Dict[Tuple[str, str], Tuple[float, float]]:
    preds = {}
    for record in load_jsonl(path):
        id_ = record.get("id") or record.get("ID")
        aspect = record.get("aspect") or record.get("Aspect")
        v = record.get("valence")
        a = record.get("arousal")
        if id_ is None or aspect is None or v is None or a is None:
            continue
        preds[(id_, aspect)] = (float(v), float(a))
    return preds


def rmse(pairs: List[Tuple[float, float]]) -> float:
    if not pairs:
        return float("nan")
    mse = sum((p - g) ** 2 for p, g in pairs) / len(pairs)
    return math.sqrt(mse)


def evaluate(lang: str, domain: str, gold_file: str, pred_file: str):
    gold = gold_records_from_file(gold_file)
    preds = preds_from_file(pred_file)
    v_pairs, a_pairs = [], []
    missing = 0
    for gid, aspect, gv, ga in gold:
        key = (gid, aspect)
        if key not in preds:
            missing += 1
            continue
        pv, pa = preds[key]
        v_pairs.append((pv, gv))
        a_pairs.append((pa, ga))
    v_rmse = rmse(v_pairs)
    a_rmse = rmse(a_pairs)
    avg_rmse = (v_rmse + a_rmse) / 2 if (not math.isnan(v_rmse) and not math.isnan(a_rmse)) else float("nan")
    return {
        "lang": lang,
        "domain": domain,
        "n_gold_aspects": len(gold),
        "n_eval_aspects": len(v_pairs),
        "n_missing": missing,
        "rmse_valence": v_rmse,
        "rmse_arousal": a_rmse,
        "rmse_avg": avg_rmse,
        "gold_file": gold_file,
        "pred_file": pred_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 1 predictions per language/domain")
    parser.add_argument("--test_dir", required=True, help="Directory with gold test files (by language)")
    parser.add_argument("--pred_dir", required=True, help="Directory with prediction files pred_{lang}_{domain}.jsonl")
    parser.add_argument("--output_json", default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    results = []
    by_lang = defaultdict(list)
    by_domain = defaultdict(list)

    # Iterate gold files
    for lang in sorted(os.listdir(args.test_dir)):
        lang_dir = os.path.join(args.test_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        for fname in os.listdir(lang_dir):
            if not fname.endswith(".jsonl"):
                continue
            gold_path = os.path.join(lang_dir, fname)
            domain = parse_domain_from_filename(fname)
            pred_path = os.path.join(args.pred_dir, f"pred_{lang}_{domain}.jsonl")
            if not os.path.exists(pred_path):
                print(f"[warn] Missing predictions for {lang}_{domain}: {pred_path}")
                continue
            res = evaluate(lang, domain, gold_path, pred_path)
            results.append(res)
            by_lang[lang].append(res)
            by_domain[domain].append(res)

    def summarize(group: List[dict]) -> dict:
        if not group:
            return {"rmse_avg": float("nan")}
        weights = [g["n_eval_aspects"] for g in group]
        total = sum(weights)
        def wavg(key):
            acc = 0.0
            for g, w in zip(group, weights):
                acc += g[key] * w
            return acc / total if total > 0 else float("nan")
        return {
            "rmse_avg": wavg("rmse_avg"),
            "rmse_valence": wavg("rmse_valence"),
            "rmse_arousal": wavg("rmse_arousal"),
            "n_eval_aspects": total,
        }

    # Print table
    print("\nPer language/domain RMSE:")
    for r in results:
        print(f"{r['lang']:>3} | {r['domain']:>10} | rmse_avg={r['rmse_avg']:.4f} | "
              f"V={r['rmse_valence']:.4f} A={r['rmse_arousal']:.4f} | eval_n={r['n_eval_aspects']}, miss={r['n_missing']}")

    print("\nPer language summary (weighted by evaluated aspects):")
    for lang in sorted(by_lang.keys()):
        s = summarize(by_lang[lang])
        print(f"{lang:>3} | rmse_avg={s['rmse_avg']:.4f} | V={s['rmse_valence']:.4f} A={s['rmse_arousal']:.4f} | eval_n={s['n_eval_aspects']}")

    print("\nPer domain summary (weighted by evaluated aspects):")
    for dom in sorted(by_domain.keys()):
        s = summarize(by_domain[dom])
        print(f"{dom:>10} | rmse_avg={s['rmse_avg']:.4f} | V={s['rmse_valence']:.4f} A={s['rmse_arousal']:.4f} | eval_n={s['n_eval_aspects']}")

    if args.output_json:
        report = {
            "results": results,
            "by_language": {k: summarize(v) for k, v in by_lang.items()},
            "by_domain": {k: summarize(v) for k, v in by_domain.items()},
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
