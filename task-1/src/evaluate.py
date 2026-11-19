"""
src/evaluate.py

Deterministic evaluation script for Base and Fine-tuned models on held-out test data.
Executes greedy generation, computes JSON validity, intent accuracy, EM, and field F1,
and persists both sample-level predictions and aggregate metrics.
"""

import argparse
import json
import time
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional
from tqdm import tqdm
import torch

from src.utils import set_seed, load_jsonl, save_jsonl, load_yaml_config, extract_json_payload
from src.model import load_model_for_inference, get_device
from src.metrics import evaluate_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Base vs LoRA Fine-Tuned Model")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to YAML config")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter checkpoint (None for base model)")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl", help="Path to held-out test set")
    parser.add_argument("--output_predictions", type=str, default=None, help="Output path for JSONL predictions")
    parser.add_argument("--output_metrics", type=str, default=None, help="Output path for JSON metrics")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick smoke test")
    return parser.parse_args()

def run_evaluation(
    model_name: str,
    adapter_path: Optional[str],
    test_path: str,
    output_preds_path: str,
    output_metrics_path: str,
    max_new_tokens: int = 120,
    batch_size: int = 8,
    limit: Optional[int] = None,
    seed: int = 42
):
    set_seed(seed)
    device = get_device()
    print(f"\n[Environment] Device: {device} | Adapter: {adapter_path or 'None (Base Model)'} | Batch Size: {batch_size}")

    # Load test dataset
    samples = load_jsonl(test_path)
    if limit is not None and limit > 0:
        samples = samples[:limit]
    print(f"Loaded {len(samples)} held-out evaluation samples from: {test_path}")

    # Load model and tokenizer
    start_load = time.time()
    model, tokenizer = load_model_for_inference(model_name, adapter_path=adapter_path, device=str(device))
    # For batched generation on causal LMs, padding MUST be on the left
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded in {time.time() - start_load:.2f}s")

    # Run deterministic batched inference
    print(f"Executing deterministic greedy generation (batch_size={batch_size}, do_sample=False)...")
    raw_outputs = []

    # Process in batches
    for i in tqdm(range(0, len(samples), batch_size), desc="Batched Inference"):
        batch_samples = samples[i:i + batch_size]
        batch_prompts = []
        for sample in batch_samples:
            chat_prompt_messages = [
                m for m in sample["messages"] if m["role"] in ("system", "user")
            ]
            prompt_text = tokenizer.apply_chat_template(
                chat_prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(prompt_text)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Slice generated response tokens for each sample in the batch
        input_len = inputs["input_ids"].shape[1]
        for out_seq in output_tokens:
            gen_tokens = out_seq[input_len:]
            raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            raw_outputs.append(raw_text)

    # Compute complete metrics suite
    ground_truths = [s["ground_truth"] for s in samples]
    metrics, parsed_preds = evaluate_predictions(raw_outputs, ground_truths)

    # Build detailed prediction records
    generation_records = []
    for s, raw, parsed in zip(samples, raw_outputs, parsed_preds):
        user_query = ""
        for m in s["messages"]:
            if m["role"] == "user":
                user_query = m["content"]
                break

        gt = s["ground_truth"]
        exact_match = (parsed == gt) if parsed else False
        intent_match = (parsed.get("intent") == gt.get("intent")) if parsed else False

        generation_records.append({
            "id": s["id"],
            "user_query": user_query,
            "raw_output": raw,
            "parsed_prediction": parsed,
            "ground_truth": gt,
            "exact_match": exact_match,
            "intent_match": intent_match
        })

    # Save predictions and metrics
    save_jsonl(generation_records, output_preds_path)
    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS SUMMARY: {adapter_path or 'Base Model'}")
    print("=" * 70)
    print(f"  * Total Samples:               {metrics['total_samples']}")
    print(f"  * Raw JSON Validity Rate:      {metrics['raw_json_validity_rate'] * 100:.2f}%")
    print(f"  * Extracted JSON Validity:     {metrics['extracted_json_validity_rate'] * 100:.2f}%")
    print(f"  * Intent Accuracy:             {metrics['intent_accuracy'] * 100:.2f}%")
    print(f"  * Exact Match (EM) Accuracy:   {metrics['exact_match_accuracy'] * 100:.2f}%")
    print(f"  * Field Micro Precision:       {metrics['field_precision'] * 100:.2f}%")
    print(f"  * Field Micro Recall:          {metrics['field_recall'] * 100:.2f}%")
    print(f"  * Field Micro F1 Score:        {metrics['field_f1'] * 100:.2f}%")
    print(f"  * Field Micro Accuracy:        {metrics['field_accuracy'] * 100:.2f}%")
    print("=" * 70)
    print(f"Predictions saved to: {output_preds_path}")
    print(f"Metrics saved to:     {output_metrics_path}")
    print("=" * 70)

    return metrics

def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    model_name = cfg["model"]["base_model_name_or_path"]
    test_path = args.test_data or cfg["paths"]["test_data"]
    
    if args.adapter_path:
        out_preds = args.output_predictions or cfg["paths"]["finetuned_predictions"]
        out_metrics = args.output_metrics or cfg["paths"]["finetuned_results"]
    else:
        out_preds = args.output_predictions or cfg["paths"]["baseline_predictions"]
        out_metrics = args.output_metrics or cfg["paths"]["baseline_results"]

    run_evaluation(
        model_name=model_name,
        adapter_path=args.adapter_path,
        test_path=test_path,
        output_preds_path=out_preds,
        output_metrics_path=out_metrics,
        max_new_tokens=cfg["generation"].get("max_new_tokens", 160),
        limit=args.limit,
        seed=cfg["training"].get("seed", 42)
    )

if __name__ == "__main__":
    main()
