"""
src/metrics.py

Comprehensive evaluation metrics for Structured JSON extraction:
1. JSON Validity Rate (Raw vs Extracted)
2. Intent Classification Accuracy
3. Exact Match Accuracy (Canonical JSON equality)
4. Field-level Micro Precision, Recall, F1, and Accuracy
5. Per-field breakdown statistics
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from src.utils import extract_json_payload

def canonical_normalize(val: Any) -> Any:
    """Recursively normalizes values for fair equality comparison."""
    if isinstance(val, dict):
        return {k: canonical_normalize(v) for k, v in sorted(val.items())}
    elif isinstance(val, list):
        # Sort lists of primitive types for order-invariance where appropriate
        norm_list = [canonical_normalize(item) for item in val]
        if all(isinstance(x, (str, int, float)) for x in norm_list):
            return sorted(norm_list)
        return norm_list
    elif isinstance(val, float):
        return round(val, 2)
    elif isinstance(val, str):
        return val.strip()
    return val

def compute_json_validity(raw_outputs: List[str]) -> Tuple[float, float, List[Optional[Dict[str, Any]]]]:
    """
    Computes:
    - raw_validity_rate: proportion that are directly valid JSON without extraction.
    - extracted_validity_rate: proportion valid after markdown/brace extraction.
    Returns:
    (raw_validity_rate, extracted_validity_rate, parsed_objects)
    """
    n = len(raw_outputs)
    if n == 0:
        return 0.0, 0.0, []

    raw_valid_cnt = 0
    extracted_valid_cnt = 0
    parsed_objects: List[Optional[Dict[str, Any]]] = []

    for raw in raw_outputs:
        clean = raw.strip()
        is_raw_valid = False
        parsed_obj = None

        try:
            parsed_obj = json.loads(clean)
            if isinstance(parsed_obj, dict):
                is_raw_valid = True
                raw_valid_cnt += 1
                extracted_valid_cnt += 1
        except Exception:
            pass

        if not is_raw_valid:
            extracted_str, _ = extract_json_payload(clean)
            if extracted_str:
                try:
                    p = json.loads(extracted_str)
                    if isinstance(p, dict):
                        parsed_obj = p
                        extracted_valid_cnt += 1
                except Exception:
                    parsed_obj = None

        parsed_objects.append(parsed_obj)

    return (raw_valid_cnt / n), (extracted_valid_cnt / n), parsed_objects

def compute_intent_accuracy(parsed_preds: List[Optional[Dict[str, Any]]], ground_truths: List[Dict[str, Any]]) -> float:
    """Computes accuracy of 'intent' classification across all samples."""
    correct = 0
    total = len(ground_truths)
    if total == 0:
        return 0.0

    for pred, gt in zip(parsed_preds, ground_truths):
        if pred is not None and isinstance(pred, dict):
            if pred.get("intent") == gt.get("intent"):
                correct += 1
    return correct / total

def compute_exact_match(parsed_preds: List[Optional[Dict[str, Any]]], ground_truths: List[Dict[str, Any]]) -> float:
    """
    Computes Exact Match (EM) accuracy using deep canonical equality:
    All keys and values must match perfectly.
    """
    correct = 0
    total = len(ground_truths)
    if total == 0:
        return 0.0

    for pred, gt in zip(parsed_preds, ground_truths):
        if pred is not None and isinstance(pred, dict):
            norm_pred = canonical_normalize(pred)
            norm_gt = canonical_normalize(gt)
            if norm_pred == norm_gt:
                correct += 1
    return correct / total

def compute_field_level_metrics(
    parsed_preds: List[Optional[Dict[str, Any]]],
    ground_truths: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes micro Precision, Recall, F1, and Accuracy over all slot keys and values.
    Excludes top-level 'intent' to evaluate slot extraction performance independently.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_field_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gt in zip(parsed_preds, ground_truths):
        gt_fields = {k: v for k, v in gt.items() if k != "intent"}
        pred_fields = {k: v for k, v in pred.items() if k != "intent"} if (pred and isinstance(pred, dict)) else {}

        # Evaluate expected fields in GT
        for k, gt_val in gt_fields.items():
            norm_gt_val = canonical_normalize(gt_val)
            if k in pred_fields:
                norm_pred_val = canonical_normalize(pred_fields[k])
                if norm_pred_val == norm_gt_val:
                    total_tp += 1
                    per_field_stats[k]["tp"] += 1
                else:
                    total_fp += 1
                    total_fn += 1
                    per_field_stats[k]["fp"] += 1
                    per_field_stats[k]["fn"] += 1
            else:
                total_fn += 1
                per_field_stats[k]["fn"] += 1

        # Evaluate extra spurious fields in Pred
        for k in pred_fields:
            if k not in gt_fields:
                total_fp += 1
                per_field_stats[k]["fp"] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    field_breakdown = {}
    for fld, counts in per_field_stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        field_breakdown[fld] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn
        }

    return {
        "field_precision": round(precision, 4),
        "field_recall": round(recall, 4),
        "field_f1": round(f1, 4),
        "field_accuracy": round(accuracy, 4),
        "field_breakdown": field_breakdown
    }

def evaluate_predictions(
    raw_outputs: List[str],
    ground_truths: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Optional[Dict[str, Any]]]]:
    """
    Runs full evaluation suite on a set of generated strings vs ground truths.
    """
    raw_validity, ext_validity, parsed_preds = compute_json_validity(raw_outputs)
    intent_acc = compute_intent_accuracy(parsed_preds, ground_truths)
    exact_match = compute_exact_match(parsed_preds, ground_truths)
    field_metrics = compute_field_level_metrics(parsed_preds, ground_truths)

    summary = {
        "total_samples": len(ground_truths),
        "raw_json_validity_rate": round(raw_validity, 4),
        "extracted_json_validity_rate": round(ext_validity, 4),
        "intent_accuracy": round(intent_acc, 4),
        "exact_match_accuracy": round(exact_match, 4),
        "field_precision": field_metrics["field_precision"],
        "field_recall": field_metrics["field_recall"],
        "field_f1": field_metrics["field_f1"],
        "field_accuracy": field_metrics["field_accuracy"],
        "field_breakdown": field_metrics["field_breakdown"]
    }

    return summary, parsed_preds
