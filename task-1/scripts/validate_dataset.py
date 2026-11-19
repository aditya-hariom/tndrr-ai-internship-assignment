#!/usr/bin/env python3
"""
scripts/validate_dataset.py

Audits the generated datasets (train.jsonl, validation.jsonl, test.jsonl)
against strict ML engineering standards:
1. Split sizes & ratio verification (ensuring test >= 50).
2. JSON integrity & ChatML message structure validation.
3. Strict schema compliance for every ground truth JSON object.
4. Duplicate detection (intra-split and cross-split).
5. Data leakage detection (exact match & n-gram lexical overlap across splits).
6. Missing slot (null value) distribution and edge case coverage.
7. Intent distribution and token length profiling.
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Set, Tuple

REQUIRED_SPLITS = {
    "train.jsonl": 1280,
    "validation.jsonl": 160,
    "test.jsonl": 160
}

VALID_INTENTS = {"create_reminder", "schedule_meeting", "set_timer", "log_expense", "unknown"}

VALID_TIME_REGEX = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

def validate_schema(sample_id: str, obj: Dict[str, Any]) -> List[str]:
    """Validates an individual ground_truth object against the intent schema."""
    errors = []
    if "intent" not in obj:
        return [f"[{sample_id}] Missing 'intent' field"]

    intent = obj["intent"]
    if intent not in VALID_INTENTS:
        return [f"[{sample_id}] Unknown intent: {intent}"]

    if intent == "create_reminder":
        expected_keys = {"intent", "task", "date", "time", "priority"}
        if set(obj.keys()) != expected_keys:
            errors.append(f"[{sample_id}] reminder keys mismatch. Found: {set(obj.keys())}, Expected: {expected_keys}")
        if not isinstance(obj.get("task"), str) or not obj.get("task").strip():
            errors.append(f"[{sample_id}] 'task' must be a non-empty string")
        if obj.get("date") is not None and not isinstance(obj["date"], str):
            errors.append(f"[{sample_id}] 'date' must be string or null")
        if obj.get("time") is not None:
            if not isinstance(obj["time"], str) or not VALID_TIME_REGEX.match(obj["time"]):
                errors.append(f"[{sample_id}] 'time' must be HH:MM format or null, got: {obj['time']}")
        if obj.get("priority") is not None and obj["priority"] not in {"low", "medium", "high"}:
            errors.append(f"[{sample_id}] 'priority' must be low/medium/high/null, got: {obj['priority']}")

    elif intent == "schedule_meeting":
        expected_keys = {"intent", "title", "participants", "date", "time", "duration_minutes"}
        if set(obj.keys()) != expected_keys:
            errors.append(f"[{sample_id}] meeting keys mismatch. Found: {set(obj.keys())}, Expected: {expected_keys}")
        if not isinstance(obj.get("title"), str) or not obj.get("title").strip():
            errors.append(f"[{sample_id}] 'title' must be non-empty string")
        parts = obj.get("participants")
        if not isinstance(parts, list) or len(parts) == 0 or not all(isinstance(p, str) for p in parts):
            errors.append(f"[{sample_id}] 'participants' must be non-empty list of strings")
        if obj.get("date") is not None and not isinstance(obj["date"], str):
            errors.append(f"[{sample_id}] 'date' must be string or null")
        if obj.get("time") is not None:
            if not isinstance(obj["time"], str) or not VALID_TIME_REGEX.match(obj["time"]):
                errors.append(f"[{sample_id}] 'time' must be HH:MM format or null, got: {obj['time']}")
        if obj.get("duration_minutes") is not None and (not isinstance(obj["duration_minutes"], int) or obj["duration_minutes"] <= 0):
            errors.append(f"[{sample_id}] 'duration_minutes' must be positive integer or null")

    elif intent == "set_timer":
        expected_keys = {"intent", "duration_seconds", "label"}
        if set(obj.keys()) != expected_keys:
            errors.append(f"[{sample_id}] timer keys mismatch. Found: {set(obj.keys())}, Expected: {expected_keys}")
        if not isinstance(obj.get("duration_seconds"), int) or obj["duration_seconds"] <= 0:
            errors.append(f"[{sample_id}] 'duration_seconds' must be positive integer")
        if obj.get("label") is not None and not isinstance(obj["label"], str):
            errors.append(f"[{sample_id}] 'label' must be string or null")

    elif intent == "log_expense":
        expected_keys = {"intent", "amount", "currency", "category", "merchant"}
        if set(obj.keys()) != expected_keys:
            errors.append(f"[{sample_id}] expense keys mismatch. Found: {set(obj.keys())}, Expected: {expected_keys}")
        amount = obj.get("amount")
        if not isinstance(amount, (int, float)) or amount <= 0:
            errors.append(f"[{sample_id}] 'amount' must be positive number")
        curr = obj.get("currency")
        if not isinstance(curr, str) or len(curr) != 3 or not curr.isupper():
            errors.append(f"[{sample_id}] 'currency' must be 3-letter ISO code, got: {curr}")
        if obj.get("category") is not None and not isinstance(obj["category"], str):
            errors.append(f"[{sample_id}] 'category' must be string or null")
        if obj.get("merchant") is not None and not isinstance(obj["merchant"], str):
            errors.append(f"[{sample_id}] 'merchant' must be string or null")

    elif intent == "unknown":
        expected_keys = {"intent", "error", "raw_query"}
        if set(obj.keys()) != expected_keys:
            errors.append(f"[{sample_id}] unknown keys mismatch. Found: {set(obj.keys())}, Expected: {expected_keys}")
        if obj.get("error") not in {"insufficient_information", "unsupported_intent"}:
            errors.append(f"[{sample_id}] 'error' must be insufficient_information or unsupported_intent")
        if not isinstance(obj.get("raw_query"), str) or not obj.get("raw_query").strip():
            errors.append(f"[{sample_id}] 'raw_query' must be non-empty string")

    return errors

def extract_user_query(sample: Dict[str, Any]) -> str:
    """Extracts normalized user utterance text from messages."""
    for m in sample.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "").strip().lower()
    return ""

def get_character_ngrams(text: str, n: int = 4) -> Set[str]:
    clean = re.sub(r"[^\w\s]", "", text)
    tokens = clean.split()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def inspect_datasets(data_dir: Path) -> bool:
    print("=" * 80)
    print("STARTING DATASET INTEGRITY & LEAKAGE AUDIT")
    print("=" * 80)

    has_errors = False
    split_records: Dict[str, List[Dict[str, Any]]] = {}
    split_queries: Dict[str, Set[str]] = {}
    split_ngrams: Dict[str, Set[str]] = {}

    # 1. File existence & line parsing
    for filename, expected_count in REQUIRED_SPLITS.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"[FAIL] Missing file: {filepath}")
            return False

        records = []
        queries = set()
        schema_errors = []
        intra_duplicates = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    schema_errors.append(f"Line {line_idx}: Invalid JSON ({e})")
                    continue

                records.append(data)
                q = extract_user_query(data)
                if q in queries:
                    intra_duplicates.append((data.get("id", f"line_{line_idx}"), q))
                queries.add(q)

                # Validate ChatML structure
                messages = data.get("messages", [])
                if len(messages) != 3 or [m["role"] for m in messages] != ["system", "user", "assistant"]:
                    schema_errors.append(f"Sample {data.get('id')}: Malformed messages list")
                
                # Validate Assistant JSON parsability
                try:
                    parsed_assistant = json.loads(messages[2]["content"])
                    if parsed_assistant != data.get("ground_truth"):
                        schema_errors.append(f"Sample {data.get('id')}: assistant content does not match ground_truth")
                except Exception as e:
                    schema_errors.append(f"Sample {data.get('id')}: assistant content is not valid JSON ({e})")

                # Validate strict intent schema
                gt_errors = validate_schema(data.get("id", f"line_{line_idx}"), data.get("ground_truth", {}))
                schema_errors.extend(gt_errors)

        split_records[filename] = records
        split_queries[filename] = queries

        # Build 4-grams for lexical leakage detection
        all_ngrams = set()
        for q in queries:
            all_ngrams.update(get_character_ngrams(q, n=4))
        split_ngrams[filename] = all_ngrams

        # Report split metrics
        actual_count = len(records)
        print(f"\nAudit for: {filename}")
        print(f"  * Total Samples: {actual_count} (Expected: {expected_count})")
        if actual_count != expected_count:
            print(f"  [FAIL] Count mismatch: expected {expected_count}, got {actual_count}")
            has_errors = True
        else:
            print(f"  [PASS] Sample count matches specification.")

        if actual_count < 50 and "test" in filename:
            print(f"  [FAIL] Test set has fewer than 50 examples!")
            has_errors = True

        print(f"  * Unique Queries: {len(queries)}")
        if intra_duplicates:
            print(f"  [FAIL] Found {len(intra_duplicates)} intra-split duplicate queries!")
            for sample_id, dup_q in intra_duplicates[:3]:
                print(f"     - Duplicate: '{dup_q}' ({sample_id})")
            has_errors = True
        else:
            print(f"  [PASS] Zero intra-split duplicate queries.")

        if schema_errors:
            print(f"  [FAIL] Found {len(schema_errors)} schema errors!")
            for err in schema_errors[:5]:
                print(f"     - {err}")
            has_errors = True
        else:
            print(f"  [PASS] 100% of samples satisfy the predefined JSON schema.")

    # 2. Cross-Split Leakage Check
    print("\n" + "-" * 50)
    print("Cross-Split Leakage Analysis")
    print("-" * 50)
    
    splits = list(REQUIRED_SPLITS.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_queries[s1].intersection(split_queries[s2])
            print(f"  * Exact Query Overlap between '{s1}' and '{s2}': {len(overlap)} samples")
            if overlap:
                print(f"    [FAIL] Data leakage detected! Common queries: {list(overlap)[:3]}")
                has_errors = True
            else:
                print(f"    [PASS] Zero query leakage between {s1} and {s2}.")

            # 4-gram overlap percentage
            ng1 = split_ngrams[s1]
            ng2 = split_ngrams[s2]
            ngram_overlap = ng1.intersection(ng2)
            jaccard = len(ngram_overlap) / max(1, len(ng1.union(ng2)))
            print(f"    * 4-gram Jaccard overlap index: {jaccard:.4f}")

    # 3. Intent Distribution & Missing Field Statistics
    print("\n" + "-" * 50)
    print("Dataset Intent Distribution & Edge-Case Coverage")
    print("-" * 50)

    for filename, records in split_records.items():
        intent_counts = Counter(r["intent"] for r in records)
        total = len(records)
        print(f"\nDistribution for {filename} (N={total}):")
        for intent, cnt in intent_counts.most_common():
            pct = (cnt / total) * 100
            print(f"  - {intent:<20}: {cnt:>4} ({pct:5.1f}%)")

        # Missing optional fields analysis
        null_counts = defaultdict(int)
        field_totals = defaultdict(int)
        for r in records:
            gt = r["ground_truth"]
            intent = gt["intent"]
            if intent == "create_reminder":
                for fld in ["date", "time", "priority"]:
                    field_totals[f"reminder.{fld}"] += 1
                    if gt.get(fld) is None:
                        null_counts[f"reminder.{fld}"] += 1
            elif intent == "schedule_meeting":
                for fld in ["date", "time", "duration_minutes"]:
                    field_totals[f"meeting.{fld}"] += 1
                    if gt.get(fld) is None:
                        null_counts[f"meeting.{fld}"] += 1
            elif intent == "set_timer":
                field_totals["timer.label"] += 1
                if gt.get("label") is None:
                    null_counts["timer.label"] += 1
            elif intent == "log_expense":
                for fld in ["category", "merchant"]:
                    field_totals[f"expense.{fld}"] += 1
                    if gt.get(fld) is None:
                        null_counts[f"expense.{fld}"] += 1

        print(f"  * Optional Slot Null Rates (Missing Info Edge Cases):")
        for fld, tot in field_totals.items():
            n_null = null_counts[fld]
            print(f"    - {fld:<20}: {n_null}/{tot} null ({n_null/tot*100:4.1f}%)")

    print("\n" + "=" * 80)
    if not has_errors:
        print("[SUCCESS] All dataset integrity, schema validity, and zero-leakage checks PASSED!")
    else:
        print("[FAILURE] Some audit checks failed. Review errors above.")
    print("=" * 80)
    return not has_errors

if __name__ == "__main__":
    data_path = Path("data")
    success = inspect_datasets(data_path)
    sys.exit(0 if success else 1)
