import json
from pathlib import Path

def extract_failures():
    preds_path = Path("results/finetuned_predictions.jsonl")
    with open(preds_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    failures = [r for r in records if not r.get("exact_match")]
    print(f"Total test samples: {len(records)}")
    print(f"Total exact match failures: {len(failures)} ({len(failures)/len(records)*100:.2f}%)")

    # Group by error types
    intent_mismatches = [r for r in failures if not r.get("intent_match")]
    slot_mismatches = [r for r in failures if r.get("intent_match")]

    print(f"Intent mismatches: {len(intent_mismatches)}")
    print(f"Slot extraction mismatches: {len(slot_mismatches)}\n")

    print("=" * 80)
    print("SAMPLE INTENT MISMATCHES:")
    for r in intent_mismatches[:5]:
        print(f"ID: {r['id']}")
        print(f"Query: {r['user_query']}")
        print(f"Pred:  {r['parsed_prediction']}")
        print(f"GT:    {r['ground_truth']}")
        print("-" * 50)

    print("\n" + "=" * 80)
    print("SAMPLE SLOT EXTRACTION MISMATCHES:")
    for r in slot_mismatches[:8]:
        print(f"ID: {r['id']}")
        print(f"Query: {r['user_query']}")
        print(f"Pred:  {r['parsed_prediction']}")
        print(f"GT:    {r['ground_truth']}")
        print("-" * 50)

if __name__ == "__main__":
    extract_failures()
