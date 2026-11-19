import json
import re

with open("d:/Ai/Task 3/benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("="*60)
print("VERIFYING BENCHMARK DATA AGAINST JSON")
print("="*60)

stats = {}
for model_key in ["llama3.2:3b", "llama3.2:3b-instruct-q8_0"]:
    mdata = data[model_key]
    runs = mdata["runs"]
    eval_rates = [r["metrics"]["eval_rate"] for r in runs]
    prompt_eval_rates = [r["metrics"]["prompt_eval_rate"] for r in runs]
    load_times = [r["metrics"]["load_duration_s"] for r in runs]
    vram_usages = [r["metrics"]["vram_mb"] for r in runs]
    ram_usages = [r["metrics"]["ram_mb"] for r in runs]
    
    stats[model_key] = {
        "count": len(runs),
        "avg_eval_rate": round(sum(eval_rates) / len(eval_rates), 2),
        "avg_prompt_eval_rate": round(sum(prompt_eval_rates) / len(prompt_eval_rates), 2),
        "init_load": round(runs[0]["metrics"]["load_duration_s"], 4),
        "peak_vram": max(vram_usages),
        "peak_ram": max(ram_usages),
        "eval_rates": eval_rates,
        "prompt_eval_rates": prompt_eval_rates
    }
    print(f"\nModel: {model_key}")
    print(f"  Runs count: {stats[model_key]['count']}")
    print(f"  Avg Gen Speed: {stats[model_key]['avg_eval_rate']} tok/s")
    print(f"  Avg Prompt Speed: {stats[model_key]['avg_prompt_eval_rate']} tok/s")
    print(f"  Initial Load Time: {stats[model_key]['init_load']} s")
    print(f"  Peak VRAM: {stats[model_key]['peak_vram']} MiB")
    print(f"  Peak RAM: {stats[model_key]['peak_ram']} MB")

with open("d:/Ai/Task 3/TASK_3_FINAL_REPORT.md", "r", encoding="utf-8") as f:
    report = f.read()

print("\n" + "="*60)
print("CHECKING REPORT NUMBERS")
print("="*60)

# Check 4-bit numbers in report
c4_gen = "39.43" in report
c4_prompt = "343.50" in report or "343.5" in report
c4_load = "6.14" in report
c4_vram = "2,327" in report or "2327" in report
c4_ram = "10,780" in report or "10780" in report

print(f"Config A (4-bit) Avg Gen Speed (39.43): {'MATCH' if c4_gen else 'MISMATCH'}")
print(f"Config A (4-bit) Avg Prompt Speed (343.50): {'MATCH' if c4_prompt else 'MISMATCH'}")
print(f"Config A (4-bit) Load Time (6.14): {'MATCH' if c4_load else 'MISMATCH'}")
print(f"Config A (4-bit) Peak VRAM (2,327): {'MATCH' if c4_vram else 'MISMATCH'}")
print(f"Config A (4-bit) Peak RAM (10,780): {'MATCH' if c4_ram else 'MISMATCH'}")

# Check 8-bit numbers in report
c8_gen = "14.18" in report
c8_prompt = "221.18" in report
c8_load = "7.31" in report
c8_vram = "2,351" in report or "2351" in report
c8_ram = "11,792" in report or "11792" in report

print(f"Config B (8-bit) Avg Gen Speed (14.18): {'MATCH' if c8_gen else 'MISMATCH'}")
print(f"Config B (8-bit) Avg Prompt Speed (221.18): {'MATCH' if c8_prompt else 'MISMATCH'}")
print(f"Config B (8-bit) Load Time (7.31): {'MATCH' if c8_load else 'MISMATCH'}")
print(f"Config B (8-bit) Peak VRAM (2,351): {'MATCH' if c8_vram else 'MISMATCH'}")
print(f"Config B (8-bit) Peak RAM (11,792): {'MATCH' if c8_ram else 'MISMATCH'}")

# Check ratio
ratio = round(stats["llama3.2:3b"]["avg_eval_rate"] / stats["llama3.2:3b-instruct-q8_0"]["avg_eval_rate"], 2)
print(f"Calculated Speedup Ratio: {ratio}x")
print(f"Report mentions 2.78x: {'MATCH' if '2.78' in report else 'MISMATCH'}")

# Check individual prompts
print("\n" + "="*60)
print("CHECKING PER-PROMPT GENERATION SPEEDS")
print("="*60)
all_prompts_match = True
for i in range(10):
    r4 = stats["llama3.2:3b"]["eval_rates"][i]
    r8 = stats["llama3.2:3b-instruct-q8_0"]["eval_rates"][i]
    if str(r4) not in report:
        print(f"Prompt {i+1} 4-bit ({r4}) missing from report!")
        all_prompts_match = False
    if str(r8) not in report:
        print(f"Prompt {i+1} 8-bit ({r8}) missing from report!")
        all_prompts_match = False

if all_prompts_match:
    print("All 10 prompt generation speed values matched exactly in the report!")
