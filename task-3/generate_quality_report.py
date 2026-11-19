import json
import os

with open("d:/Ai/Task 3/benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

m4_runs = data["llama3.2:3b"]["runs"]
m8_runs = data["llama3.2:3b-instruct-q8_0"]["runs"]

output_path = "d:/Ai/Task 3/quality_comparison.md"
with open(output_path, "w", encoding="utf-8") as out:
    out.write("# Stage 4: Qualitative Output Comparison (4-bit vs 8-bit)\n\n")
    out.write("This document contains the side-by-side output comparison for all 10 evaluation prompts across the two configurations of Meta Llama 3.2 3B.\n\n")
    out.write("---\n\n")
    
    for i in range(len(m4_runs)):
        p_id = m4_runs[i]["prompt_id"]
        cat = m4_runs[i]["category"]
        prompt = m4_runs[i]["prompt"]
        r4 = m4_runs[i]["metrics"]["response"]
        r8 = m8_runs[i]["metrics"]["response"]
        
        out.write(f"## Prompt {p_id}: {cat}\n\n")
        out.write(f"**Prompt:**\n> {prompt}\n\n")
        
        out.write("### Configuration A: 4-bit (`llama3.2:3b`)\n")
        out.write(f"```text\n{r4.strip()}\n```\n\n")
        
        out.write("### Configuration B: 8-bit (`llama3.2:3b-instruct-q8_0`)\n")
        out.write(f"```text\n{r8.strip()}\n```\n\n")
        out.write("---\n\n")

print(f"Generated {output_path}")
