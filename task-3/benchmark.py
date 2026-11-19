import urllib.request
import json
import time
import subprocess
import ctypes
import os

# Define Windows memory status query via ctypes (no external pip dependencies needed)
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

def get_system_ram_mb():
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        used_bytes = stat.ullTotalPhys - stat.ullAvailPhys
        return round(used_bytes / (1024 * 1024), 2)
    except Exception as e:
        return 0.0

def get_vram_usage_mb():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(res.stdout.strip().split("\n")[0])
    except Exception as e:
        return 0.0

def unload_model(model_name):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
    time.sleep(2)

def query_ollama(model_name, prompt):
    req_body = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "seed": 42  # Fixed seed for fair reproducibility
        }
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
        
    vram_mb = get_vram_usage_mb()
    ram_mb = get_system_ram_mb()
    
    # Durations are returned in nanoseconds
    load_duration_s = result.get("load_duration", 0) / 1e9
    prompt_eval_count = result.get("prompt_eval_count", 0)
    prompt_eval_duration_s = result.get("prompt_eval_duration", 0) / 1e9
    prompt_eval_rate = (prompt_eval_count / prompt_eval_duration_s) if prompt_eval_duration_s > 0 else 0.0
    
    eval_count = result.get("eval_count", 0)
    eval_duration_s = result.get("eval_duration", 0) / 1e9
    eval_rate = (eval_count / eval_duration_s) if eval_duration_s > 0 else 0.0
    
    total_duration_s = result.get("total_duration", 0) / 1e9
    
    return {
        "response": result.get("response", ""),
        "load_duration_s": round(load_duration_s, 4),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_s": round(prompt_eval_duration_s, 4),
        "prompt_eval_rate": round(prompt_eval_rate, 2),
        "eval_count": eval_count,
        "eval_duration_s": round(eval_duration_s, 4),
        "eval_rate": round(eval_rate, 2),
        "total_duration_s": round(total_duration_s, 4),
        "vram_mb": vram_mb,
        "ram_mb": ram_mb
    }

PROMPTS = [
    {
        "id": 1,
        "category": "General Knowledge",
        "prompt": "Explain how photosynthesis works in plants and why it is essential for life on Earth."
    },
    {
        "id": 2,
        "category": "Reasoning",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your step-by-step reasoning."
    },
    {
        "id": 3,
        "category": "Mathematics",
        "prompt": "Solve the quadratic equation 2x^2 - 7x + 3 = 0 using the quadratic formula. Show all calculation steps."
    },
    {
        "id": 4,
        "category": "Summarisation",
        "prompt": "Summarize the following passage in exactly three bullet points:\n'Artificial intelligence has revolutionized numerous industries over the past decade. In healthcare, machine learning algorithms assist radiologists in identifying tumors with precision and accelerate drug discovery. In finance, automated algorithmic trading handles high-frequency market shifts while fraud detection models monitor millions of transactions in real time. In transportation, autonomous driving systems combine computer vision, radar, and lidar to navigate complex environments, reducing human error which accounts for over ninety percent of automobile collisions.'"
    },
    {
        "id": 5,
        "category": "Instruction Following",
        "prompt": "List 5 countries in Africa. For each country, state its capital city and official language. Format your output strictly as a JSON array of objects with keys: 'country', 'capital', and 'language'. Do not include any conversational preamble or markdown backticks."
    },
    {
        "id": 6,
        "category": "Coding",
        "prompt": "Write a Python function named `find_first_duplicate(nums)` that takes a list of integers and returns the first number that appears more than once. If no duplicate exists, return `None`. Include type hints and time/space complexity analysis in docstrings."
    },
    {
        "id": 7,
        "category": "Creative Writing",
        "prompt": "Write a short story (under 150 words) about a lighthouse keeper who discovers that the light is being answered by something deep underwater."
    },
    {
        "id": 8,
        "category": "Information Extraction",
        "prompt": "From the following email, extract: (1) Meeting date and time, (2) Meeting location/link, (3) Key agenda items, and (4) Action items assigned to Sarah.\nEmail: 'Hi Team, Let us meet on Thursday, Nov 26 at 3:30 PM EST in Conference Room B (or join via meet.google.com/xyz-abc). We will review the Q4 marketing budget, finalize the keynote speaker list, and vote on the new logo proposal. Sarah, please bring the updated financial projection sheets and send out the speaker bios by Wednesday noon. Best, Dave'"
    },
    {
        "id": 9,
        "category": "Factual Accuracy",
        "prompt": "What is the speed of light in a vacuum, who experimentally verified the bending of light by gravity during a solar eclipse in 1919, and what theory did this confirm?"
    },
    {
        "id": 10,
        "category": "Challenging Multi-Step Task",
        "prompt": "Plan a 3-day itinerary for a budget-conscious solo traveler visiting Kyoto, Japan. For each day, include: (1) morning activity, (2) affordable lunch recommendation with typical cost in Yen, (3) afternoon activity, (4) evening activity, and (5) one specific transit tip to save money. Conclude with an estimated total budget in USD."
    }
]

MODELS = [
    {
        "name": "llama3.2:3b",
        "label": "Llama 3.2 3B (4-bit Q4_K_M)",
        "quant": "4-bit (Q4_K_M)"
    },
    {
        "name": "llama3.2:3b-instruct-q8_0",
        "label": "Llama 3.2 3B (8-bit Q8_0)",
        "quant": "8-bit (Q8_0)"
    }
]

def main():
    print("=== STARTING LLM BENCHMARK EXPERIMENT ===")
    results = {}
    
    for model_info in MODELS:
        m_name = model_info["name"]
        m_label = model_info["label"]
        print(f"\n==========================================")
        print(f"Testing Model: {m_label}")
        print(f"==========================================")
        
        # Unload any active models to start fresh
        unload_model("llama3.2:3b")
        unload_model("llama3.2:3b-instruct-q8_0")
        time.sleep(2)
        
        model_runs = []
        
        for p in PROMPTS:
            p_id = p["id"]
            cat = p["category"]
            prompt_text = p["prompt"]
            print(f"[{m_name}] Running Prompt {p_id}/10 ({cat})...", end="", flush=True)
            
            t0 = time.time()
            res = query_ollama(m_name, prompt_text)
            elapsed = time.time() - t0
            
            run_data = {
                "prompt_id": p_id,
                "category": cat,
                "prompt": prompt_text,
                "metrics": res
            }
            model_runs.append(run_data)
            print(f" Done ({res['eval_rate']} tok/s, {res['eval_count']} tokens in {round(elapsed, 2)}s)")
            time.sleep(1) # brief cool-off
            
        results[m_name] = {
            "info": model_info,
            "runs": model_runs
        }
        
    # Save raw JSON results
    json_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Raw results saved to {json_path}")
    
    # Calculate Summary Statistics
    summary_path = os.path.join(os.path.dirname(__file__), "benchmark_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# LLM Benchmark Experiment Summary\n\n")
        f.write("## Overall Average Metrics\n\n")
        f.write("| Configuration | Quantisation | Avg Generation Speed (tok/s) | Avg Prompt Eval Speed (tok/s) | Initial Load Time (s) | Peak VRAM (MiB) | Peak RAM (MB) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        
        for m_name, data in results.items():
            info = data["info"]
            runs = data["runs"]
            
            eval_rates = [r["metrics"]["eval_rate"] for r in runs]
            prompt_eval_rates = [r["metrics"]["prompt_eval_rate"] for r in runs]
            initial_load = runs[0]["metrics"]["load_duration_s"]
            peak_vram = max([r["metrics"]["vram_mb"] for r in runs])
            peak_ram = max([r["metrics"]["ram_mb"] for r in runs])
            
            avg_eval_rate = round(sum(eval_rates) / len(eval_rates), 2)
            avg_prompt_eval_rate = round(sum(prompt_eval_rates) / len(prompt_eval_rates), 2)
            
            f.write(f"| {info['label']} | {info['quant']} | {avg_eval_rate} | {avg_prompt_eval_rate} | {initial_load} | {peak_vram} | {peak_ram} |\n")
            
        f.write("\n\n## Per-Prompt Detailed Metrics\n\n")
        f.write("| Prompt # | Category | 4-bit Gen (tok/s) | 8-bit Gen (tok/s) | Speed Ratio (4-bit / 8-bit) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: |\n")
        
        m4_runs = results["llama3.2:3b"]["runs"]
        m8_runs = results["llama3.2:3b-instruct-q8_0"]["runs"]
        
        for i in range(len(PROMPTS)):
            p_id = PROMPTS[i]["id"]
            cat = PROMPTS[i]["category"]
            r4 = m4_runs[i]["metrics"]["eval_rate"]
            r8 = m8_runs[i]["metrics"]["eval_rate"]
            ratio = round(r4 / r8, 2) if r8 > 0 else "N/A"
            f.write(f"| {p_id} | {cat} | {r4} | {r8} | {ratio}x |\n")
            
    print(f"[SUCCESS] Markdown summary saved to {summary_path}")

if __name__ == "__main__":
    main()
