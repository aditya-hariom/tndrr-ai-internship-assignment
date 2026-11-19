# Task 3: Running an Open LLM Locally & Optimization

[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%202050%20(4GB)-76B900?logo=nvidia)](https://www.nvidia.com)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20x64-0078D6?logo=windows)](https://microsoft.com)
[![Engine](https://img.shields.io/badge/Inference-Ollama%20v0.40.16-black)](https://ollama.com)
[![Model](https://img.shields.io/badge/Model-Meta%20Llama%203.2%20(3B)-blue)](https://ai.meta.com/llama)

This repository contains the complete experimental benchmarking suite, qualitative evaluation dataset, and academic technical report for **Task 3: Running a model locally & optimisation**.

The experiment empirically evaluates **Meta's Llama 3.2 3B Instruct** across two quantization levels (**4-bit `Q4_K_M`** vs. **8-bit `Q8_0`**) on a consumer-grade laptop equipped with an **NVIDIA GeForce RTX 2050 (4 GB VRAM)** and an **AMD Ryzen 7 5800HS CPU**.

---

## 📌 Executive Summary & Key Results

| Metric | Configuration A: 4-bit (`Q4_K_M`) | Configuration B: 8-bit (`Q8_0`) | Delta / Factor |
| :--- | :---: | :---: | :---: |
| **Model On-Disk Size** | 2.0 GB | 3.4 GB | +70% size |
| **Avg Generation Speed** | **39.43 tokens/sec** | **14.18 tokens/sec** | **4-bit is 2.78x faster** |
| **Avg Prompt Processing Speed** | **343.50 tokens/sec** | **221.18 tokens/sec** | **4-bit is 1.55x faster** |
| **Initial Cold Load Time** | **6.14 seconds** | **7.31 seconds** | 1.17s faster load |
| **Peak VRAM Allocated** | 2,327 MiB | 2,351 MiB | VRAM Saturated |
| **Peak System RAM Allocated** | 10,780 MB | 11,792 MB | +1,012 MB offload to RAM |
| **Hardware Offload Ratio** | **80% GPU / 20% CPU** | **54% GPU / 46% CPU** | Massive memory cliff |
| **Qualitative Score (10 Prompts)** | **50 / 50 (100%)** | **48 / 50 (96%)** | Parity; 4-bit better on strict JSON |

### The Core Finding: The "Memory Cliff"
Because the 8-bit model required ~4.4 GB of total runtime memory (weights + context buffer), it exceeded the 4 GB physical VRAM boundary of the RTX 2050. The runtime engine (**Ollama / llama.cpp**) was forced to offload **46% of model layers to system RAM**, introducing substantial PCIe bus and DDR4 memory latency. Consequently, **4-bit quantization yielded a 2.78x speedup** with zero loss in reasoning, mathematics, or coding performance.

---

## 📂 Repository Structure

```text
├── README.md                  # Main repository documentation & quickstart
├── TASK_3_FINAL_REPORT.md     # Complete academic report (Aim, Methodology, Theory, 20-User Plan)
├── benchmark.py               # Automated benchmark harness (queries Ollama API, collects VRAM/RAM)
├── benchmark_results.json     # Raw experimental telemetry (timestamps, token counts, timings, outputs)
├── benchmark_summary.md       # Markdown summary tables generated directly from raw telemetry
├── quality_comparison.md      # Side-by-side verbatim outputs of all 10 evaluation prompts
├── generate_quality_report.py # Script used to compile raw JSON outputs into formatted comparison markdown
└── verify_benchmark.py        # Integrity validation script checking report claims against raw JSON
```

---

## 🔬 Evaluated Prompts

The benchmark tests both models across 10 distinct functional domains:
1. **General Knowledge:** Photosynthesis mechanisms and ecological importance.
2. **Reasoning:** Algebraic bat-and-ball problem ($x + (x + 1.00) = 1.10$).
3. **Mathematics:** Quadratic equation solving ($2x^2 - 7x + 3 = 0$) using quadratic formula.
4. **Summarisation:** Constrained 3-bullet summary of multi-industry AI impacts.
5. **Instruction Following:** Strict JSON array generation with explicit keys and zero preamble.
6. **Coding:** Python duplicate detection function with type hints and docstrings.
7. **Creative Writing:** Atmospheric micro-fiction under 150 words.
8. **Information Extraction:** Multi-field extraction from an unstructured meeting email.
9. **Factual Accuracy:** Speed of light, Eddington's 1919 solar eclipse, and General Relativity.
10. **Challenging Multi-Step Task:** Complete 3-day budget travel itinerary for Kyoto, Japan.

To view the complete verbatim outputs from both configurations, see [quality_comparison.md](quality_comparison.md).

---

## 🚀 How to Reproduce

### 1. Prerequisites
* **Operating System:** Windows 10/11, Linux, or macOS.
* **Python:** Python 3.10+ (standard library only; no pip dependencies required).
* **Ollama:** Install from [ollama.com](https://ollama.com).

### 2. Pull Model Weights
In PowerShell or bash, download both model configurations:
```bash
# Configuration A: 4-bit medium quantization (~2.0 GB)
ollama pull llama3.2:3b

# Configuration B: 8-bit quantization (~3.4 GB)
ollama pull llama3.2:3b-instruct-q8_0
```

### 3. Run Benchmark
Execute the automated benchmarking suite:
```bash
python benchmark.py
```
This runs all 10 prompts sequentially, measures hardware metrics via `nvidia-smi` and system memory APIs, and outputs:
* `benchmark_results.json` (full raw telemetry)
* `benchmark_summary.md` (summary tables)

### 4. Verify Results Integrity
Run the built-in validation script to verify that report tables match raw JSON measurements:
```bash
python verify_benchmark.py
```

---

## 📑 Full Academic Report
The comprehensive assignment submission report is located in **[TASK_3_FINAL_REPORT.md](TASK_3_FINAL_REPORT.md)** and includes:
* **Section 1–3:** Aim, environment specifications, and measurement protocol.
* **Section 4–6:** Master benchmark tables, per-prompt throughput breakdowns, and hardware offload analysis.
* **Section 7:** Theoretical explanations with intuitive analogies:
  * Key-Value (KV) cache architecture and $O(N)$ memory growth.
  * Weight quantization mathematics (FP16 vs INT8 vs INT4) and outlier weight preservation.
  * Discrepancy between static on-disk file size and runtime memory requirements.
* **Section 8:** Back-of-the-envelope capacity plan for serving **20 concurrent users on a single 24 GB GPU**:
  * PagedAttention memory calculation ($0.125\text{ MiB/token}$).
  * Detailed VRAM budget table.
  * Continuous iteration-level batching and latency bottlenecks.
* **Section 9:** Study limitations and conclusions.

---

## 📄 License
This project is prepared for educational and academic coursework in Artificial Intelligence & Large Language Models.
